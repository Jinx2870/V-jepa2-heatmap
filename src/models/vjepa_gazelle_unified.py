"""
Unified V-JEPA2 + Gazelle wrapper model.

This file is intentionally *additive* (no other code changed) so it can be
integrated later by refactoring the training loop to interact with a single
model instance.

Design goals (future refactor):
  - Input is frames only (Dataset returns frames).
  - Training code calls ONE model, not encoder/predictor/gazelle separately.
  - Model internally performs:
      encoder: frames -> latent tokens
      predictor: latent tokens -> future tokens (masked-token prediction)
      gazelle: frames -> gaze/scene tokens

Current repo state:
  - V-JEPA pretraining uses MultiSeq wrappers and mask indices.
  - Gazelle code lives outside this repo (e.g., /data3/lg2/human_wm/gazelle).

This module is VIDEO-ONLY:
  - The dataset returns a single video tensor `frames` of shape [B, C, T, H, W]
  - In your setup H=W=256 (crop_size=256)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vision_transformer import VisionTransformer
from src.masks.utils import apply_masks
from src.models.predictor import VisionTransformerPredictor
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper


@dataclass(frozen=True)
class GazelleConfig:
    """
    Configuration for loading and running Gazelle.
    """

    checkpoint: str
    model_name: str = "gazelle_dinov2_vitb14_inout"
    python_path: Optional[str] = None  # path to the gazelle repo root
    device: str = "cuda"
    input_size: int = 224  # Gazelle部分先 resize 到 224
    max_batch_size: int = 64  # frames per forward batch (B*T for video)
    min_cuda_free_mb: int = 2048


class VJEPAGazelleUnifiedModel(nn.Module):
    """
    A single nn.Module that owns:
      - V-JEPA2 encoder (MultiSeq-wrapped VisionTransformer)
      - V-JEPA2 predictor (PredictorMultiSeqWrapper around VisionTransformerPredictor)
      - Gazelle scene-token extractor (eager init in __init__)

    Notes:
      - This model does NOT (yet) change training semantics; it just exposes
        unified APIs so train.py can be refactored later.
      - For Gazelle, we extract *scene tokens* (transformer output before heatmap head).
        These are often what downstream code calls "gazelle tokens" / "gaze tokens".
    """

    def __init__(
        self,
        # V-JEPA encoder config
        img_size: int = 256,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        model_name: str = "vit_base",
        # predictor config
        pred_depth: int = 6,
        pred_embed_dim: int = 384,
        pred_num_heads: Optional[int] = None,
        uniform_power: bool = False,
        use_sdpa: bool = False,
        use_rope: bool = False,
        use_silu: bool = False,
        use_pred_silu: bool = False,
        wide_silu: bool = True,
        use_activation_checkpointing: bool = False,
        use_mask_tokens: bool = False,
        num_mask_tokens: int = 2,
        zero_init_mask_tokens: bool = True,
        # Gazelle config
        gazelle_cfg: Optional[GazelleConfig] = None,
        # For quick smoke tests (and for environments without Gazelle deps / checkpoints),
        # you can skip Gazelle initialization entirely. Forward can still run encoder/predictor
        # as long as `use_gazelle_condition=False`.
        init_gazelle: bool = True,
    ):
        super().__init__()
        if init_gazelle and gazelle_cfg is None:
            raise ValueError("init_gazelle=True 时，gazelle_cfg 必须提供。")

        # -----------------------
        # V-JEPA encoder/predictor
        # -----------------------
        # Build encoder from the same factory pattern used elsewhere in this repo:
        # src/models/vision_transformer.py exposes constructors in __dict__ keyed by model_name.
        # We fall back to constructing VisionTransformer directly if model_name is missing.
        vit_ctor = getattr(__import__("src.models.vision_transformer", fromlist=[model_name]), model_name, None)
        if callable(vit_ctor):
            encoder_backbone = vit_ctor(
                img_size=img_size,
                patch_size=patch_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                uniform_power=uniform_power,
                use_sdpa=use_sdpa,
                use_silu=use_silu,
                wide_silu=wide_silu,
                use_activation_checkpointing=use_activation_checkpointing,
                use_rope=use_rope,
            )
        else:
            encoder_backbone = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                uniform_power=uniform_power,
                use_sdpa=use_sdpa,
                use_silu=use_silu,
                wide_silu=wide_silu,
                use_activation_checkpointing=use_activation_checkpointing,
                use_rope=use_rope,
            )

        self.encoder = MultiSeqWrapper(encoder_backbone)

        predictor_backbone = VisionTransformerPredictor(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=self.encoder.backbone.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=self.encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
            uniform_power=uniform_power,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=zero_init_mask_tokens,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
            use_silu=use_pred_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        # Wrap predictor so it can accept Gazelle condition tokens.
        self.predictor = _PredictorMultiSeqWrapperWithCondition(
            _VisionTransformerPredictorWithCondition(predictor_backbone)
        )

        # -----------------------
        # Gazelle
        # -----------------------
        self._gazelle_cfg = gazelle_cfg
        self._gazelle_model = None
        self._gazelle_device = torch.device("cpu")

        if init_gazelle:
            assert self._gazelle_cfg is not None

            # Ensure gazelle repo is on PYTHONPATH
            if self._gazelle_cfg.python_path:
                path = os.path.abspath(os.path.expanduser(self._gazelle_cfg.python_path))
                if os.path.isdir(path) and path not in sys.path:
                    sys.path.insert(0, path)
            env_path = os.environ.get("GAZELLE_REPO") or os.environ.get("GAZELLE_PYTHONPATH")
            if env_path:
                path = os.path.abspath(os.path.expanduser(env_path))
                if os.path.isdir(path) and path not in sys.path:
                    sys.path.insert(0, path)

            # 按 Gazelle 主模型的定义方式构建（但强制 in_size=224），这样 pos_embed/head_maps 的网格
            # 天然是 16x16（ViT-B/14: 224/14=16），无需任何插值。
            from gazelle.backbone import DinoV2Backbone  # type: ignore
            from gazelle.model import GazeLLE  # type: ignore

            name = (self._gazelle_cfg.model_name or "").lower()
            if "vitb14" in name:
                backbone_name = "dinov2_vitb14"
            elif "vitl14" in name:
                backbone_name = "dinov2_vitl14"
            else:
                raise ValueError(
                    f"不支持的 Gazelle model_name={self._gazelle_cfg.model_name}（仅支持 vitb14/vitl14 系列）"
                )

            inout = "inout" in name
            in_size = (int(self._gazelle_cfg.input_size), int(self._gazelle_cfg.input_size))

            backbone = DinoV2Backbone(backbone_name)
            _gazelle_model = GazeLLE(backbone, inout=inout, in_size=in_size)

            state = torch.load(self._gazelle_cfg.checkpoint, map_location="cpu", weights_only=True)
            # 由于我们把 in_size 改成 224，checkpoint 里可能带有 448 对应的 pos_embed（32x32）。
            # 为了保证 forward 与 “in_size=224 的主模型”完全一致，这里明确不用 checkpoint 的 pos_embed，
            # 使用当前模型按 224 自动生成的 pos_embed。
            state = dict(state)
            state.pop("pos_embed", None)
            _gazelle_model.load_gazelle_state_dict(state)
            _gazelle_model.eval()
            for p in _gazelle_model.parameters():
                p.requires_grad = False

            gazelle_dev = torch.device(self._gazelle_cfg.device)
            if gazelle_dev.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"GazelleConfig.device={self._gazelle_cfg.device}，但当前环境没有可用 CUDA")
            _gazelle_model.to(gazelle_dev)

            self._gazelle_model = _gazelle_model
            self._gazelle_device = gazelle_dev
        # Trainable projection from Gazelle scene-token dim -> predictor embed dim.
        # (Input feature dim is inferred at first use.)
        self._scene_proj = nn.LazyLinear(pred_embed_dim, bias=True)

        # cache these for convenience
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

    # ---------------------------------------------------------------------
    # Gazelle
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def gazelle_scene_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract Gazelle scene tokens from a video clip.

        Args:
            frames: torch Tensor of shape [B, C, T, H, W] (video only).
                    Values can be uint8 [0..255] or float in [0..1] or ImageNet-normalized floats.

        Returns:
            scene_tokens: torch Tensor of shape [B, T, HW, D_scene] on the model's device.
                         HW is Gazelle backbone grid (typically 16*16 for input_size=224, or 32*32 for 448).
        """
        if self._gazelle_model is None:
            raise RuntimeError(
                "Gazelle 未初始化：构造模型时请传 init_gazelle=True + gazelle_cfg，"
                "或者在 forward() 里设置 use_gazelle_condition=False 来跳过 Gazelle conditioning。"
            )

        if frames.dim() != 5:
            raise ValueError(f"Expected 5D frames tensor, got shape={tuple(frames.shape)}")

        if frames.size(1) != 3:
            raise ValueError(f"Expected channel dimension to be 3, got frames shape={tuple(frames.shape)}")

        B, C, T, H, W = frames.shape
        dev = self._gazelle_device

        # Flatten frames over time so Gazelle runs per-frame.
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = x.to(dev, non_blocking=dev.type == "cuda")

        # -----------------------------
        # 与 Gazelle 主模型的输入预处理一致（参考 gazelle/backbone.py:get_transform）：
        #   - ToTensor(): uint8 -> float in [0,1]
        #   - Normalize(mean,std)
        #   - Resize(in_size)
        # 注意：这里假设 dataset 提供的是“未做 ImageNet normalize”的原始像素（uint8 或 [0,1] float）。
        # -----------------------------
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            # 若看起来像 [0,255] 的 float，转换到 [0,1]
            if x.max() > 1.5:
                x = (x / 255.0).clamp(0.0, 1.0)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        target_size = int(self._gazelle_cfg.input_size)
        if x.shape[-2:] != (target_size, target_size):
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)

        # Run in batches (frames can be large: B*T).
        max_bs = int(self._gazelle_cfg.max_batch_size)
        outs: List[torch.Tensor] = []
        start = 0
        while start < x.size(0):
            end = min(x.size(0), start + max_bs)
            outs.append(self._gazelle_forward_scene_tokens(x[start:end]))
            start = end

        tokens = torch.cat(outs, dim=0)  # [B*T, HW, D]
        tokens = tokens.view(B, T, tokens.size(1), tokens.size(2))
        return tokens

    def _prepare_gazelle_condition_for_predictor(
        self, frames: torch.Tensor, scene_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert Gazelle scene tokens into predictor-aligned condition tokens.

        Returns:
            cond_full: [B, N_patches, D_pred] where N_patches = grid_depth * grid_h * grid_w
        """
        if self._gazelle_model is None:
            raise RuntimeError(
                "Gazelle 未初始化：无法生成 condition tokens。请传 init_gazelle=True，"
                "或者在 forward() 里设置 use_gazelle_condition=False。"
            )
        # Gazelle 只 forward 一次：外部把 scene_tokens 预先算好传进来。
        scene = scene_tokens if scene_tokens is not None else self.gazelle_scene_tokens(frames)
        # Move to the same device as predictor params.
        # (We keep Gazelle itself frozen; projection is trainable.)
        pred_dev = next(self.predictor.parameters()).device
        scene = scene.to(pred_dev, non_blocking=True)

        B, T_g, HW_g, D_scene = scene.shape
        grid_g = int(HW_g**0.5)
        if grid_g * grid_g != HW_g:
            raise ValueError(f"Gazelle HW={HW_g} is not a square grid.")

        # Target grid (V-JEPA tokens): H=W=img_size/patch_size
        grid_h = self.img_size // self.patch_size
        grid_w = self.img_size // self.patch_size

        # GazelleConfig.input_size=224 (ViT-B/14 → 16x16) 与 V-JEPA (256/16 → 16x16) 天然对齐。
        if (grid_g, grid_g) != (grid_h, grid_w):
            raise ValueError(
                f"Gazelle grid {grid_g}x{grid_g} does not match V-JEPA grid {grid_h}x{grid_w}. "
                f"set GazelleConfig.input_size to 224 (yielding a 16×16 spatial grid) to align with the 256/16 patch configuration."
            )
        scene_aligned = scene.view(B, T_g, grid_h * grid_w, D_scene)

        # Temporal align to encoder/predictor token time steps: T_enc = T_frames / tubelet_size
        T_frames = int(frames.size(2))
        T_enc = max(1, T_frames // self.tubelet_size)
        if T_g != T_enc:
            # 这里先不做时间插值；只支持标准的 tubelet 平均下采样。
            if T_g == T_frames and (T_frames % self.tubelet_size == 0):
                scene_aligned = scene_aligned.view(
                    B, T_enc, self.tubelet_size, grid_h * grid_w, D_scene
                ).mean(dim=2)
            else:
                raise ValueError(
                    f"Time mismatch: Gazelle T={T_g}, expected encoder T_enc={T_enc} "
                    f"(frames T={T_frames}, tubelet={self.tubelet_size})."
                )

        # Project to predictor embed dim, then flatten to [B, N_patches, D_pred]
        cond = self._scene_proj(scene_aligned)  # [B, T_enc, HW, D_pred]
        cond_full = cond.view(B, T_enc * grid_h * grid_w, -1)

        # Predictor expects exactly predictor_backbone.num_patches tokens.
        # Its grid_depth is configured as num_frames//tubelet_size.
        pred_grid_depth = self.num_frames // self.tubelet_size
        expected = pred_grid_depth * grid_h * grid_w
        if cond_full.size(1) != expected:
            raise ValueError(
                f"Predictor expects num_frames={self.num_frames} -> grid_depth={pred_grid_depth} "
                f"(expected {expected} tokens), but got condition tokens {cond_full.size(1)}. "
                f"Please ensure that model.num_frames matches the temporal length T used in the dataset."
            )

        return cond_full

    @torch.no_grad()
    def _gazelle_forward_scene_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Gazelle forward pass up to transformer output (scene tokens), before heatmap head.
        Mirrors `gazelle/model.py:GazeLLE.forward()` but returns tokens instead of heatmaps.
        """
        assert self._gazelle_model is not None
        model = self._gazelle_model

        # input expects a dict with images + bboxes; we want "no bbox" => zero headmap.
        # We'll replicate the internal steps to access tokens directly.
        # Backbone + linear projection: [B, C', H', W']
        x = model.backbone.forward(images)
        x = model.linear(x)
        # 与 gazelle/model.py: GazeLLE.forward 完全一致（只取 transformer 输出 tokens）
        x = x + model.pos_embed.to(x.device, dtype=x.dtype)

        head_maps = torch.cat(model.get_input_head_maps([[None]] * images.size(0)), dim=0).to(x.device)
        # gazelle/model.py 里 head_maps shape: [B, H, W]
        head_map_embeddings = (
            head_maps.unsqueeze(dim=1)
            * model.head_token.weight.to(x.device, dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        )
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # [B, HW, C]

        if getattr(model, "inout", False):
            inout_tokens = model.inout_token.weight.unsqueeze(0).repeat(x.size(0), 1, 1)
            x = torch.cat([inout_tokens, x], dim=1)

        x = model.transformer(x)
        if getattr(model, "inout", False):
            x = x[:, 1:, :]
        return x  # [B, HW, dim]

    # ---------------------------------------------------------------------
    # Unified forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        frames: torch.Tensor,
        masks_enc: Optional[List[torch.Tensor]] = None,
        masks_pred: Optional[List[torch.Tensor]] = None,
        mask_index: int = 1,
        use_gazelle_condition: bool = True,
    ) -> Dict[str, Any]:
        """
        Unified forward call.

        Video-only API.

        - `frames` must be a 5D tensor [B, C, T, H, W]. In your setup H=W=256.
        - If masks are provided, predictor will be run. If not, only encoder + gazelle are run.

        Returns :
            dict with keys:
              - "latent": encoder tokens (MultiSeq format)
              - "pred": predictor output tokens (MultiSeq format) or None
        """
        if not torch.is_tensor(frames) or frames.dim() != 5:
            raise ValueError(f"`frames` must be a 5D torch.Tensor [B,C,T,H,W], got {type(frames)} {getattr(frames,'shape',None)}")

        # Wrap into a 1-element MultiSeq list (what wrappers expect internally).
        clips: List[torch.Tensor] = [frames]

        out: Dict[str, Any] = {}

        # 1) encoder: frames -> latent tokens (MultiSeq)
        # MultiSeqWrapper expects `encoder(clips, masks_enc)` where masks_enc is list-of-list.
        if masks_enc is None:
            # If masks are not provided, default to "no masking": encode full tokens.
            # MultiSeqWrapper accepts `masks=None` and returns a list of tokens.
            latent = self.encoder(clips, None)
        else:
            latent = self.encoder(clips, masks_enc)
        out["latent"] = latent

        # 2) predictor: latent tokens (+gazelle condition) -> predicted tokens
        if masks_enc is not None and masks_pred is not None:
            # NOTE: VisionTransformerPredictor requires mask tokens when running masked prediction.
            # If constructed with use_mask_tokens=False, it will have num_mask_tokens=0 and crash.
            try:
                num_mask_tokens = int(self.predictor.backbone.backbone.num_mask_tokens)  # type: ignore[attr-defined]
            except Exception:
                num_mask_tokens = 0
            if num_mask_tokens <= 0:
                raise ValueError(
                    "Predictor path requires mask tokens, but predictor.num_mask_tokens==0. "
                    "Please construct VJEPAGazelleUnifiedModel with use_mask_tokens=True (and num_mask_tokens>=1), "
                    "or call forward() without masks_enc/masks_pred."
                )
            cond_full = None
            if use_gazelle_condition:
                # 只 forward 一次 Gazelle：这里算好 scene tokens，后续仅做投影/reshape。
                scene_tokens = self.gazelle_scene_tokens(frames)
                cond_full = self._prepare_gazelle_condition_for_predictor(frames, scene_tokens=scene_tokens)
            pred = self.predictor(latent, masks_enc, masks_pred, cond_full=cond_full)
        else:
            pred = None
        out["pred"] = pred

        return out


class _VisionTransformerPredictorWithCondition(nn.Module):
    """
    Thin wrapper that adds an optional condition token stream to the *context* tokens
    before running the standard V-JEPA predictor.
    """

    def __init__(self, backbone: VisionTransformerPredictor):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
        masks_x: torch.Tensor,
        masks_y: torch.Tensor,
        mask_index: int = 1,
        has_cls: bool = False,
        cond_full: Optional[torch.Tensor] = None,  # [B, num_patches, D_pred]
    ) -> torch.Tensor:
        """
        Same as VisionTransformerPredictor.forward, but if cond_full is provided we add:
            x_context += apply_masks(cond_full, masks_x)
        in predictor-embed space.
        """
        # Mirror the original forward, adding a single line for conditioning.
        b = self.backbone

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        B = len(x) // len(masks_x)

        # Map context tokens to predictor dimensions
        x = b.predictor_embed(x)
        if has_cls:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        _, N_ctxt, D = x.shape

        # Add positional embedding to ctxt tokens
        if not b.use_rope:
            x_pos_embed = b.predictor_pos_embed.repeat(B, 1, 1)
            x = x + apply_masks(x_pos_embed, masks_x)

        # Add Gazelle condition to ctxt tokens (aligned to full patch grid, then masked to ctxt indices)
        if cond_full is not None:
            if cond_full.dim() != 3:
                raise ValueError(f"cond_full must be [B, N_patches, D], got {tuple(cond_full.shape)}")
            if cond_full.size(0) != B:
                raise ValueError(f"cond_full batch={cond_full.size(0)} != B={B}")
            # Match predictor embed dim
            if cond_full.size(-1) != D:
                raise ValueError(f"cond_full dim={cond_full.size(-1)} != predictor_embed_dim={D}")
            x = x + apply_masks(cond_full, masks_x)

        # Make target tokens
        mask_index = mask_index % b.num_mask_tokens
        pred_tokens = b.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, b.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        # -- add pos embed
        if not b.use_rope:
            pos_embs = b.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
            from src.utils.tensors import repeat_interleave_batch

            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens = pred_tokens + pos_embs

        # Concatenate context & target tokens
        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Positions of context & target tokens
        masks_x_cat = torch.cat(masks_x, dim=0)
        masks_y_cat = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x_cat, masks_y_cat], dim=1)

        # Put tokens in sorted order
        argsort = torch.argsort(masks, dim=1)
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        # Remove the last n tokens of sorted sequence before processing
        if getattr(b, "chop_last_n_tokens", 0) > 0:
            x = x[:, : -b.chop_last_n_tokens]
            masks = masks[:, : -b.chop_last_n_tokens]

        if has_cls:
            x = torch.cat([x_cls, x], dim=1)

        # Fwd prop
        for blk in b.predictor_blocks:
            if b.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, masks, None, use_reentrant=False)
            else:
                x = blk(x, mask=masks, attn_mask=None)
        x = b.predictor_norm(x)

        if has_cls:
            x = x[:, 1:, :]

        # Return output corresponding to target tokens
        if not b.return_all_tokens:
            reverse_argsort = torch.argsort(argsort, dim=1)
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        x = b.predictor_proj(x)
        return x


class _PredictorMultiSeqWrapperWithCondition(nn.Module):
    """
    MultiSeq predictor wrapper that threads through an optional condition.
    """

    def __init__(self, backbone: _VisionTransformerPredictorWithCondition):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, has_cls: bool = False, cond_full: Optional[torch.Tensor] = None):
        outs = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mxij, myij in zip(xi, mxi, myi):
                outs[i] += [
                    self.backbone(
                        xij,
                        mxij,
                        myij,
                        mask_index=i,
                        has_cls=has_cls,
                        cond_full=cond_full,
                    )
                ]
        return outs


