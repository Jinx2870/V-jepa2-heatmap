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
      predictor: latent tokens -> future tokens (TF/AR causal prediction)
      gazelle: frames -> gaze/scene tokens (used only as conditioning signal)

Current repo state:
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
from src.models.utils.modules import ACBlock, build_action_block_causal_attention_mask
from src.utils.wrappers import MultiSeqWrapper


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
      - V-JEPA2 causal predictor (TF/AR over full token sequence)
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

        # Causal (TF/AR) predictor over the full token sequence (no actions/states),
        # conditioned by Gazelle scene tokens (added in predictor-embed space).
        self.predictor_causal = _VisionTransformerPredictorCausalWithCondition(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=self.encoder.backbone.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=self.encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
            uniform_power=uniform_power,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
            use_silu=use_pred_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=use_activation_checkpointing,
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
        pred_dev = next(self.predictor_causal.parameters()).device
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
    def gazelle_condition_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Public helper: compute predictor-aligned Gazelle condition tokens.

        Returns:
          cond_full: [B, T_enc*HW, D_pred] where T_enc = T_frames//tubelet_size
        """
        scene_tokens = self.gazelle_scene_tokens(frames)
        return self._prepare_gazelle_condition_for_predictor(frames, scene_tokens=scene_tokens)

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
        use_gazelle_condition: bool = True,
    ) -> Dict[str, Any]:
        """
        Unified forward call.

        Video-only API.

        - `frames` must be a 5D tensor [B, C, T, H, W]. In your setup H=W=256.
        Returns:
            dict with keys:
              - "latent": encoder tokens [B, N, D]
              - "cond_full": Gazelle condition tokens aligned to patch grid [B, N, D_pred] (or None)
              - "pred": predictor output tokens [B, N, D]
        """
        if not torch.is_tensor(frames) or frames.dim() != 5:
            raise ValueError(f"`frames` must be a 5D torch.Tensor [B,C,T,H,W], got {type(frames)} {getattr(frames,'shape',None)}")

        out: Dict[str, Any] = {}

        # 1) encoder: frames -> latent tokens
        latent = self.encoder.backbone(frames)  # [B, N, D_enc]
        out["latent"] = latent

        # 2) optional Gazelle condition (aligned to predictor patch grid)
        cond_full = None
        if use_gazelle_condition:
            # 只 forward 一次 Gazelle：这里算好 scene tokens，后续仅做投影/reshape。
            scene_tokens = self.gazelle_scene_tokens(frames)
            cond_full = self._prepare_gazelle_condition_for_predictor(frames, scene_tokens=scene_tokens)
        out["cond_full"] = cond_full

        # 3) predictor: TF/AR causal over full token sequence
        pred = self.predictor_causal(latent, cond_full=cond_full)
        out["pred"] = pred

        return out


class _VisionTransformerPredictorCausalWithCondition(nn.Module):
    """
    Causal predictor over full patch-token sequences (no action/state tokens).

    This is conceptually the TF/AR predictor used in original V-JEPA AC training,
    but without action/state/extrinsics. We keep the same *causal attention mask*
    idea (like `ac_predictor.py`), and allow adding Gazelle condition tokens
    (aligned to the same patch grid) in predictor-embed space.

    Input/Output:
      - input x: [B, N, D_enc], where N = T * (H_p*W_p)
      - optional cond_full: [B, N, D_pred] (already projected to predictor dim)
      - output: [B, N, D_enc]
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_frames: int,
        tubelet_size: int,
        embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        uniform_power: bool = False,
        use_rope: bool = True,
        use_sdpa: bool = True,
        use_silu: bool = False,
        wide_silu: bool = True,
        use_activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.img_height = int(img_size)
        self.img_width = int(img_size)
        self.patch_size = int(patch_size)
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size
        self.num_patches_per_t = int(self.grid_height * self.grid_width)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.uniform_power = bool(uniform_power)

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_blocks = nn.ModuleList(
            [
                ACBlock(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    use_sdpa=use_sdpa,
                    is_causal=False,  # we provide an explicit attn_mask
                    wide_silu=wide_silu,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                )
                for _ in range(int(depth))
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Precompute causal attention mask over token-time steps (tubelets)
        grid_depth = self.num_frames // self.tubelet_size
        self.attn_mask = build_action_block_causal_attention_mask(
            grid_depth, self.grid_height, self.grid_width, add_tokens=0
        )

    def forward(self, x: torch.Tensor, cond_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,N,D], got {tuple(x.shape)}")
        x = self.predictor_embed(x)
        B, N, D = x.shape
        if N % self.num_patches_per_t != 0:
            raise ValueError(f"N={N} not divisible by num_patches_per_t={self.num_patches_per_t}")
        T = N // self.num_patches_per_t

        if cond_full is not None:
            if cond_full.shape != (B, N, D):
                raise ValueError(f"cond_full must be [B,N,D_pred] == {(B,N,D)}, got {tuple(cond_full.shape)}")
            x = x + cond_full

        # Causal attention mask sized to current sequence length
        attn_mask = self.attn_mask[:N, :N].to(x.device, non_blocking=True)

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, None, attn_mask, T, self.grid_height, self.grid_width, 0, use_reentrant=False
                )
            else:
                x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=self.grid_height, W=self.grid_width, action_tokens=0)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x


