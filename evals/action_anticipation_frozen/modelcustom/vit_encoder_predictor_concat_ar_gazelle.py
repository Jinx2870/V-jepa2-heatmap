"""
Gaze-conditioned action anticipation wrapper (frozen encoder + frozen predictor).

This follows the existing `vit_encoder_predictor_concat_ar.py` contract used by
`evals.action_anticipation_frozen`:
  - init_module(...) builds an nn.Module with forward(x, anticipation_times) -> tokens
  - encoder and predictor are loaded from a pretraining checkpoint and frozen

Difference:
  - We load Gazelle (frozen) to extract *scene tokens* (transformer outputs; typically 3 layers in Gazelle)
  - We load a learned `scene_proj` from the same pretraining checkpoint
  - We use `scene_proj(scene_tokens)` as a condition added in predictor-embed space
    (masked by `masks_x`, same design as the gaze-conditioned pretraining code)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.predictor as vit_pred
import src.models.vision_transformer as vit
from src.masks.utils import apply_masks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GazelleCfg:
    checkpoint: str
    model_name: str = "gazelle_dinov2_vitb14_inout"
    python_path: Optional[str] = None
    device: str = "cuda"
    input_size: int = 224
    max_batch_size: int = 64  # frames per forward batch (B*T)


class _VisionTransformerPredictorWithCondition(nn.Module):
    """
    Wrapper around `src.models.predictor.VisionTransformerPredictor` that adds an
    optional dense condition token stream `cond_full` in predictor-embed space.

    `cond_full` is aligned to the *full* patch grid (same length as pos_embed),
    and is masked down to context indices via `apply_masks(cond_full, masks_x)`.
    """

    def __init__(self, backbone: vit_pred.VisionTransformerPredictor):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False, cond_full: Optional[torch.Tensor] = None):
        b = self.backbone
        assert (masks_x is not None) and (masks_y is not None), "Cannot run predictor without mask indices"
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        # Batch size
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
                raise ValueError(f"cond_full must be [B, N_patches, D_pred], got {tuple(cond_full.shape)}")
            if cond_full.size(0) != B:
                raise ValueError(f"cond_full batch={cond_full.size(0)} != B={B}")
            if cond_full.size(1) != b.num_patches:
                raise ValueError(f"cond_full N={cond_full.size(1)} != predictor.num_patches={b.num_patches}")
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
            from src.utils.tensors import repeat_interleave_batch

            pos_embs = b.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
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
        argsort = torch.argsort(masks, dim=1)  # [B, N]
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
            reverse_argsort = torch.argsort(argsort, dim=1)  # [B, N]
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        x = b.predictor_proj(x)
        return x


class AnticipativeWrapperGazelle(nn.Module):
    """Action anticipation wrapper with Gazelle scene-token conditioning."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: _VisionTransformerPredictorWithCondition,
        gazelle_cfg: GazelleCfg,
        scene_proj: nn.Module,
        frames_per_second=4,
        crop_size=224,
        patch_size=16,
        tubelet_size=2,
        # -- wrapper kwargs
        no_predictor=False,
        num_output_frames=2,
        num_steps=1,
        no_encoder=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.grid_size = crop_size // patch_size
        self.tubelet_size = tubelet_size
        self.no_predictor = no_predictor
        self.num_output_frames = max(num_output_frames, tubelet_size)
        self.frames_per_second = frames_per_second
        self.num_steps = num_steps
        self.no_encoder = no_encoder
        self._gazelle_cfg = gazelle_cfg
        self._scene_proj = scene_proj

        assert not (self.no_predictor and self.no_encoder), "Anticipative wrapper must use predictor or encoder"

        # -----------------------
        # Gazelle init (frozen)
        # -----------------------
        if self._gazelle_cfg.python_path:
            path = os.path.abspath(os.path.expanduser(self._gazelle_cfg.python_path))
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        env_path = os.environ.get("GAZELLE_REPO") or os.environ.get("GAZELLE_PYTHONPATH")
        if env_path:
            path = os.path.abspath(os.path.expanduser(env_path))
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        from gazelle.backbone import DinoV2Backbone  # type: ignore
        from gazelle.model import GazeLLE  # type: ignore

        name = (self._gazelle_cfg.model_name or "").lower()
        if "vitb14" in name:
            backbone_name = "dinov2_vitb14"
        elif "vitl14" in name:
            backbone_name = "dinov2_vitl14"
        else:
            raise ValueError(f"Unsupported Gazelle model_name={self._gazelle_cfg.model_name} (expect vitb14/vitl14)")

        inout = "inout" in name
        in_size = (int(self._gazelle_cfg.input_size), int(self._gazelle_cfg.input_size))
        backbone = DinoV2Backbone(backbone_name)
        gmodel = GazeLLE(backbone, inout=inout, in_size=in_size)

        state = torch.load(self._gazelle_cfg.checkpoint, map_location="cpu")
        # align with `src/models/vjepa_gazelle_unified.py`: do not load checkpoint pos_embed
        state = dict(state)
        state.pop("pos_embed", None)
        gmodel.load_gazelle_state_dict(state)
        gmodel.eval()
        for p in gmodel.parameters():
            p.requires_grad = False

        gdev = torch.device(self._gazelle_cfg.device)
        gmodel.to(gdev)
        self._gazelle_model = gmodel
        self._gazelle_device = gdev

        # cache for alignment
        self.img_size = int(crop_size)
        self.patch_size = int(patch_size)

    @torch.no_grad()
    def _gazelle_forward_scene_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward Gazelle up to transformer outputs (scene tokens), before heatmap head.
        Returns: [B, HW, D_scene]
        """
        model = self._gazelle_model
        x = model.backbone.forward(images)
        x = model.linear(x)
        x = x + model.pos_embed.to(x.device, dtype=x.dtype)
        head_maps = torch.cat(model.get_input_head_maps([[None]] * images.size(0)), dim=0).to(x.device)
        head_map_embeddings = (
            head_maps.unsqueeze(dim=1) * model.head_token.weight.to(x.device, dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)
        )
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # [B, HW, C]
        if getattr(model, "inout", False):
            inout_tokens = model.inout_token.weight.unsqueeze(0).repeat(x.size(0), 1, 1)
            x = torch.cat([inout_tokens, x], dim=1)
        x = model.transformer(x)
        if getattr(model, "inout", False):
            x = x[:, 1:, :]
        return x

    @torch.no_grad()
    def gazelle_scene_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B,C,T,H,W]
        returns: [B,T,HW,D_scene]
        """
        if frames.dim() != 5 or frames.size(1) != 3:
            raise ValueError(f"Expected frames [B,3,T,H,W], got {tuple(frames.shape)}")
        B, C, T, H, W = frames.shape
        dev = self._gazelle_device

        x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(dev, non_blocking=(dev.type == "cuda"))

        # Preprocess matches `src/models/vjepa_gazelle_unified.py` (keep identical for consistency)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if x.max() > 1.5:
                x = (x / 255.0).clamp(0.0, 1.0)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        target_size = int(self._gazelle_cfg.input_size)
        if x.shape[-2:] != (target_size, target_size):
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)

        max_bs = int(self._gazelle_cfg.max_batch_size)
        outs = []
        for start in range(0, x.size(0), max_bs):
            outs.append(self._gazelle_forward_scene_tokens(x[start : start + max_bs]))
        tokens = torch.cat(outs, dim=0)  # [B*T, HW, D]
        return tokens.view(B, T, tokens.size(1), tokens.size(2))

    @torch.no_grad()
    def gazelle_condition_tokens(self, frames: torch.Tensor, predictor_embed_dim: int, num_frames: int) -> torch.Tensor:
        """
        Convert Gazelle scene tokens -> predictor-aligned condition tokens.
        Returns: cond_full [B, N_patches, D_pred] where N_patches == predictor.num_patches
        """
        scene = self.gazelle_scene_tokens(frames)  # [B, T_frames, HW, D_scene]

        # move to predictor device
        pdev = next(self.predictor.parameters()).device
        scene = scene.to(pdev, non_blocking=True)

        B, T_g, HW_g, _ = scene.shape
        grid_g = int(HW_g**0.5)
        if grid_g * grid_g != HW_g:
            raise ValueError(f"Gazelle HW={HW_g} not square.")
        grid_h = self.img_size // self.patch_size
        grid_w = self.img_size // self.patch_size
        if (grid_g, grid_g) != (grid_h, grid_w):
            raise ValueError(f"Gazelle grid {grid_g}x{grid_g} != V-JEPA grid {grid_h}x{grid_w}")

        # temporal align to encoder token time steps: T_enc = T_frames / tubelet_size
        T_frames = int(frames.size(2))
        T_enc = max(1, T_frames // int(self.tubelet_size))
        scene_aligned = scene.view(B, T_g, grid_h * grid_w, -1)
        if T_g != T_enc:
            if T_g == T_frames and (T_frames % int(self.tubelet_size) == 0):
                scene_aligned = scene_aligned.view(B, T_enc, int(self.tubelet_size), grid_h * grid_w, -1).mean(dim=2)
            else:
                raise ValueError(f"Time mismatch: Gazelle T={T_g}, expected T_enc={T_enc} (frames T={T_frames})")

        # project to predictor embed dim and flatten
        cond = self._scene_proj(scene_aligned)  # [B, T_enc, HW, D_pred]
        cond_full = cond.view(B, T_enc * grid_h * grid_w, predictor_embed_dim)

        expected = (int(num_frames) // int(self.tubelet_size)) * grid_h * grid_w
        if cond_full.size(1) != expected:
            raise ValueError(f"cond_full tokens={cond_full.size(1)} != expected={expected} (check num_frames/tubelet_size)")
        return cond_full

    def forward(self, x: torch.Tensor, anticipation_times: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,T,H,W]
        anticipation_times: [B]
        returns: [B, N_out, D]
        """
        tokens = self.encoder(x)
        if self.no_predictor:
            return tokens

        B, N, D = tokens.size()

        if self.no_encoder:
            x_accumulate = torch.rand(tokens.size(0), 0, tokens.size(2), device=tokens.device, dtype=tokens.dtype)
        else:
            x_accumulate = tokens.clone()

        ctxt_positions = torch.arange(N, device=tokens.device).unsqueeze(0).repeat(B, 1)
        anticipation_steps = (anticipation_times * self.frames_per_second / self.tubelet_size).to(torch.int64)
        skip_positions = N + int(self.grid_size**2) * anticipation_steps

        N_pred = int(self.grid_size**2 * (self.num_output_frames // self.tubelet_size))
        tgt_positions = torch.arange(N_pred, device=tokens.device).unsqueeze(0).repeat(B, 1)
        tgt_positions = tgt_positions + skip_positions.unsqueeze(1).repeat(1, N_pred)

        ctxt_positions = ctxt_positions % N
        tgt_positions = tgt_positions % N

        pred_embed_dim = int(self.predictor.backbone.predictor_embed.out_features)
        num_frames = int(getattr(self.encoder, "num_frames", x.size(2)))
        cond_full = self.gazelle_condition_tokens(x, predictor_embed_dim=pred_embed_dim, num_frames=num_frames)

        for _ in range(self.num_steps):
            x_pred = self.predictor(tokens, masks_x=ctxt_positions, masks_y=tgt_positions, cond_full=cond_full)
            x_accumulate = torch.cat([x_accumulate, x_pred], dim=1)
            tokens = torch.cat([tokens[:, N_pred:, :], x_pred], dim=1)

        return x_accumulate


def init_module(
    frames_per_clip: int,
    frames_per_second: int,
    resolution: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"Loading pretrained checkpoint from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")

    # -----------------------
    # Encoder
    # -----------------------
    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key", "encoder")
    enc_model_name = enc_kwargs.get("model_name")

    encoder = vit.__dict__[enc_model_name](img_size=resolution, num_frames=frames_per_clip, **enc_kwargs)
    if enc_ckp_key not in ckpt:
        raise KeyError(f"Checkpoint missing encoder key={enc_ckp_key}. Available keys={list(ckpt.keys())}")
    pretrained_dict = ckpt[enc_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'encoder key "{k}" missing in checkpoint; using init weights')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'encoder key "{k}" shape mismatch {pretrained_dict[k].shape} vs {v.shape}; using init weights')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded encoder with msg: {msg}")

    # -----------------------
    # Predictor (masked)
    # -----------------------
    prd_kwargs = model_kwargs["predictor"]
    prd_ckp_key = prd_kwargs.get("checkpoint_key", "predictor")
    prd_model_name = prd_kwargs.get("model_name")

    predictor_backbone = vit_pred.__dict__[prd_model_name](
        img_size=resolution,
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        num_frames=frames_per_clip,
        **prd_kwargs,
    )
    if prd_ckp_key not in ckpt:
        raise KeyError(
            f"Checkpoint missing predictor key={prd_ckp_key}. "
            f"If your checkpoint uses a different key, update config model_kwargs.predictor.checkpoint_key."
        )
    pretrained_dict = ckpt[prd_ckp_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in predictor_backbone.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'predictor key "{k}" missing in checkpoint; using init weights')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'predictor key "{k}" shape mismatch {pretrained_dict[k].shape} vs {v.shape}; using init weights')
            pretrained_dict[k] = v
    msg = predictor_backbone.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded predictor with msg: {msg}")

    predictor = _VisionTransformerPredictorWithCondition(predictor_backbone)

    # -----------------------
    # Gazelle + scene_proj (load from checkpoint)
    # -----------------------
    gaz = model_kwargs.get("gazelle", {}) or {}
    if not gaz:
        raise ValueError("model_kwargs.gazelle must be provided for gaze-conditioned eval")
    gazelle_cfg = GazelleCfg(
        checkpoint=str(gaz["checkpoint"]),
        model_name=str(gaz.get("model_name", "gazelle_dinov2_vitb14_inout")),
        python_path=gaz.get("python_path", None),
        device=str(gaz.get("device", "cuda")),
        input_size=int(gaz.get("input_size", 224)),
        max_batch_size=int(gaz.get("max_batch_size", gaz.get("batch_size", 64))),
    )

    scene_proj = nn.LazyLinear(predictor_backbone.predictor_embed.out_features, bias=True)
    scene_ckpt_key = str(model_kwargs.get("scene_proj_checkpoint_key", "scene_proj"))
    if scene_ckpt_key not in ckpt or ckpt.get(scene_ckpt_key, None) is None:
        raise KeyError(
            f"Checkpoint missing scene projection key={scene_ckpt_key}. "
            "Expected gaze pretraining checkpoint to contain a trained projection (e.g. scene_proj)."
        )
    msg = scene_proj.load_state_dict(ckpt[scene_ckpt_key], strict=False)
    logger.info(f"Loaded scene_proj with msg: {msg}")

    # -----------------------
    # Build wrapper + freeze encoder/predictor (and projector)
    # -----------------------
    model = AnticipativeWrapperGazelle(
        encoder=encoder,
        predictor=predictor,
        gazelle_cfg=gazelle_cfg,
        scene_proj=scene_proj,
        frames_per_second=frames_per_second,
        crop_size=resolution,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        **wrapper_kwargs,
    )
    model.embed_dim = encoder.embed_dim

    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = False
    for p in model._scene_proj.parameters():
        p.requires_grad = False

    return model


