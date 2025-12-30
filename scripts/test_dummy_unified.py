#!/usr/bin/env python3
"""
Smoke-test for VJEPAGazelleUnifiedModel with dummy video input.

Examples:
  # Encoder only (no masks -> no predictor), skip Gazelle init:
  python scripts/test_dummy_unified.py --device cpu

  # Encoder + predictor (with dummy masks), skip Gazelle init:
  python scripts/test_dummy_unified.py --device cuda --with-predictor

  # If you want to actually init Gazelle (requires dinov2 download + checkpoint):
  python scripts/test_dummy_unified.py --init-gazelle \
    --gazelle-python-path /data3/lg2/human_wm/gazelle \
    --gazelle-checkpoint /path/to/gazelle_ckpt.pt \
    --gazelle-device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch


def _repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def _ensure_importable() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def _make_disjoint_masks(B: int, N: int, k_ctxt: int, k_pred: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    k_ctxt = min(k_ctxt, N)
    k_pred = min(k_pred, max(0, N - k_ctxt))
    enc = []
    pred = []
    for _ in range(B):
        perm = torch.randperm(N, device=device)
        enc.append(perm[:k_ctxt])
        pred.append(perm[k_ctxt : k_ctxt + k_pred])
    return torch.stack(enc, dim=0).long(), torch.stack(pred, dim=0).long()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--W", type=int, default=256)
    p.add_argument("--with-predictor", action="store_true")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--dtype", default="float32", choices=["float32", "uint8"])

    # Gazelle (optional)
    p.add_argument("--init-gazelle", action="store_true")
    p.add_argument("--gazelle-python-path", default=None)
    p.add_argument("--gazelle-checkpoint", default=None)
    p.add_argument("--gazelle-device", default="cuda")
    p.add_argument("--gazelle-model-name", default="gazelle_dinov2_vitb14_inout")
    p.add_argument("--gazelle-input-size", type=int, default=224)

    args = p.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你传了 --device cuda，但当前 torch.cuda.is_available()=False")

    _ensure_importable()
    from src.models.vjepa_gazelle_unified import GazelleConfig, VJEPAGazelleUnifiedModel

    gazelle_cfg = None
    if args.init_gazelle:
        if not args.gazelle_checkpoint:
            raise ValueError("--init-gazelle 需要同时提供 --gazelle-checkpoint")
        gazelle_cfg = GazelleConfig(
            checkpoint=args.gazelle_checkpoint,
            model_name=args.gazelle_model_name,
            python_path=args.gazelle_python_path,
            device=args.gazelle_device,
            input_size=args.gazelle_input_size,
        )

    model = VJEPAGazelleUnifiedModel(
        img_size=args.H,
        patch_size=args.patch_size,
        num_frames=args.T,
        tubelet_size=args.tubelet_size,
        use_mask_tokens=bool(args.with_predictor),
        num_mask_tokens=2,
        gazelle_cfg=gazelle_cfg,
        init_gazelle=bool(args.init_gazelle),
    ).to(dev)

    # IMPORTANT: nn.Module.to(dev) moves *all* submodules, including Gazelle.
    # If user wants Gazelle on a different device than the main model, move it back.
    if args.init_gazelle:
        gaz_dev = torch.device(args.gazelle_device)
        if getattr(model, "_gazelle_model", None) is not None:
            model._gazelle_model.to(gaz_dev)  # type: ignore[attr-defined]
            model._gazelle_device = gaz_dev  # type: ignore[attr-defined]
    model.eval()

    # Dummy video: [B, 3, T, H, W]
    if args.dtype == "uint8":
        frames = torch.randint(0, 256, (args.B, 3, args.T, args.H, args.W), dtype=torch.uint8, device=dev)
        # Encoder expects float; keep dummy simple: convert to [0,1] float
        frames = frames.float() / 255.0
    else:
        frames = torch.rand((args.B, 3, args.T, args.H, args.W), dtype=torch.float32, device=dev)

    masks_enc = masks_pred = None
    if args.with_predictor:
        T_enc = max(1, args.T // args.tubelet_size)
        H_p = args.H // args.patch_size
        W_p = args.W // args.patch_size
        N = T_enc * H_p * W_p
        # keep ~50% as context, predict ~25%
        k_ctxt = max(1, N // 2)
        k_pred = max(1, N // 4)
        mask_enc, mask_pred = _make_disjoint_masks(args.B, N, k_ctxt, k_pred, device=dev)
        # MultiSeqWrapper expects: list (per-clip) of list (per-multimask) of [B,K] index tensors
        masks_enc = [[mask_enc]]
        masks_pred = [[mask_pred]]

    with torch.no_grad():
        out = model(
            frames=frames,
            masks_enc=masks_enc,
            masks_pred=masks_pred,
            use_gazelle_condition=bool(args.init_gazelle),  # only condition if gazelle is initialized
        )

    latent = out["latent"]
    pred = out["pred"]

    def _shape(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, list):
            # list-of-(list-of-tensors) in multiseq mode, or list-of-tensors in no-mask mode
            return [ _shape(xx) for xx in x ]
        return type(x)

    print("frames:", tuple(frames.shape), frames.dtype, frames.device)
    print("latent:", _shape(latent))
    print("pred:", _shape(pred))


if __name__ == "__main__":
    main()


