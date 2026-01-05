#!/usr/bin/env python3
"""
Gaze-conditioned training entrypoint for Ego4D video clips with TF + AR loss.

This is intentionally isolated under `app/vjepa_ego4d/`.

Data contract:
  - Uses `Ego4DGazeVideoDataset` (via `init_gaze_data`) which returns ONLY frames.
  - Dataloader yields:
      frames: [B, C, T, H, W]

Config:
  - You can run this script with a YAML config file, similar to other apps:
      python app/vjepa_ego4d/train_gaze_unified.py --fname /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import pprint
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

def _ensure_repo_root_on_path() -> None:
    """
    Allow running this file directly:
      python app/vjepa_ego4d/train_gaze_unified.py --fname ...
    Without requiring `PYTHONPATH=.` or `python -m app....`.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()

from app.vjepa.transforms import make_transforms
from app.vjepa_ego4d.ego4d import init_gaze_data
from src.utils.logging import AverageMeter, WandbLogger
from src.utils.distributed import init_distributed
from src.utils.checkpoint_loader import robust_checkpoint_loader


_GLOBAL_SEED = 0


def main(args: Dict[str, Any]) -> None:
    # -------------------------
    # Meta / reproducibility
    # -------------------------
    cfgs_meta = args.get("meta", {})
    folder = args.get("folder", None)
    seed = int(cfgs_meta.get("seed", _GLOBAL_SEED))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    world_size, rank = init_distributed()
    if rank == 0:
        pprint.PrettyPrinter(indent=2).pprint(args)
        if folder:
            os.makedirs(folder, exist_ok=True)

    # -------------------------
    # Wandb (rank=0 only; safe if wandb not installed)
    # -------------------------
    wandb_cfg = (cfgs_meta.get("wandb", {}) or {}) if isinstance(cfgs_meta, dict) else {}
    use_wandb = bool(wandb_cfg.get("enable", False)) and (rank == 0)
    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", None),
        name=wandb_cfg.get("name", None),
        config=args,
        enabled=use_wandb,
        entity=wandb_cfg.get("entity", None),
        group=wandb_cfg.get("group", None),
        tags=wandb_cfg.get("tags", None),
        notes=wandb_cfg.get("notes", None),
    )

    # -------------------------
    # Device
    # -------------------------
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -------------------------
    # Mixed precision (match original `app/vjepa_ego4d/train.py`)
    # -------------------------
    which_dtype = str(cfgs_meta.get("dtype", "float32"))
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    # -------------------------
    # Data
    # -------------------------
    cfgs_data = args.get("data", {})
    dataset_paths = cfgs_data.get("datasets", [])
    if not dataset_paths:
        raise ValueError("Config must provide data.datasets = [<csv_with_abs_mp4_paths>]")
    dataset_path = dataset_paths[0]

    batch_size = int(cfgs_data.get("batch_size", 2))
    fps = cfgs_data.get("fps", 5)
    crop_size = int(cfgs_data.get("crop_size", 256))
    patch_size = int(cfgs_data.get("patch_size", 16))
    tubelet_size = int(cfgs_data.get("tubelet_size", 2))
    num_workers = int(cfgs_data.get("num_workers", 4))
    pin_mem = bool(cfgs_data.get("pin_mem", True))
    persistent_workers = bool(cfgs_data.get("persistent_workers", True))
    frames_per_clip = int(cfgs_data.get("frames_per_clip", cfgs_data.get("max_num_frames", 16)))

    cfgs_data_aug = args.get("data_aug", {})
    transform = make_transforms(
        random_horizontal_flip=bool(cfgs_data_aug.get("horizontal_flip", False)),
        random_resize_aspect_ratio=tuple(cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])),
        random_resize_scale=tuple(cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])),
        reprob=float(cfgs_data_aug.get("reprob", 0.0)),
        auto_augment=bool(cfgs_data_aug.get("auto_augment", False)),
        motion_shift=bool(cfgs_data_aug.get("motion_shift", False)),
        crop_size=crop_size,
    )

    collator = torch.utils.data.default_collate

    timeout = int(cfgs_data.get("timeout", 0))
    prefetch_factor = int(cfgs_data.get("prefetch_factor", 2))
    (loader, sampler) = init_gaze_data(
        data_path=dataset_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=fps,
        # NOTE: we keep the dataset temporal length intact for TF/AR (no downsample here)
        tubelet_size=1,
        transform=transform,
        collator=collator,
        num_workers=num_workers,
        timeout=timeout,
        prefetch_factor=prefetch_factor,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        drop_last=True,
        shuffle=True,
    )
    sampler.set_epoch(0)

    # --- Validation loader (rank0 only, full val set)
    cfgs_val = cfgs_data.get("val", None)
    val_loader = None
    if isinstance(cfgs_val, dict) and cfgs_val.get("datasets", None):
        val_paths = cfgs_val.get("datasets", [])
        val_path = val_paths[0]
        val_bs = int(cfgs_val.get("batch_size", 1))
        val_workers = int(cfgs_val.get("num_workers", 0))
        val_pin = bool(cfgs_val.get("pin_mem", True))
        val_persist = bool(cfgs_val.get("persistent_workers", False))
        val_fps = cfgs_val.get("fps", fps)
        val_tubelet = int(cfgs_val.get("tubelet_size", tubelet_size))
        val_timeout = int(cfgs_val.get("timeout", 0))
        val_prefetch = int(cfgs_val.get("prefetch_factor", 2))

        val_aug = cfgs_val.get("data_aug", {}) or {}
        val_transform = make_transforms(
            random_horizontal_flip=bool(val_aug.get("horizontal_flip", False)),
            random_resize_aspect_ratio=tuple(val_aug.get("random_resize_aspect_ratio", [1.0, 1.0])),
            random_resize_scale=tuple(val_aug.get("random_resize_scale", [1.0, 1.0])),
            reprob=float(val_aug.get("reprob", 0.0)),
            auto_augment=bool(val_aug.get("auto_augment", False)),
            motion_shift=bool(val_aug.get("motion_shift", False)),
            crop_size=crop_size,
        )

        if rank == 0:
            (val_loader, _val_sampler) = init_gaze_data(
                data_path=val_path,
                batch_size=val_bs,
                frames_per_clip=frames_per_clip,
                fps=val_fps,
                tubelet_size=1,
                transform=val_transform,
                collator=collator,
                num_workers=val_workers,
                timeout=val_timeout,
                prefetch_factor=val_prefetch,
                world_size=1,
                rank=0,
                pin_mem=val_pin,
                persistent_workers=val_persist,
                drop_last=False,
                shuffle=False,
            )

    # -------------------------
    # Model (Unified: V-JEPA encoder + Gazelle + causal predictor) for TF + AR losses
    # -------------------------
    cfgs_model = args.get("model", {})
    pred_depth = int(cfgs_model.get("pred_depth", 6))
    pred_embed_dim = int(cfgs_model.get("pred_embed_dim", 384))
    model_name = str(cfgs_model.get("model_name", "vit_base"))
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_is_frame_causal = bool(cfgs_model.get("pred_is_frame_causal", True))
    uniform_power = bool(cfgs_model.get("uniform_power", False))
    use_rope = bool(cfgs_model.get("use_rope", False))
    # allow meta.use_sdpa to be the default if model.use_sdpa is not specified
    use_sdpa = bool(cfgs_model.get("use_sdpa", cfgs_meta.get("use_sdpa", False)))
    use_silu = bool(cfgs_model.get("use_silu", False))
    use_pred_silu = bool(cfgs_model.get("use_pred_silu", False))
    wide_silu = bool(cfgs_model.get("wide_silu", True))
    use_activation_checkpointing = bool(cfgs_model.get("use_activation_checkpointing", False))
    condition_mode = str(cfgs_model.get("condition_mode", cfgs_data.get("condition_mode", "none"))).lower()
    use_gazelle = condition_mode == "gazelle"

    gaz_cfg = cfgs_data.get("gazelle", {}) or args.get("gazelle", {})
    gazelle_cfg = None
    if use_gazelle:
        ckpt = gaz_cfg.get("checkpoint", None)
        if not ckpt:
            raise ValueError("condition_mode=gazelle requires data.gazelle.checkpoint in config")
        from src.models.vjepa_gazelle_unified import GazelleConfig

        gazelle_cfg = GazelleConfig(
            checkpoint=str(ckpt),
            model_name=str(gaz_cfg.get("model_name", "gazelle_dinov2_vitb14_inout")),
            python_path=gaz_cfg.get("python_path", None),
            device=str(gaz_cfg.get("device", "cuda")),
            input_size=int(gaz_cfg.get("input_size", 224)),
            max_batch_size=int(gaz_cfg.get("batch_size", gaz_cfg.get("max_batch_size", 64))),
            min_cuda_free_mb=int(gaz_cfg.get("min_cuda_free_mb", 2048)),
        )

    from src.models.vjepa_gazelle_unified import VJEPAGazelleUnifiedModel

    model = VJEPAGazelleUnifiedModel(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        model_name=model_name,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        pred_num_heads=pred_num_heads,
        predictor_mode="causal",
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        gazelle_cfg=gazelle_cfg,
        init_gazelle=bool(use_gazelle),
    ).to(device)

    if model.predictor_causal is None:
        raise RuntimeError("Model was constructed with predictor_mode='causal' but predictor_causal is None.")

    # Optional: load pretrained encoder weights
    meta = args.get("meta", {}) or {}
    p_file = meta.get("pretrain_checkpoint", None)
    context_encoder_key = meta.get("context_encoder_key", "encoder")
    if p_file:
        ckpt = robust_checkpoint_loader(str(p_file), map_location="cpu")
        if isinstance(ckpt, dict) and context_encoder_key in ckpt:
            sd = ckpt[context_encoder_key]
            if isinstance(sd, dict):
                sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
                msg = model.encoder.backbone.load_state_dict(sd, strict=False)
                if rank == 0:
                    print(f"Loaded pretrained encoder from {p_file} with msg: {msg}")

    # For TF/AR training we only train the causal predictor (+ scene projection if gazelle).
    model.encoder.backbone.eval()
    for p in model.encoder.backbone.parameters():
        p.requires_grad = False

    model.predictor_causal.train()
    if use_gazelle:
        model._scene_proj.train()

    # -------------------------
    # Target encoder (EMA copy) + optimizer
    # -------------------------
    target_encoder = copy.deepcopy(model.encoder.backbone).to(device)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    cfgs_opt = args.get("optimization", {})
    lr = float(cfgs_opt.get("lr", 1e-4))
    wd = float(cfgs_opt.get("weight_decay", 0.05))
    num_epochs = int(cfgs_opt.get("epochs", 1))
    ipe = cfgs_opt.get("ipe", None)
    ipe = int(ipe) if ipe is not None else None

    cfgs_loss = args.get("loss", {})
    loss_exp = float(cfgs_loss.get("loss_exp", 2.0))
    normalize_reps = bool(cfgs_loss.get("normalize_reps", True))
    # TF/AR here runs on V-JEPA token-time steps: T_enc = T_frames / tubelet_size
    T_enc = max(1, frames_per_clip // max(1, tubelet_size))
    auto_steps = int(max(1, min(cfgs_loss.get("auto_steps", 1), max(T_enc - 1, 1))))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd,
    )

    # -------------------------
    # Checkpointing (match repo style: latest.pt + periodic e{epoch}.pt)
    # -------------------------
    save_every_freq = int(cfgs_meta.get("save_every_freq", -1))
    resume_ckpt = cfgs_meta.get("resume_checkpoint", None)
    latest_path = os.path.join(folder, "latest.pt") if folder else None

    def _save_checkpoint(epoch_1based: int) -> None:
        if rank != 0 or not folder:
            return
        save_dict = {
            "epoch": int(epoch_1based),
            "global_step": int(global_step),
            # only save trainable parts + (optional) frozen encoder for reproducibility
            "encoder": model.encoder.backbone.state_dict(),
            "predictor_causal": model.predictor_causal.state_dict(),
            "scene_proj": (model._scene_proj.state_dict() if use_gazelle else None),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "config": args,
        }
        torch.save(save_dict, latest_path)
        if save_every_freq > 0 and (epoch_1based % save_every_freq == 0):
            torch.save(save_dict, os.path.join(folder, f"e{epoch_1based}.pt"))

    def _try_resume() -> int:
        """
        Returns start_epoch (0-based) to continue training.
        """
        if resume_ckpt is None and latest_path and os.path.exists(latest_path):
            ckpt_path = latest_path
        elif resume_ckpt is not None:
            ckpt_path = os.path.join(folder, resume_ckpt) if folder and not os.path.isabs(str(resume_ckpt)) else str(resume_ckpt)
        else:
            return 0

        if not ckpt_path or not os.path.exists(ckpt_path):
            return 0
        ckpt = robust_checkpoint_loader(ckpt_path, map_location="cpu")
        try:
            if "encoder" in ckpt:
                model.encoder.backbone.load_state_dict(ckpt["encoder"], strict=False)
            if "predictor_causal" in ckpt:
                model.predictor_causal.load_state_dict(ckpt["predictor_causal"], strict=False)
            if use_gazelle and ckpt.get("scene_proj", None) is not None:
                model._scene_proj.load_state_dict(ckpt["scene_proj"], strict=False)
            if "opt" in ckpt:
                optimizer.load_state_dict(ckpt["opt"])
            if scaler is not None and ckpt.get("scaler", None) is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch_1based = int(ckpt.get("epoch", 0))
            if rank == 0:
                print(f"Resumed from checkpoint: {ckpt_path} (epoch={start_epoch_1based})")
            return max(0, start_epoch_1based - 1)
        except Exception as e:
            if rank == 0:
                print(f"WARNING: failed to resume from {ckpt_path}: {e}")
            return 0

    tokens_per_frame = int((crop_size // patch_size) ** 2)

    def forward_target(clips: torch.Tensor) -> torch.Tensor:
        """
        Compute target latents (tubelet-time patch tokens) using EMA target encoder.
        Returns: h [B, (T_enc*HW), D] where T_enc = T_frames//tubelet_size.
        """
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                h = target_encoder(clips)  # [B, T_enc*HW, D]
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
            return h

    def forward_predictions(z: torch.Tensor, cond_full: Optional[torch.Tensor]):
        """
        TF: predict next-step tubelet tokens from context tubelet tokens.
        AR: roll out auto-regressively using the causal predictor.
        """

        def _step_predictor(_z: torch.Tensor, _cond: Optional[torch.Tensor]) -> torch.Tensor:
            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                _z = model.predictor_causal(_z, cond_full=_cond)
            if normalize_reps:
                _z = F.layer_norm(_z, (_z.size(-1),))
            return _z

        # Teacher-forcing: input tokens for steps [0..T_enc-2], predict [1..T_enc-1]
        z_ctxt = z[:, :-tokens_per_frame]
        cond_ctxt = cond_full[:, : z_ctxt.size(1), :] if cond_full is not None else None
        z_tf = _step_predictor(z_ctxt, cond_ctxt)

        # Auto-regressive rollouts: seed with first GT step + first predicted step
        z_ar_ctx = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
        for _ in range(1, auto_steps):
            cond_seq = cond_full[:, : z_ar_ctx.size(1), :] if cond_full is not None else None
            z_nxt = _step_predictor(z_ar_ctx, cond_seq)[:, -tokens_per_frame:]
            z_ar_ctx = torch.cat([z_ar_ctx, z_nxt], dim=1)
        z_ar = z_ar_ctx[:, tokens_per_frame:]

        return z_tf, z_ar

    def loss_fn(z_pred: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Align to "next-frame" target tokens
        h_shift = h[:, tokens_per_frame : z_pred.size(1) + tokens_per_frame]
        return torch.mean(torch.abs(z_pred - h_shift) ** loss_exp) / loss_exp

    @torch.no_grad()
    def run_validation(epoch: int, global_step: int) -> None:
        if val_loader is None or rank != 0:
            return
        # Optional: limit validation length / log frequency (helps diagnose slow / stuck video decode)
        val_cfg = (cfgs_data.get("val", {}) or {}) if isinstance(cfgs_data, dict) else {}
        val_max_iters = int(val_cfg.get("max_iters", -1))
        val_log_freq = int(val_cfg.get("log_freq", cfgs_meta.get("val_log_freq", 50)))

        model.predictor_causal.eval()
        if use_gazelle:
            model._scene_proj.eval()

        loss_m = AverageMeter()
        tf_m = AverageMeter()
        ar_m = AverageMeter()

        try:
            n_batches = len(val_loader)
        except Exception:
            n_batches = None
        print(
            f"[VAL e{epoch}] start"
            + (f" num_batches={n_batches}" if n_batches is not None else "")
            + f" bs={getattr(val_loader, 'batch_size', 'na')}"
            + f" workers={getattr(val_loader, 'num_workers', 'na')}"
            + (f" max_iters={val_max_iters}" if val_max_iters and val_max_iters > 0 else "")
        )

        for vitr, vframes in enumerate(val_loader):
            if val_max_iters and val_max_iters > 0 and vitr >= val_max_iters:
                break
            vframes = vframes.to(device, non_blocking=True)
            h_v = forward_target(vframes)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                cond_v = model.gazelle_condition_tokens(vframes) if use_gazelle else None
            z_tf_v, z_ar_v = forward_predictions(h_v, cond_full=cond_v)
            j_v = loss_fn(z_tf_v, h_v)
            s_v = loss_fn(z_ar_v, h_v)
            l_v = j_v + s_v
            loss_m.update(float(l_v.detach()), n=int(vframes.size(0)))
            tf_m.update(float(j_v.detach()), n=int(vframes.size(0)))
            ar_m.update(float(s_v.detach()), n=int(vframes.size(0)))

            if val_log_freq > 0 and (vitr % val_log_freq == 0):
                print(
                    f"[VAL e{epoch} it{vitr}] loss={loss_m.avg:.6f} tf={tf_m.avg:.6f} ar={ar_m.avg:.6f}"
                )

        print(f"[VAL e{epoch}] loss={loss_m.avg:.6f} tf={tf_m.avg:.6f} ar={ar_m.avg:.6f}")
        wandb_logger.log(
            {
                "epoch": epoch,
                "val/loss": float(loss_m.avg),
                "val/tf_loss": float(tf_m.avg),
                "val/ar_loss": float(ar_m.avg),
            },
            step=global_step,
        )

        model.predictor_causal.train()
        if use_gazelle:
            model._scene_proj.train()

    # -------------------------
    # Training loop (TF + AR)
    # -------------------------
    global_step = 0
    start_epoch = _try_resume()
    eval_freq = int(cfgs_meta.get("eval_freq", 1))
    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        it = iter(loader)
        epoch_ipe = ipe if ipe is not None else len(loader)
        for itr in range(epoch_ipe):
            try:
                frames = next(it).to(device, non_blocking=True)  # [B,C,T,H,W]
            except StopIteration:
                it = iter(loader)
                frames = next(it).to(device, non_blocking=True)

            B, _, T, _, _ = frames.shape
            if T != frames_per_clip:
                raise ValueError(
                    f"Expected T==frames_per_clip ({frames_per_clip}), got T={T}. "
                    f"Check dataset frames_per_clip and any frameskip settings."
                )

            h = forward_target(frames)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                cond_full = model.gazelle_condition_tokens(frames) if use_gazelle else None
            z_tf, z_ar = forward_predictions(h, cond_full=cond_full)
            jloss = loss_fn(z_tf, h)
            sloss = loss_fn(z_ar, h)
            loss = jloss + sloss

            # Step 2. Backward & step (match original style)
            if mixed_precision:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and (itr % 10 == 0 or itr == epoch_ipe - 1):
                _loss = float((jloss + sloss).detach())
                _tf = float(jloss.detach())
                _ar = float(sloss.detach())
                print(
                    f"[e{epoch} it{itr} step{global_step}] frames={tuple(frames.shape)} "
                    f"loss={_loss:.6f} tf={_tf:.6f} ar={_ar:.6f} "
                    f"(mixed_precision={mixed_precision}, dtype={which_dtype}, sdpa={use_sdpa})"
                )
                wandb_logger.log(
                    {
                        "epoch": epoch,
                        "iter": itr,
                        "train/loss": _loss,
                        "train/tf_loss": _tf,
                        "train/ar_loss": _ar,
                        "train/batch_size": int(frames.shape[0]),
                    },
                    step=global_step,
                )

            global_step += 1

        _save_checkpoint(epoch_1based=epoch + 1)
        if eval_freq > 0 and ((epoch + 1) % eval_freq == 0):
            run_validation(epoch=epoch + 1, global_step=global_step)

    wandb_logger.finish()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fname", type=str, required=True, help="Path to YAML config file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.fname, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)


