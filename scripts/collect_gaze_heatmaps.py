#!/usr/bin/env python3
"""
批量遍历外部 results 目录，将每个视频子目录中的 gaze.npy 整理成 V-JEPA 2 heatmap 输入格式。

支持两种输出格式：
1. 单文件 .npz：{output_root}/{video_id}.npz（键名默认为 heatmap，形状 [T, H, W]）
2. 逐帧 .npy：{output_root}/{video_id}/frame_000000.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

LOGGER = logging.getLogger("collect_gaze_heatmaps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="遍历 results 目录，将各视频的 gaze.npy 整理为 heatmap 数据集"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="results 根目录，每个子目录对应一个视频，内部包含 gaze.npy 或其它热力图文件",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="整理后的输出目录，供 configs 中 heatmap_base_path 使用",
    )
    parser.add_argument(
        "--gaze-filename",
        default="gaze.npy",
        help="每个视频目录内的热力图文件名（默认 gaze.npy）",
    )
    parser.add_argument(
        "--output-format",
        choices=("npz", "frames"),
        default="npz",
        help="输出格式：npz 单文件；frames 逐帧 .npy",
    )
    parser.add_argument(
        "--npz-key",
        default="heatmap",
        help="保存 .npz 时使用的键名",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="输出数组 dtype（例如 float32）；默认保持原始 dtype",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印即将执行的操作，不写入文件",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印调试日志",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


def iter_video_dirs(input_root: Path) -> Iterable[Path]:
    for path in sorted(input_root.iterdir()):
        if path.is_dir():
            yield path


def load_gaze_file(video_dir: Path, filename: str) -> np.ndarray:
    gaze_path = video_dir / filename
    if not gaze_path.exists():
        raise FileNotFoundError(f"{gaze_path} 不存在")

    array = np.load(gaze_path, allow_pickle=False)
    if array.ndim not in (2, 3):
        raise ValueError(f"{gaze_path} 的 shape 为 {array.shape}，无法转换为 [T, H, W]")

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    return array


def ensure_clean_path(target: Path, overwrite: bool, dry_run: bool) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} 已存在，添加 --overwrite 以覆盖")
        if dry_run:
            LOGGER.info("将覆盖 %s", target)
        else:
            if target.is_file():
                target.unlink()
            else:
                for child in target.glob("**/*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(target.glob("**"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
    else:
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, array: np.ndarray, key: str, dtype: str | None, dry_run: bool) -> None:
    if dtype:
        array = array.astype(dtype)
    if dry_run:
        LOGGER.info("模拟保存 .npz -> %s shape=%s dtype=%s", path, array.shape, array.dtype)
    else:
        np.savez_compressed(path, **{key: array})


def save_frames(directory: Path, array: np.ndarray, dtype: str | None, dry_run: bool) -> None:
    if dtype:
        array = array.astype(dtype)
    if dry_run:
        LOGGER.info("模拟写入目录 -> %s (共 %d 帧)", directory, array.shape[0])
        return
    directory.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(array):
        frame_path = directory / f"frame_{idx:06d}.npy"
        np.save(frame_path, frame)


def process_video(
    video_dir: Path,
    output_root: Path,
    output_format: str,
    gaze_filename: str,
    npz_key: str,
    dtype: str | None,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[str, Tuple[int, int, int]]:
    video_id = video_dir.name
    LOGGER.info("处理 %s", video_id)
    heatmap = load_gaze_file(video_dir, gaze_filename)

    if output_format == "npz":
        output_path = output_root / f"{video_id}.npz"
        ensure_clean_path(output_path, overwrite, dry_run)
        save_npz(output_path, heatmap, npz_key, dtype, dry_run)
    else:
        output_dir = output_root / video_id
        ensure_clean_path(output_dir, overwrite, dry_run)
        save_frames(output_dir, heatmap, dtype, dry_run)

    return video_id, heatmap.shape


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        LOGGER.error("输入目录不存在：%s", input_root)
        sys.exit(1)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    video_dirs = list(iter_video_dirs(input_root))
    if not video_dirs:
        LOGGER.error("未在 %s 下找到任何子目录", input_root)
        sys.exit(1)

    processed = 0
    for video_dir in video_dirs:
        try:
            _, shape = process_video(
                video_dir=video_dir,
                output_root=output_root,
                output_format=args.output_format,
                gaze_filename=args.gaze_filename,
                npz_key=args.npz_key,
                dtype=args.dtype,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            LOGGER.debug("%s shape=%s", video_dir.name, shape)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("处理 %s 失败：%s", video_dir, exc, exc_info=args.verbose)

    LOGGER.info("完成：成功处理 %d/%d 个视频", processed, len(video_dirs))


if __name__ == "__main__":
    main()

