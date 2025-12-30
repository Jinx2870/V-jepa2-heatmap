# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from math import ceil
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu


_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
):
    dataset = Ego4DVideoACDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        frameskip=tubelet_size,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("Ego4DVideoACDataset data loader created")

    return data_loader, dist_sampler


class Ego4DVideoACDataset(torch.utils.data.Dataset):
    """Video dataset adapter for AC training on Ego4D.

    CSV format: each line contains a single absolute MP4 path.
    Returns 5-tuple per sample to match DROID interface:
      (buffer [C,T,H,W], actions [T-1,7], states [T,7], extrinsics [T,7], indices [np.ndarray])
    """

    def __init__(
        self,
        data_path,
        frameskip=1,
        frames_per_clip=16,
        fps=5,
        transform=None,
    ):
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load list of absolute MP4 paths
        # Accept both comma-separated and whitespace-separated single-column files
        df = pd.read_csv(data_path, header=None)
        samples = list(df.iloc[:, 0].astype(str).values)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]

        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self._loadvideo_decord(path)
                loaded_video = True
            except Exception as e:
                logging.info(f"Encountered exception when loading video {path=} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                path = self.samples[index]

        return buffer, actions, states, extrinsics, indices

    def _loadvideo_decord(self, vpath):
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))

        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # sample a random window of nframes
        ef = np.random.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

        # fetch frames
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        if self.transform is not None:
            buffer = self.transform(buffer)  # -> [C, T, H, W]

        # Build zero-condition placeholders
        T = indices.shape[0] if self.frameskip <= 1 else int(np.ceil(indices.shape[0] / self.frameskip))
        states = np.zeros((T, 7), dtype=np.float32)
        actions = np.zeros((max(T - 1, 0), 7), dtype=np.float32)
        extrinsics = np.zeros((T, 7), dtype=np.float32)

        return buffer, actions, states, extrinsics, indices


