# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import random
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from src.datasets.utils.worker_init_fn import pl_worker_init_function

multiprocessing.set_start_method("spawn", force=True)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards_list):
    num_shards = len(shards_list)
    total_size = num_shards
    return total_size, num_shards


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class split_by_node(wds.PipelineStage):
    """Node splitter that uses provided rank/world_size instead of from torch.distributed"""

    def __init__(
        self,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size

    def run(self, src):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


class decode_videos_to_clips(wds.PipelineStage):

    def __init__(
        self,
        annotations,
        frames_per_clip=16,
        fps=5,
        transform=None,
        anticipation_time_sec=[0.0, 0.0],
        anticipation_point=[0.25, 0.75],
        heatmap_base_path=None,
        use_heatmap=False,
    ):
        self.annotations = annotations
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.anticipation_time = anticipation_time_sec
        self.anticipation_point = anticipation_point
        self.heatmap_base_path = heatmap_base_path
        self.use_heatmap = use_heatmap

    def run(self, src):
        for path in src:
            # -- get all action annotations for the video
            video_id = path.split("/")[-1].split(".")[0]
            ano = self.annotations[video_id]

            # -- get action annotations and frame stamps
            start_frames = ano["start_frame"].values
            stop_frames = ano["stop_frame"].values

            # -- load clips corresponding to action annotations
            try:
                vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
                vr.seek(0)
                # --
                vfps = vr.get_avg_fps()
                fpc = self.frames_per_clip
                fstp = int(vfps / self.fps)
                nframes = int(fpc * fstp)
            except Exception as e:
                logging.info(f"Encountered exception loading video {e=}")
                continue

            for i, (sf, ef) in enumerate(zip(start_frames, stop_frames)):
                labels_verb = int(ano["verb_class"].values[i])
                labels_noun = int(ano["noun_class"].values[i])

                # sample an anticipation time
                at = random.uniform(*self.anticipation_time)
                aframes = int(at * vfps)

                # sample an anticipation frame b/w start and end of action
                ap = random.uniform(*self.anticipation_point)
                af = int(sf * ap + (1 - ap) * ef - aframes)

                indices = np.arange(af - nframes, af, fstp).astype(np.int64)
                # If not enough frames in video for anticipation, just pad with
                # first frame
                indices[indices < 0] = 0

                try:
                    buffer = vr.get_batch(indices).asnumpy()
                except Exception as e:
                    logging.info(f"Encountered exception getting indices {e=}")
                    continue

                if self.transform is not None:
                    buffer = self.transform(buffer)

                # Load gaze heatmap if enabled
                heatmap = None
                if self.use_heatmap and self.heatmap_base_path is not None:
                    try:
                        # Construct heatmap path: heatmap_base_path/video_id/frame_indices.npy
                        # or heatmap_base_path/video_id.npz containing all frames
                        heatmap_path = os.path.join(self.heatmap_base_path, video_id)
                        if os.path.exists(heatmap_path + ".npz"):
                            # Load from npz file
                            heatmap_data = np.load(heatmap_path + ".npz")
                            # Assume the key is 'heatmap' or similar
                            heatmap_key = list(heatmap_data.keys())[0]
                            all_heatmaps = heatmap_data[heatmap_key]
                            # Extract heatmaps for the same frame indices
                            # Ensure indices are within bounds
                            valid_indices = np.clip(indices, 0, len(all_heatmaps) - 1)
                            heatmap = all_heatmaps[valid_indices]
                            # Ensure heatmap is [T, H, W] format
                            if len(heatmap.shape) == 3 and heatmap.shape[0] == len(indices):
                                pass  # Already correct shape
                            elif len(heatmap.shape) == 4:
                                # If [T, 1, H, W], squeeze channel dimension
                                heatmap = heatmap.squeeze(1)
                        elif os.path.isdir(heatmap_path):
                            # Load individual frame heatmaps
                            heatmap_frames = []
                            for idx in indices:
                                frame_heatmap_path = os.path.join(heatmap_path, f"frame_{idx:06d}.npy")
                                if os.path.exists(frame_heatmap_path):
                                    heatmap_frames.append(np.load(frame_heatmap_path))
                                else:
                                    # If frame not found, use zero heatmap
                                    # You may need to adjust shape based on your heatmap format
                                    heatmap_frames.append(np.zeros((224, 224), dtype=np.float32))
                            heatmap = np.stack(heatmap_frames, axis=0)
                        else:
                            logging.info(f"Heatmap path not found: {heatmap_path}")
                            # Create zero heatmap as fallback
                            heatmap = np.zeros((len(indices), 224, 224), dtype=np.float32)
                    except Exception as e:
                        logging.info(f"Encountered exception loading heatmap {e=}")
                        # Create zero heatmap as fallback
                        heatmap = np.zeros((len(indices), 224, 224), dtype=np.float32)

                output_dict = dict(
                    video=buffer,
                    verb=labels_verb,
                    noun=labels_noun,
                    anticipation_time=at,
                )
                if self.use_heatmap:
                    output_dict["heatmap"] = heatmap
                
                yield output_dict


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, epoch, training):
        super().__init__()
        self.epoch = epoch
        self.training = training
        self.urls = np.array(urls)
        logging.info("Done initializing ResampledShards")

    def __iter__(self):
        """Return an iterator over the shards."""
        if self.training:
            epoch = self.epoch.get_value()
            gen = torch.Generator()
            gen.manual_seed(epoch)
            yield from self.urls[torch.randperm(len(self.urls), generator=gen)]
        else:
            yield from self.urls[torch.arange(len(self.urls))]


def get_video_wds_dataset(
    batch_size,
    input_shards,
    video_decoder,
    training,
    epoch=0,
    world_size=1,
    rank=0,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    use_heatmap=False,
):
    assert input_shards is not None
    _, num_shards = get_dataset_size(input_shards)
    logging.info(f"Total number of shards across all data is {num_shards=}")

    epoch = SharedEpoch(epoch=epoch)
    # Build tuple keys based on whether heatmap is used
    tuple_keys = ["video", "verb", "noun", "anticipation_time"]
    if use_heatmap:
        tuple_keys.append("heatmap")
    
    pipeline = [
        ResampledShards(input_shards, epoch=epoch, training=training),
        split_by_node(rank=rank, world_size=world_size),
        wds.split_by_worker,
        video_decoder,
        wds.to_tuple(*tuple_keys),
        wds.batched(batch_size, partial=True, collation_fn=torch.utils.data.default_collate),
    ]
    dataset = wds.DataPipeline(*pipeline)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        worker_init_fn=pl_worker_init_function,
        pin_memory=pin_memory,
    )

    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def filter_annotations(
    base_path,
    train_annotations_path,
    val_annotations_path,
    file_format=1,
):

    tdf = pd.read_csv(train_annotations_path)
    vdf = pd.read_csv(val_annotations_path)

    # 1. Remove actions in val that are not in train
    tactions = set([(v, n) for v, n in zip(tdf["verb_class"].values, tdf["noun_class"].values)])
    tverbs = set([v for v, _ in tactions])
    tnouns = set([n for _, n in tactions])
    keep_inds = [(v, n) in tactions for v, n in zip(vdf["verb_class"].values, vdf["noun_class"].values)]
    vdf = vdf[keep_inds]

    # 2. Determine new class labels
    verb_classes = {k: i for i, k in enumerate(tverbs)}
    noun_classes = {k: i for i, k in enumerate(tnouns)}
    action_classes = {k: i for i, k in enumerate(tactions)}

    val_verb_classes = set([verb_classes[v] for v in vdf["verb_class"].values])
    val_noun_classes = set([noun_classes[n] for n in vdf["noun_class"].values])
    val_action_classes = set([action_classes[a] for a in zip(vdf["verb_class"].values, vdf["noun_class"].values)])

    def build_annotations(df):
        video_paths, annotations = [], {}
        unique_videos = list(dict.fromkeys(df["video_id"].values))
        for uv in unique_videos:
            pid = uv.split("_")[0]
            # There are two common file formats for storing EK
            if file_format == 0:
                # File format 0: $base_path/$participant_id/videos/$video_id.MP4
                fpath = os.path.join(base_path, pid, "videos", uv + ".MP4")
            else:
                # File format 1: $base_path/$participant_id/$video_id.MP4
                fpath = os.path.join(base_path, pid, uv + ".MP4")
            if not os.path.exists(fpath):
                logging.info(f"file path not found {fpath=}")
                continue
            video_paths += [fpath]
            annotations[uv] = df[df["video_id"] == uv].sort_values(by="start_frame")
        return video_paths, annotations

    train_annotations = build_annotations(tdf)
    val_annotations = build_annotations(vdf)

    return dict(
        verbs=verb_classes,
        nouns=noun_classes,
        actions=action_classes,
        val_verbs=val_verb_classes,
        val_nouns=val_noun_classes,
        val_actions=val_action_classes,
        train=train_annotations,
        val=val_annotations,
    )


def make_webvid(
    base_path,
    annotations_path,
    batch_size,
    transform,
    frames_per_clip=16,
    fps=5,
    num_workers=8,
    world_size=1,
    rank=0,
    anticipation_time_sec=[0.0, 0.0],
    persistent_workers=True,
    pin_memory=True,
    training=True,
    anticipation_point=[0.1, 0.1],
    heatmap_base_path=None,
    use_heatmap=False,
    **kwargs,
):

    paths, annotations = annotations_path
    num_clips = sum([len(a) for a in annotations.values()])

    video_decoder = decode_videos_to_clips(
        annotations=annotations,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        anticipation_time_sec=anticipation_time_sec,
        anticipation_point=anticipation_point,
        heatmap_base_path=heatmap_base_path,
        use_heatmap=use_heatmap,
    )

    dataset, datainfo = get_video_wds_dataset(
        batch_size=batch_size,
        input_shards=paths,
        epoch=0,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        video_decoder=video_decoder,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        training=training,
        use_heatmap=use_heatmap,
    )

    datainfo.dataloader.num_batches = num_clips // (world_size * batch_size)
    datainfo.dataloader.num_samples = num_clips

    return dataset, datainfo.dataloader, datainfo
