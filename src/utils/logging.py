# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = []
        mode = kwargs.get("mode", "+a")
        self.delim = kwargs.get("delim", ",")
        # -- print headers
        with open(self.fname, mode) as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=self.delim, file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.delim if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


class WandbLogger(object):
    """Weights & Biases logger wrapper"""

    def __init__(self, project=None, name=None, config=None, enabled=True, **kwargs):
        """
        Initialize wandb logger
        
        Args:
            project: wandb project name
            name: run name
            config: configuration dict to log
            enabled: whether to enable wandb logging
            **kwargs: additional arguments passed to wandb.init
        """
        self.enabled = enabled and WANDB_AVAILABLE
        
        if self.enabled:
            try:
                wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    **kwargs
                )
                get_logger(__name__).info(f"Wandb initialized: project={project}, name={name}")
            except Exception as e:
                get_logger(__name__).warning(f"Failed to initialize wandb: {e}")
                self.enabled = False
        elif not WANDB_AVAILABLE:
            get_logger(__name__).warning("wandb not installed, logging disabled")

    def log(self, metrics, step=None, commit=True):
        """Log metrics to wandb"""
        if self.enabled:
            try:
                wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                get_logger(__name__).warning(f"Failed to log to wandb: {e}")

    def finish(self):
        """Finish wandb run"""
        if self.enabled:
            try:
                wandb.finish()
            except Exception:
                pass


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jepa_rootpath():
    this_file = os.path.abspath(__file__)
    return "/".join(this_file.split("/")[:-3])


def git_information():
    jepa_root = jepa_rootpath()
    try:
        resp = (
            subprocess.check_output(["git", "-C", jepa_root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        commit, branch = resp.split("\n")
        return f"branch: {branch}\ncommit: {commit}\n"
    except Exception:
        return "unknown"
