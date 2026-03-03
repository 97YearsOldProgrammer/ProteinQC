"""Distributed training utilities for DGX Spark multi-node setup.

Ported from GeneT5/lib/train/distributed.py with minor simplifications.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed(backend: str = "nccl") -> dict | None:
    """Initialize distributed training from torchrun env vars.

    Returns dict with rank/world_size/local_rank/is_main, or None if not
    running under torchrun.
    """
    if dist.is_initialized():
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "is_main": dist.get_rank() == 0,
        }

    if "RANK" not in os.environ:
        return None

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def wrap_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_params: bool = False,
) -> nn.Module:
    """Wrap model with DDP. No-op if not distributed."""
    if not dist.is_initialized():
        return model.to(device)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(device)
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_params,
        gradient_as_bucket_view=True,
    )


def unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DDP module."""
    if hasattr(model, "module"):
        return model.module
    return model
