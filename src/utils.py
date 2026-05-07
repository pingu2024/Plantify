from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism costs performance. We keep the toggles optional and conservative.
    # (cuDNN flags matter only when running on CUDA.)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def set_deterministic(deterministic: bool) -> None:
    # Only meaningful for CUDA; harmless elsewhere.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def get_device() -> torch.device:
    # Prefer Apple Silicon acceleration if available, else CUDA, else CPU.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_path(repo_root: str | os.PathLike[str], maybe_relative: str) -> str:
    p = Path(maybe_relative)
    if p.is_absolute():
        return str(p)
    return str(Path(repo_root) / p)


def as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default
