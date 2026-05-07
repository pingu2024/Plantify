from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class DataSplits:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: list[str]


class TwoCropsTransform:
    """Returns two independent augmentations of the same image (SimCLR)."""

    def __init__(self, transform: transforms.Compose) -> None:
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def build_plantvillage_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return train_tfms, eval_tfms


def build_simclr_transforms(image_size: int) -> transforms.Compose:
    """Stronger augments for contrastive pretraining."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def make_imagefolder_splits(
    root: str | Path,
    image_size: int,
    seed: int,
    split_train: float,
    split_val: float,
    split_test: float,
) -> DataSplits:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    if abs((split_train + split_val + split_test) - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")

    train_tfms, eval_tfms = build_plantvillage_transforms(image_size=image_size)

    full_train = datasets.ImageFolder(root=str(root), transform=train_tfms)
    full_eval = datasets.ImageFolder(root=str(root), transform=eval_tfms)

    n = len(full_train)
    n_train = int(n * split_train)
    n_val = int(n * split_val)
    n_test = n - n_train - n_val

    g = torch.Generator().manual_seed(seed)
    idx_train, idx_val, idx_test = random_split(range(n), [n_train, n_val, n_test], generator=g)

    train_ds = Subset(full_train, idx_train.indices)
    val_ds = Subset(full_eval, idx_val.indices)
    test_ds = Subset(full_eval, idx_test.indices)

    return DataSplits(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        class_names=list(full_train.classes),
    )


def cap_train_samples_per_class(train_subset: Subset, max_per_class: int | None, seed: int) -> Subset:
    """Limit labeled training images per class (few-shot fine-tuning)."""
    if max_per_class is None or max_per_class <= 0:
        return train_subset

    base: datasets.ImageFolder = train_subset.dataset  # type: ignore[assignment]
    indices_by_class: dict[int, list[int]] = {}
    for idx in train_subset.indices:
        y = int(base.targets[idx])
        indices_by_class.setdefault(y, []).append(idx)

    rng = random.Random(seed)
    kept: list[int] = []
    for _, idxs in sorted(indices_by_class.items()):
        rng.shuffle(idxs)
        kept.extend(idxs[:max_per_class])

    return Subset(base, sorted(kept))


def make_simclr_dataset(root: str | Path, image_size: int) -> Dataset:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    aug = build_simclr_transforms(image_size=image_size)
    return datasets.ImageFolder(root=str(root), transform=TwoCropsTransform(aug))


def make_loaders(
    splits: DataSplits,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    persistent_workers = bool(num_workers and num_workers > 0)
    pf = prefetch_factor if persistent_workers and prefetch_factor else None
    kw: dict = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if pf is not None:
        kw["prefetch_factor"] = pf

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **kw,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kw,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **kw,
    )
    return train_loader, val_loader, test_loader


def make_simclr_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None = 2,
    drop_last: bool = True,
) -> DataLoader:
    persistent_workers = bool(num_workers and num_workers > 0)
    pf = prefetch_factor if persistent_workers and prefetch_factor else None
    kw: dict = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        shuffle=True,
        drop_last=drop_last,
    )
    if pf is not None:
        kw["prefetch_factor"] = pf
    return DataLoader(dataset, batch_size=batch_size, **kw)
