from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.dataset import make_imagefolder_splits, make_loaders
from src.model import build_resnet18
from src.utils import (
    AverageMeter,
    accuracy_top1,
    count_parameters,
    ensure_dir,
    get_device,
    resolve_path,
    set_seed,
    set_deterministic,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every_steps: int,
    writer: SummaryWriter | None,
    global_step: int,
    epoch: int,
) -> tuple[float, float, int]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc=f"Train e{epoch}", leave=False)
    for step, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        loss_meter.update(loss.item(), n=bs)
        acc_meter.update(accuracy_top1(logits, targets), n=bs)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

        if writer and (step % max(1, log_every_steps) == 0):
            writer.add_scalar("train/loss", loss_meter.avg, global_step)
            writer.add_scalar("train/acc_top1", acc_meter.avg, global_step)

        global_step += 1

    return loss_meter.avg, acc_meter.avg, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        bs = targets.size(0)
        loss_meter.update(loss.item(), n=bs)
        acc_meter.update(accuracy_top1(logits, targets), n=bs)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(resolve_path(repo_root, args.config))
    cfg = yaml.safe_load(config_path.read_text())

    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    set_deterministic(bool(cfg["project"].get("deterministic", False)))

    device = get_device()
    print(f"Device: {device}")

    plantvillage_root = resolve_path(repo_root, cfg["paths"]["plantvillage_root"])
    splits = make_imagefolder_splits(
        root=plantvillage_root,
        image_size=int(cfg["data"]["image_size"]),
        seed=seed,
        split_train=float(cfg["data"]["split"]["train"]),
        split_val=float(cfg["data"]["split"]["val"]),
        split_test=float(cfg["data"]["split"]["test"]),
    )
    train_loader, val_loader, test_loader = make_loaders(
        splits=splits,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2) or 2),
    )

    num_classes = len(splits.class_names)
    model = build_resnet18(
        num_classes=num_classes,
        imagenet_pretrained=bool(cfg["model"]["imagenet_pretrained"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    print(f"Classes: {num_classes}")
    print(f"Trainable params: {count_parameters(model):,}")
    print(
        f"Dataset sizes | train {len(splits.train)} | val {len(splits.val)} | test {len(splits.test)} "
        f"| steps/epoch {len(train_loader)}"
    )

    label_smoothing = float(cfg["train"]["label_smoothing"])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    opt_name = str(cfg["train"]["optimizer"]).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg["train"]["momentum"]),
            weight_decay=wd,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    run_name = args.run_name
    if not run_name:
        run_name = f"pv_sup_{int(time.time())}"

    runs_dir = ensure_dir(resolve_path(repo_root, cfg["paths"]["runs_dir"]))
    ckpt_dir = ensure_dir(resolve_path(repo_root, cfg["paths"]["checkpoints_dir"]))

    writer = None
    if bool(cfg["logging"]["tensorboard"]):
        writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    best_val_acc = -1.0
    global_step = 0
    epochs = int(cfg["train"]["epochs"])
    log_every_steps = int(cfg["logging"]["log_every_steps"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_every_steps=log_every_steps,
            writer=writer,
            global_step=global_step,
            epoch=epoch,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc=f"Val e{epoch}")

        if writer:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/train_acc_top1", train_acc, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_acc_top1", val_acc, epoch)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / f"{run_name}_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": splits.class_names,
                    "config": cfg,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{'saved' if improved else '':5s} | {dt:.1f}s"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()

