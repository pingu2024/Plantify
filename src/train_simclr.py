from __future__ import annotations

import os
import argparse
import time
from pathlib import Path

# Helps on macOS where some ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.dataset import make_simclr_dataset, make_simclr_loader
from src.model import SimCLR, build_resnet18_encoder, nt_xent_loss
from src.utils import AverageMeter, ensure_dir, get_device, resolve_path, set_deterministic, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    writer: SummaryWriter | None,
    global_step: int,
    epoch: int,
    log_every_steps: int,
) -> tuple[float, int]:
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc=f"SimCLR e{epoch}", leave=False)
    for step, ((x1, x2), _) in pbar:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        # AMP is reliable on CUDA; MPS autocast support varies by torch build.
        use_amp = device.type == "cuda"
        with autocast(device_type="cuda", enabled=use_amp):
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x1.size(0)
        loss_meter.update(loss.item(), n=bs)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if writer and step % max(1, log_every_steps) == 0:
            writer.add_scalar("train/loss", loss_meter.avg, global_step)
        global_step += 1

    return loss_meter.avg, global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/simclr.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(resolve_path(repo_root, args.config)).read_text())

    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    set_deterministic(bool(cfg["project"].get("deterministic", False)))

    device = get_device()
    print(f"Device: {device}")

    pv_root = resolve_path(repo_root, cfg["paths"]["plantvillage_root"])
    image_size = int(cfg["data"]["image_size"])
    ds = make_simclr_dataset(pv_root, image_size=image_size)
    loader = make_simclr_loader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2) or 2),
        drop_last=bool(cfg["ssl"].get("drop_last", True)),
    )

    encoder = build_resnet18_encoder(imagenet_pretrained=bool(cfg["model"]["imagenet_pretrained"]))
    model = SimCLR(
        encoder,
        projection_dim=int(cfg["model"]["projection_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
    ).to(device)

    lr = float(cfg["ssl"]["lr"])
    wd = float(cfg["ssl"]["weight_decay"])
    opt_name = str(cfg["ssl"]["optimizer"]).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg["ssl"]["momentum"]),
            weight_decay=wd,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    run_name = args.run_name or f"simclr_{int(time.time())}"
    runs_dir = ensure_dir(resolve_path(repo_root, cfg["paths"]["runs_dir"]))
    ckpt_dir = ensure_dir(resolve_path(repo_root, cfg["paths"]["checkpoints_dir"]))

    writer = None
    if bool(cfg["logging"]["tensorboard"]):
        writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    temperature = float(cfg["ssl"]["temperature"])
    epochs = int(cfg["ssl"]["epochs"])
    log_every = int(cfg["logging"]["log_every_steps"])
    global_step = 0
    best_loss = float("inf")

    print(f"SimCLR | samples {len(ds)} | steps/epoch {len(loader)}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, global_step = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            writer=writer,
            global_step=global_step,
            epoch=epoch,
            log_every_steps=log_every,
        )
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs} | train loss {loss:.4f} | {dt:.1f}s")

        if writer:
            writer.add_scalar("epoch/train_loss", loss, epoch)

        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "encoder_state": model.encoder.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "train_loss": loss,
                },
                ckpt_dir / f"{run_name}_best.pt",
            )

    last_path = ckpt_dir / f"{run_name}_last.pt"
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "config": cfg,
            "epoch": epochs,
            "train_loss": loss,
        },
        last_path,
    )
    print(f"Saved encoder checkpoints to {ckpt_dir} ({run_name}_best.pt, {run_name}_last.pt)")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
