from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.dataset import cap_train_samples_per_class, make_imagefolder_splits, make_loaders
from src.model import build_resnet18, load_encoder_weights_into_classifier
from src.train_supervised import evaluate, train_one_epoch
from src.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    resolve_path,
    set_deterministic,
    set_seed,
)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("fc"):
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def build_model_for_finetune(
    init: str,
    num_classes: int,
    pretrained_path: str | None,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    init = init.lower().strip()
    if init == "imagenet":
        return build_resnet18(
            num_classes=num_classes,
            imagenet_pretrained=True,
            dropout=dropout,
        ).to(device)

    if init == "scratch":
        return build_resnet18(
            num_classes=num_classes,
            imagenet_pretrained=False,
            dropout=dropout,
        ).to(device)

    if pretrained_path is None or not Path(pretrained_path).is_file():
        raise FileNotFoundError(f"finetune.init={init} needs a valid pretrained_path")

    blob = torch.load(pretrained_path, map_location=device)

    if init == "ssl":
        model = build_resnet18(
            num_classes=num_classes,
            imagenet_pretrained=False,
            dropout=dropout,
        ).to(device)
        enc = blob.get("encoder_state")
        if enc is None:
            raise KeyError("SSL checkpoint must contain encoder_state")
        load_encoder_weights_into_classifier(model, enc, strict_encoder=True)
        return model

    if init == "supervised_pv":
        model = build_resnet18(
            num_classes=num_classes,
            imagenet_pretrained=False,
            dropout=dropout,
        ).to(device)
        st = blob.get("model_state")
        if st is None:
            raise KeyError("Supervised checkpoint must contain model_state")
        enc = {k: v for k, v in st.items() if not k.startswith("fc")}
        missing, unexpected = model.load_state_dict(enc, strict=False)
        bad = [k for k in missing if not k.startswith("fc")]
        if bad:
            raise RuntimeError(f"Incomplete backbone load; missing: {bad[:8]}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected[:8]}")
        return model

    raise ValueError(f"Unknown finetune.init: {init}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(resolve_path(repo_root, args.config)).read_text())

    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    set_deterministic(bool(cfg["project"].get("deterministic", False)))

    device = get_device()
    print(f"Device: {device}")

    target_root = resolve_path(repo_root, cfg["paths"]["target_root"])
    splits = make_imagefolder_splits(
        root=target_root,
        image_size=int(cfg["data"]["image_size"]),
        seed=seed,
        split_train=float(cfg["data"]["split"]["train"]),
        split_val=float(cfg["data"]["split"]["val"]),
        split_test=float(cfg["data"]["split"]["test"]),
    )

    few = cfg["data"].get("fewshot_per_class")
    few_n = int(few) if few is not None and str(few).strip() != "" else None
    if few_n is not None:
        splits.train = cap_train_samples_per_class(splits.train, few_n, seed=seed)

    train_loader, val_loader, test_loader = make_loaders(
        splits=splits,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        prefetch_factor=int(cfg["data"].get("prefetch_factor", 2) or 2),
    )

    num_classes = len(splits.class_names)
    init = str(cfg["finetune"]["init"])
    pre_path = cfg["paths"].get("pretrained_path")
    pre_resolved = resolve_path(repo_root, pre_path) if pre_path else None

    model = build_model_for_finetune(
        init=init,
        num_classes=num_classes,
        pretrained_path=pre_resolved,
        dropout=float(cfg["finetune"]["dropout"]),
        device=device,
    )

    freeze_epochs = int(cfg["finetune"].get("freeze_backbone_epochs", 0))
    print(f"Classes: {num_classes} | init: {init} | trainable params: {count_parameters(model):,}")
    print(
        f"Dataset sizes | train {len(splits.train)} | val {len(splits.val)} | test {len(splits.test)} "
        f"| steps/epoch {len(train_loader)}"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["finetune"]["label_smoothing"]))
    lr = float(cfg["finetune"]["lr"])
    wd = float(cfg["finetune"]["weight_decay"])
    opt_name = str(cfg["finetune"]["optimizer"]).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg["finetune"]["momentum"]),
            weight_decay=wd,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    run_name = args.run_name or f"ft_{init}_{int(time.time())}"
    runs_dir = ensure_dir(resolve_path(repo_root, cfg["paths"]["runs_dir"]))
    ckpt_dir = ensure_dir(resolve_path(repo_root, cfg["paths"]["checkpoints_dir"]))

    writer = None
    if bool(cfg["logging"]["tensorboard"]):
        writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    epochs = int(cfg["finetune"]["epochs"])
    log_every = int(cfg["logging"]["log_every_steps"])
    global_step = 0
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        frozen = freeze_epochs > 0 and epoch <= freeze_epochs
        set_backbone_trainable(model, trainable=not frozen)
        if epoch == 1 or (freeze_epochs > 0 and epoch == freeze_epochs + 1):
            print(f"Epoch {epoch}: backbone {'frozen' if frozen else 'unfrozen'}")

        t0 = time.time()
        train_loss, train_acc, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_every_steps=log_every,
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

        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": splits.class_names,
                    "config": cfg,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "finetune_init": init,
                },
                ckpt_dir / f"{run_name}_best.pt",
            )

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} | train {train_loss:.4f} acc {train_acc:.4f} | "
            f"val {val_loss:.4f} acc {val_acc:.4f} | {'saved' if improved else ''} | {dt:.1f}s"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
