from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from src.dataset import build_plantvillage_transforms
from src.inference import load_classifier_checkpoint
from src.utils import get_device, resolve_path


def save_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_per_class_bars(report: dict, out: Path) -> None:
    # report is sklearn classification_report(output_dict=True)
    class_keys = [k for k in report.keys() if k not in {"accuracy", "macro avg", "weighted avg"}]
    prec = [report[k]["precision"] for k in class_keys]
    rec = [report[k]["recall"] for k in class_keys]
    f1 = [report[k]["f1-score"] for k in class_keys]

    x = np.arange(len(class_keys))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, prec, width=w, label="precision")
    ax.bar(x, rec, width=w, label="recall")
    ax.bar(x + w, f1, width=w, label="f1")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_keys, rotation=30, ha="right")
    ax.set_title("Per-class metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="Classification checkpoint (*_best.pt)")
    p.add_argument("--data_root", required=True, type=str, help="ImageFolder root to evaluate")
    p.add_argument("--out_dir", default="runs/plots", type=str)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    device = get_device()

    ckpt = Path(args.ckpt) if Path(args.ckpt).is_absolute() else Path(resolve_path(repo_root, args.ckpt))
    data_root = (
        Path(args.data_root) if Path(args.data_root).is_absolute() else Path(resolve_path(repo_root, args.data_root))
    )
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else Path(resolve_path(repo_root, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, image_size = load_classifier_checkpoint(ckpt, device=device)
    _, eval_tfms = build_plantvillage_transforms(image_size=image_size)
    ds = datasets.ImageFolder(root=str(data_root), transform=eval_tfms)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    import torch

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())

    txt = classification_report(y_true, y_pred, target_names=ds.classes, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(txt)

    report_dict = classification_report(
        y_true, y_pred, target_names=ds.classes, digits=4, zero_division=0, output_dict=True
    )

    save_confusion_matrix(y_true, y_pred, ds.classes, out=out_dir / "confusion_matrix.png")
    save_per_class_bars(report_dict, out=out_dir / "per_class_metrics.png")

    print(f"Saved plots to: {out_dir}")
    print(f"- {out_dir / 'confusion_matrix.png'}")
    print(f"- {out_dir / 'per_class_metrics.png'}")
    print(f"- {out_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    main()

