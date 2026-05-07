from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets

from src.dataset import build_plantvillage_transforms
from src.inference import eval_transform, load_classifier_checkpoint
from src.utils import get_device, resolve_path


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(targets.tolist())
    return y_true, y_pred


def maybe_gradcam(
    model: nn.Module,
    image_path: Path,
    image_size: int,
    class_names: list[str],
    device: torch.device,
    out_path: Path,
) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as e:
        raise RuntimeError("Install grad-cam / pytorch_grad_cam for heatmaps.") from e

    from PIL import Image

    pil_orig = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    rgb = np.array(pil_orig, dtype=np.float32) / 255.0
    x = eval_transform(image_size)(pil_orig).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
    pred = int(torch.argmax(logits, dim=1).item())

    # Grad-CAM is most reliable on CPU/CUDA; MPS often needs a CPU fallback.
    cam_device = device
    m = model
    x_cam = x
    if device.type == "mps":
        cam_device = torch.device("cpu")
        m = model.to(cam_device)
        x_cam = x.to(cam_device)

    target_layers = [m.layer4[-1]]
    cam = GradCAM(model=m, target_layers=target_layers)
    grayscale = cam(input_tensor=x_cam, targets=[ClassifierOutputTarget(pred)])[0, :]

    if device.type == "mps":
        model.to(device)

    vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(out_path)
    print(f"Saved Grad-CAM to {out_path} (pred={class_names[pred]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True, help="ImageFolder root (e.g. corn val/test)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--gradcam_image", type=str, default=None)
    parser.add_argument("--gradcam_out", type=str, default="runs/gradcam.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    device = get_device()
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else Path(resolve_path(repo_root, args.ckpt))
    data_root = Path(args.data_root) if Path(args.data_root).is_absolute() else Path(resolve_path(repo_root, args.data_root))

    model, class_names, image_size = load_classifier_checkpoint(ckpt_path, device=device)

    _, eval_tfms = build_plantvillage_transforms(image_size=image_size)
    ds = datasets.ImageFolder(root=str(data_root), transform=eval_tfms)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if set(ds.classes) != set(class_names):
        print("Warning: folder class names differ from checkpoint order; metrics use folder index order.")

    y_true, y_pred = collect_predictions(model, loader, device)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({len(y_true)} images)")
    print(classification_report(y_true, y_pred, target_names=ds.classes, zero_division=0))

    if args.gradcam_image:
        maybe_gradcam(
            model=model,
            image_path=Path(args.gradcam_image),
            image_size=image_size,
            class_names=class_names,
            device=device,
            out_path=Path(args.gradcam_out),
        )


if __name__ == "__main__":
    main()
