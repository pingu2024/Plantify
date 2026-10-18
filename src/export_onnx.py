"""
Export a classification checkpoint to ONNX and check it against PyTorch using OpenCV DNN.

Writes <out_dir>/<name>.onnx, labels.txt (one class per line, model output order) and
model_info.json (input size + normalisation), which is all src/infer_opencv.py needs.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.cv_preprocess import MEAN, STD, preprocess_for_model
from src.inference import eval_transform, load_classifier_checkpoint
from src.utils import resolve_path


def export(model: torch.nn.Module, image_size: int, out_path: Path, opset: int) -> None:
    dummy = torch.randn(1, 3, image_size, image_size)
    # Fixed batch of 1: OpenCV's ONNX importer is happiest with static shapes.
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )


def check_same_input(model: torch.nn.Module, net: cv2.dnn.Net, image_size: int) -> float:
    """Feed one identical tensor to both runtimes; returns max abs logit difference."""
    x = np.random.RandomState(0).randn(1, 3, image_size, image_size).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    net.setInput(x)
    out = net.forward()
    return float(np.abs(ref - out).max())


def check_real_images(
    model: torch.nn.Module,
    net: cv2.dnn.Net,
    image_size: int,
    data_root: Path,
    per_class: int,
) -> tuple[int, int]:
    """Original PIL/torchvision pipeline vs. pure OpenCV pipeline on real images."""
    tfm = eval_transform(image_size)
    rng = random.Random(0)
    files: list[str] = []
    for cls_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        imgs = sorted(glob.glob(str(cls_dir / "*")))
        files += rng.sample(imgs, min(per_class, len(imgs)))

    agree = 0
    for f in files:
        with torch.no_grad():
            p_torch = int(model(tfm(Image.open(f).convert("RGB")).unsqueeze(0)).argmax(1))
        net.setInput(preprocess_for_model(cv2.imread(f), image_size))
        p_cv = int(net.forward()[0].argmax())
        agree += int(p_torch == p_cv)
    return agree, len(files)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="exports")
    p.add_argument("--name", default=None, help="Output file stem (defaults to checkpoint name)")
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--check_data", default="data/corn_leaf", help="ImageFolder used for the parity check")
    p.add_argument("--check_per_class", type=int, default=20)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ckpt = Path(resolve_path(repo_root, args.ckpt))
    out_dir = Path(resolve_path(repo_root, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, image_size = load_classifier_checkpoint(ckpt, device=torch.device("cpu"))
    stem = args.name or ckpt.stem.replace("_best", "")
    onnx_path = out_dir / f"{stem}.onnx"

    export(model, image_size, onnx_path, args.opset)
    (out_dir / "labels.txt").write_text("\n".join(class_names) + "\n")
    (out_dir / "model_info.json").write_text(
        json.dumps(
            {
                "onnx": onnx_path.name,
                "source_checkpoint": ckpt.name,
                "input": [1, 3, image_size, image_size],
                "color_order": "RGB",
                "mean": MEAN,
                "std": STD,
                "classes": class_names,
            },
            indent=2,
        )
    )
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), input 1x3x{image_size}x{image_size}")

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    diff = check_same_input(model, net, image_size)
    print(f"OpenCV DNN vs PyTorch, identical input: max |logit diff| = {diff:.2e}")

    data_root = Path(resolve_path(repo_root, args.check_data))
    if data_root.is_dir():
        agree, n = check_real_images(model, net, image_size, data_root, args.check_per_class)
        print(f"PIL+torchvision vs OpenCV preprocessing, real images: top-1 agreement {agree}/{n} ({agree / n:.1%})")


if __name__ == "__main__":
    main()
