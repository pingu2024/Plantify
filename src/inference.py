from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.model import build_resnet18


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_classifier_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    blob = torch.load(path, map_location=device)
    if "model_state" not in blob or "class_names" not in blob:
        raise KeyError("Checkpoint must contain model_state and class_names (train supervised or finetune).")

    class_names: list[str] = list(blob["class_names"])
    cfg = blob.get("config") or {}
    data_cfg = cfg.get("data") or {}
    image_size = int(data_cfg.get("image_size", 128))

    dropout = 0.0
    fin = cfg.get("finetune")
    if isinstance(fin, dict):
        dropout = float(fin.get("dropout", 0.0))
    if dropout == 0.0:
        mod = cfg.get("model")
        if isinstance(mod, dict):
            dropout = float(mod.get("dropout", 0.0))

    model = build_resnet18(
        num_classes=len(class_names),
        imagenet_pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(blob["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, class_names, image_size


@torch.no_grad()
def predict_pil(
    model: nn.Module,
    image: Image.Image,
    class_names: list[str],
    image_size: int,
    device: torch.device,
) -> tuple[str, float, list[tuple[str, float]]]:
    tfm = eval_transform(image_size)
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    ranked = sorted(
        [(class_names[i], float(probs[i])) for i in range(len(class_names))],
        key=lambda t: t[1],
        reverse=True,
    )
    return class_names[int(idx)], float(conf), ranked
