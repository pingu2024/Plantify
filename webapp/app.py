from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cv_preprocess import analyze_leaf, draw_analysis
from src.inference import eval_transform, load_classifier_checkpoint, predict_pil
from src.utils import get_device


def create_app() -> Flask:
    base = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        static_folder=str(base / "static"),
        template_folder=str(base / "templates"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    ckpt = os.environ.get("PLANTIFY_CKPT", str(REPO_ROOT / "checkpoints" / "best_for_demo.pt"))
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing classification checkpoint at {ckpt_path}. "
            f"Train/finetune first, then export PLANTIFY_CKPT=/absolute/path/to/*_best.pt"
        )
    device = get_device()
    model, class_names, image_size = load_classifier_checkpoint(ckpt_path, device=device)

    app.config["PLANTIFY_CKPT"] = ckpt
    app.config["DEVICE"] = device
    app.config["MODEL"] = model
    app.config["CLASS_NAMES"] = class_names
    app.config["IMAGE_SIZE"] = image_size

    static_dir = base / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.config["HEATMAP_PATH"] = static_dir / "last_cam.png"
    app.config["CV_PATH"] = static_dir / "last_cv.jpg"

    @app.get("/")
    def index():
        return render_template("index.html", ckpt=ckpt)

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if not f or f.filename == "":
            return redirect(url_for("index"))

        pil = Image.open(f.stream).convert("RGB")
        model: torch.nn.Module = app.config["MODEL"]
        device: torch.device = app.config["DEVICE"]
        names: list[str] = app.config["CLASS_NAMES"]
        size: int = app.config["IMAGE_SIZE"]

        label, conf, ranked = predict_pil(model, pil, names, size, device)

        # Optional heatmap (CPU fallback on MPS for grad-cam).
        heatmap_url = None
        try:
            pil_s = pil.resize((size, size))
            rgb = np.asarray(pil_s, dtype=np.float32) / 255.0
            x = eval_transform(size)(pil).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                pred = int(torch.argmax(model(x), dim=1).item())

            cam_dev = device
            m = model
            x_cam = x
            if device.type == "mps":
                cam_dev = torch.device("cpu")
                m = model.to(cam_dev)
                x_cam = x.to(cam_dev)

            cam = GradCAM(model=m, target_layers=[m.layer4[-1]])
            g = cam(input_tensor=x_cam, targets=[ClassifierOutputTarget(pred)])[0, :]
            if device.type == "mps":
                model.to(device)

            vis = show_cam_on_image(rgb, g, use_rgb=True)
            out = app.config["HEATMAP_PATH"]
            Image.fromarray(vis).save(out)
            # Query string stops the browser from showing the previous upload's image.
            heatmap_url = url_for("static", filename="last_cam.png", t=int(time.time() * 1000))
        except Exception:
            heatmap_url = None

        # Classical OpenCV leaf / lesion analysis, shown next to the CNN output.
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()
        analysis = analyze_leaf(bgr)
        cv_ms = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(str(app.config["CV_PATH"]), draw_analysis(bgr, analysis))

        return render_template(
            "result.html",
            label=label,
            confidence=f"{conf:.4f}",
            ranked=ranked[:5],
            heatmap_url=heatmap_url,
            cv_url=url_for("static", filename="last_cv.jpg", t=int(time.time() * 1000)),
            leaf_pct=f"{analysis.leaf_fraction:.1%}",
            lesion_pct=f"{analysis.lesion_fraction:.1%}",
            lesion_regions=len(analysis.lesion_boxes),
            cv_ms=f"{cv_ms:.0f}",
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
