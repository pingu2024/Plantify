"""
PyTorch-free inference: OpenCV DNN runs the exported ONNX model, and the classical
leaf / lesion analysis from cv_preprocess runs next to it.

    python -m src.infer_opencv --model exports/b2_imagenet_full.onnx data/testimg1_cornleaf.jpeg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.cv_preprocess import analyze_leaf, crop_to_leaf, draw_analysis, preprocess_for_model


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def ms_since(t0: int) -> float:
    return (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()


def put_label(img: np.ndarray, text: str) -> None:
    scale = max(0.6, img.shape[1] / 900)
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 20, th + base + 20), (20, 20, 20), cv2.FILLED)
    cv2.putText(img, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default="exports/b2_imagenet_full.onnx")
    p.add_argument("--crop_leaf", action="store_true", help="Classify the segmented leaf crop instead of the full image")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--out_dir", default="runs/opencv_infer")
    args = p.parse_args()

    model_path = Path(args.model)
    info = json.loads((model_path.parent / "model_info.json").read_text())
    labels = (model_path.parent / "labels.txt").read_text().splitlines()
    size = int(info["input"][-1])

    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Skipping unreadable image: {path}")
            continue

        t0 = cv2.getTickCount()
        analysis = analyze_leaf(bgr)
        t_cv = ms_since(t0)

        src = crop_to_leaf(bgr, analysis) if args.crop_leaf else bgr
        t0 = cv2.getTickCount()
        net.setInput(preprocess_for_model(src, size))
        probs = softmax(net.forward()[0])
        t_dnn = ms_since(t0)

        order = np.argsort(probs)[::-1][: args.topk]
        print(f"\n{path}")
        for i in order:
            print(f"  {labels[i]:<30s} {probs[i]:.4f}")
        print(
            f"  leaf {analysis.leaf_fraction:.1%} of image | lesion-like area {analysis.lesion_fraction:.1%} of leaf | "
            f"{len(analysis.lesion_boxes)} regions"
        )
        print(f"  timing: leaf analysis {t_cv:.1f} ms, DNN {t_dnn:.1f} ms")

        vis = draw_analysis(bgr, analysis)
        put_label(vis, f"{labels[order[0]]} ({probs[order[0]]:.1%})")
        out = out_dir / f"{Path(path).stem}_opencv.jpg"
        cv2.imwrite(str(out), vis)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
