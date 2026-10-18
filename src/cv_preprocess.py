"""
Classical OpenCV leaf analysis: leaf segmentation, lesion masking and bounding boxes.

This runs alongside the CNN (it does not replace it).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# All morphology runs at this working resolution so kernel sizes mean the same thing
# for 200px crops and 4000px phone photos.
WORK_SIZE = 512

# Leaf = pixels that are green in LAB (a* below neutral 128). The cut-off comes from Otsu,
# clamped to a sane range: soil / dry grass sit at a* >= 128, and Otsu alone would split
# images where the leaf fills the whole frame.
LEAF_A_MIN_THR, LEAF_A_MAX_THR = 120, 126
LEAF_V_MIN = 40           # HSV value; drops deep shadow
MAX_HOLE_FRACTION = 0.05  # holes bigger than this (vs. leaf area) are background, not lesions

# Lesion rules, all relative to the leaf's own median colour so they hold up across
# exposure / white-balance differences between field photos.
LESION_H_SHIFT = 10      # HSV hue this far below the median = yellower / browner tissue
LESION_S_MIN = 40
LESION_A_DELTA = 12      # LAB a*: less green than the median leaf pixel ...
LESION_L_BRIGHT = 20     # ... but not much brighter (that's the midrib or glare)
LESION_MIN_AREA = 30     # px at WORK_SIZE; smaller blobs are treated as noise
# A single "lesion" covering a fifth of the leaf is almost always background or shadow
# showing between leaves. Darkness alone is not used as a cue for the same reason.
LESION_MAX_FRACTION = 0.20

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class LeafAnalysis:
    leaf_mask: np.ndarray                 # uint8 {0,255}, original image size
    lesion_mask: np.ndarray               # uint8 {0,255}, original image size
    leaf_box: tuple[int, int, int, int] | None   # x, y, w, h
    lesion_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    leaf_fraction: float = 0.0            # leaf pixels / image pixels
    lesion_fraction: float = 0.0          # lesion pixels / leaf pixels


def _resize_long_side(bgr: np.ndarray, target: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    scale = target / max(h, w)
    if scale >= 1.0:
        return bgr, 1.0
    small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def segment_leaf(bgr_small: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)[:, :, 1]
    v = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)[:, :, 2]

    otsu, _ = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu, LEAF_A_MIN_THR, LEAF_A_MAX_THR))
    mask = ((a <= thr) & (v >= LEAF_V_MIN)).astype(np.uint8) * 255

    # Open first to cut thin bridges to background plants, then smooth the outline.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill interior holes (lesions are holes in a pure-green mask), but leave large
    # enclosed regions alone since those are usually background seen between leaves.
    contours, hierarchy = cv2.findContours(leaf, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    leaf_area = float(stats[largest, cv2.CC_STAT_AREA])
    if hierarchy is not None:
        for c, h in zip(contours, hierarchy[0]):
            is_hole = h[3] >= 0
            if is_hole and cv2.contourArea(c) < MAX_HOLE_FRACTION * leaf_area:
                cv2.drawContours(leaf, [c], -1, 255, thickness=cv2.FILLED)
    return leaf


def segment_lesions(bgr_small: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(leaf) == 0:
        return np.zeros_like(leaf)

    # Ignore the leaf border, where background colour bleeds in.
    inner = cv2.erode(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    H, S, _ = cv2.split(hsv)
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    L, a, _ = cv2.split(lab)

    if cv2.countNonZero(inner) == 0:
        return np.zeros_like(leaf)
    a_med = float(np.median(a[inner > 0]))
    L_med = float(np.median(L[inner > 0]))
    H_med = float(np.median(H[inner > 0]))
    L16, a16, H16 = L.astype(np.int16), a.astype(np.int16), H.astype(np.int16)

    brown = (H16 < H_med - LESION_H_SHIFT) & (S >= LESION_S_MIN)
    pale = (a16 > a_med + LESION_A_DELTA) & (L16 <= L_med + LESION_L_BRIGHT)

    lesions = ((brown | pale) & (inner > 0)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesions = cv2.morphologyEx(lesions, cv2.MORPH_OPEN, k)
    lesions = cv2.morphologyEx(lesions, cv2.MORPH_CLOSE, k)

    max_area = LESION_MAX_FRACTION * cv2.countNonZero(leaf)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lesions, connectivity=8)
    keep = np.zeros_like(lesions)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if LESION_MIN_AREA <= area <= max_area:
            keep[labels == i] = 255
    return keep


def _boxes_from_mask(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def analyze_leaf(bgr: np.ndarray) -> LeafAnalysis:
    small, scale = _resize_long_side(bgr, WORK_SIZE)
    leaf = segment_leaf(small)
    lesions = segment_lesions(small, leaf)

    leaf_px = cv2.countNonZero(leaf)
    lesion_px = cv2.countNonZero(lesions)

    h, w = bgr.shape[:2]
    inv = 1.0 / scale

    def up(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (int(b[0] * inv), int(b[1] * inv), int(b[2] * inv), int(b[3] * inv))

    leaf_boxes = _boxes_from_mask(leaf, min_area=1)
    lesion_boxes = _boxes_from_mask(lesions, min_area=LESION_MIN_AREA)

    return LeafAnalysis(
        leaf_mask=cv2.resize(leaf, (w, h), interpolation=cv2.INTER_NEAREST),
        lesion_mask=cv2.resize(lesions, (w, h), interpolation=cv2.INTER_NEAREST),
        leaf_box=up(leaf_boxes[0]) if leaf_boxes else None,
        lesion_boxes=[up(b) for b in lesion_boxes],
        leaf_fraction=leaf_px / float(small.shape[0] * small.shape[1]),
        lesion_fraction=lesion_px / float(max(1, leaf_px)),
    )


def crop_to_leaf(bgr: np.ndarray, analysis: LeafAnalysis, pad: float = 0.05, min_leaf_fraction: float = 0.05) -> np.ndarray:
    """Crop to the leaf bounding box. Falls back to the full image if segmentation looks unreliable."""
    if analysis.leaf_box is None or analysis.leaf_fraction < min_leaf_fraction:
        return bgr
    x, y, w, h = analysis.leaf_box
    H, W = bgr.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return bgr[y0:y1, x0:x1]


def draw_analysis(bgr: np.ndarray, analysis: LeafAnalysis, max_boxes: int = 25) -> np.ndarray:
    out = bgr.copy()
    thick = max(2, round(max(out.shape[:2]) / 300))

    red = np.zeros_like(out)
    red[:] = (40, 40, 230)
    lesion = analysis.lesion_mask > 0
    out[lesion] = cv2.addWeighted(out, 0.45, red, 0.55, 0)[lesion]

    contours, _ = cv2.findContours(analysis.leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (80, 220, 80), thick)

    for x, y, w, h in analysis.lesion_boxes[:max_boxes]:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 215, 255), thick)

    if analysis.leaf_box is not None:
        x, y, w, h = analysis.leaf_box
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 160, 40), thick)
    return out


def preprocess_for_model(bgr: np.ndarray, image_size: int) -> np.ndarray:
    """OpenCV equivalent of the eval transform: BGR uint8 -> NCHW float32 normalised blob."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return cv2.dnn.blobFromImage(x)  # HWC -> 1x3xHxW, no extra scaling


class CvLeafCrop:
    """torchvision-compatible transform (PIL -> PIL) that crops to the segmented leaf."""

    def __init__(self, pad: float = 0.05) -> None:
        self.pad = pad

    def __call__(self, img):
        from PIL import Image

        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        crop = crop_to_leaf(bgr, analyze_leaf(bgr), pad=self.pad)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _panel(images: list[np.ndarray], titles: list[str], tile: int = 360) -> np.ndarray:
    tiles = []
    for im, title in zip(images, titles):
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        s = tile / max(h, w)
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        canvas = np.full((tile + 30, tile, 3), 24, dtype=np.uint8)
        y0, x0 = 30 + (tile - im.shape[0]) // 2, (tile - im.shape[1]) // 2
        canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(canvas)
    return np.hstack(tiles)


def main() -> None:
    p = argparse.ArgumentParser(description="OpenCV leaf / lesion analysis for one or more images")
    p.add_argument("images", nargs="+", help="Image paths")
    p.add_argument("--out_dir", default="runs/cv_analysis")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Skipping unreadable image: {path}")
            continue

        t0 = cv2.getTickCount()
        res = analyze_leaf(bgr)
        ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()

        panel = _panel(
            [bgr, res.leaf_mask, res.lesion_mask, draw_analysis(bgr, res), crop_to_leaf(bgr, res)],
            ["input", "leaf mask (LAB a*, Otsu)", "lesion mask (HSV/LAB)", "overlay", "leaf crop"],
        )
        out = out_dir / f"{Path(path).stem}_analysis.jpg"
        cv2.imwrite(str(out), panel)
        print(
            f"{Path(path).name}: leaf {res.leaf_fraction:.1%} of image | "
            f"lesions {res.lesion_fraction:.1%} of leaf | {len(res.lesion_boxes)} regions | "
            f"{ms:.1f} ms -> {out}"
        )


if __name__ == "__main__":
    main()
