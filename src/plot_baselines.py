from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils import resolve_path


def _ensure_out_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_accuracy_and_error(
    labels: list[str],
    full_acc: list[float],
    few_acc: list[float],
    title: str,
    out_png: Path,
) -> None:
    x = np.arange(len(labels))
    w = 0.38

    full = np.array(full_acc) * 100.0
    few = np.array(few_acc) * 100.0
    full_err = 100.0 - full
    few_err = 100.0 - few

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.bar(x - w / 2, full, width=w, label="Full data", color="#0b57d0")
    ax1.bar(x + w / 2, few, width=w, label="10-shot", color="#22c55e")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()

    ax2.bar(x - w / 2, full_err, width=w, label="Full data", color="#94a3b8")
    ax2.bar(x + w / 2, few_err, width=w, label="10-shot", color="#f59e0b")
    ax2.set_ylabel("Error rate (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.25)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="results/results.yaml")
    p.add_argument("--out_dir", type=str, default="runs/plots")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_path = Path(resolve_path(repo_root, args.results))
    out_dir = _ensure_out_dir(Path(resolve_path(repo_root, args.out_dir)))

    blob = yaml.safe_load(results_path.read_text())
    acc_full = blob["metrics"]["accuracy"]["full"]
    acc_few = blob["metrics"]["accuracy"]["fewshot_10"]
    labels_map = blob.get("labels", {})

    # Keep consistent ordering (matches README table).
    keys = ["b1_scratch", "b2_imagenet", "b3_pv_supervised", "b4_ssl_ours"]
    labels = [labels_map.get(k, k) for k in keys]
    full = [float(acc_full[k]) for k in keys]
    few = [float(acc_few[k]) for k in keys]

    title = f"{blob.get('task', 'Baseline comparison')} — test={blob.get('test_images', 'NA')} images"
    plot_accuracy_and_error(labels, full, few, title, out_png=out_dir / "baseline_comparison.png")

    print(f"Saved: {out_dir / 'baseline_comparison.png'}")


if __name__ == "__main__":
    main()

