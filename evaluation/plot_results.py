"""Create publication-friendly plots from generated evaluation JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_json(path: Path):
    return json.loads(path.read_text())


def save_benchmark_plot(benchmark: dict, out_dir: Path):
    models = [name for name in ("original", "retrained", "unlearned_combined") if name in benchmark]
    labels = {"original": "Original", "retrained": "Retrained", "unlearned_combined": "Unlearned"}
    x = np.arange(3)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(models):
        values = [benchmark[name][split]["accuracy"] for split in ("retain", "forget", "test")]
        ax.bar(x[:len(values)] + (i - (len(models) - 1) / 2) * width, values, width, label=labels[name])
    ax.set_xticks(x)
    ax.set_xticklabels(["Retain", "Forget", "Test"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("MNIST Unlearning Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=180)
    plt.close(fig)


def save_mia_plot(mia: dict, out_dir: Path):
    models = [name for name in ("original", "retrained", "unlearned_combined") if name in mia]
    values = [mia[name]["auc"] for name in models]
    labels = {"original": "Original", "retrained": "Retrained", "unlearned_combined": "Unlearned"}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([labels[m] for m in models], values)
    ax.axhline(0.5, linestyle="--", label="Random (0.50)")
    ax.set_ylim(0.45, 0.55)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Confidence-Based Membership Inference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mia_auc_comparison.png", dpi=180)
    plt.close(fig)


def save_similarity_plot(similarity: dict, out_dir: Path):
    splits = [s for s in ("retain", "forget", "test") if s in similarity]
    values = [
        similarity[s]["unlearned_vs_retrained"]["prediction_agreement"]
        for s in splits
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(splits, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Prediction agreement")
    ax.set_title("Unlearned vs Retrained Oracle")
    fig.tight_layout()
    fig.savefig(out_dir / "oracle_prediction_agreement.png", dpi=180)
    plt.close(fig)


def main():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    result_dir = Path(cfg["paths"]["results"])
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = result_dir / "benchmark.json"
    mia_path = result_dir / "mia.json"
    similarity_path = result_dir / "unlearning_similarity.json"

    if benchmark_path.exists():
        save_benchmark_plot(load_json(benchmark_path), plot_dir)
    if mia_path.exists():
        save_mia_plot(load_json(mia_path), plot_dir)
    if similarity_path.exists():
        save_similarity_plot(load_json(similarity_path), plot_dir)

    print(f"Plots written to {plot_dir}")


if __name__ == "__main__":
    main()
