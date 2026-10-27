"""
Chapter 4 figures — training curves, robustness charts, confusion matrices.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(history, save_path: str | Path) -> None:
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, hist["loss"], label="train loss")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(epochs, hist["accuracy"], label="train acc")
    if "val_accuracy" in hist:
        axes[1].plot(epochs, hist["val_accuracy"], label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_robustness_comparison(results: dict, save_path: str | Path) -> None:
    """
    results: {model_name: eval_dict from evaluate_with_degradations}
    Bar chart of clean vs mean-degraded accuracy + robustness score annotation.
    """
    labels: list[str] = []
    clean_vals: list[float] = []
    deg_vals: list[float] = []
    for name, ev in results.items():
        labels.append(name)
        clean_vals.append(ev["clean"]["accuracy"])
        deg_vals.append(ev["mean_degraded_accuracy"])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, clean_vals, w, label="Clean accuracy")
    ax.bar(x + w/2, deg_vals, w, label="Mean degraded accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.set_title("Robustness comparison — clean vs. degraded")
    for i, ev in enumerate(results.values()):
        ax.text(i, 0.02, f"R={ev['robustness_score']:.2f}", ha="center", fontsize=8, color="white" if deg_vals[i] > 0.5 else "black")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_per_condition_heatmap(eval_dict: dict, save_path: str | Path, title: str = "Per-condition accuracy") -> None:
    per = eval_dict.get("per_condition", {})
    if not per:
        return
    # Group by degradation type
    from collections import defaultdict
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for k, v in per.items():
        deg = k.split("/")[0]
        grouped[deg].append((k, v["accuracy"]))
    # Build matrix
    deg_names = sorted(grouped.keys())
    # Determine max severities per deg
    max_sev = max(len(v) for v in grouped.values())
    mat = np.full((len(deg_names), max_sev), np.nan)
    for i, dn in enumerate(deg_names):
        for j, (_, acc) in enumerate(sorted(grouped[dn])):
            mat[i, j] = acc
    fig, ax = plt.subplots(figsize=(8, max(3, len(deg_names) * 0.7)))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax,
                xticklabels=[f"S{j+1}" for j in range(max_sev)],
                yticklabels=deg_names)
    ax.set_title(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], save_path: str | Path, title: str = "Confusion matrix") -> None:
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(arr, cmap="Blues", ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
