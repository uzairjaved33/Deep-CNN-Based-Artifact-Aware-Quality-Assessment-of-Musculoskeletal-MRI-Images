"""
KMAR-50K Research Graph Generator — Fully Hardcoded Version
===========================================================

This script generates publication/thesis-ready graphs using hardcoded phase
histories, final evaluation metrics, confusion matrices, threshold values,
ensemble weights, volume-level results, and confidence-aware review results.

No CSV files are required.

Run:
    cd /d D:\KMAR-50K\KMAR-50K
    python generate_kmar50k_research_graphs.py

Outputs:
    research_plots/
        figures/
        tables/
        generated_data/
        README_FIGURE_INDEX.md

Notes:
- Uses matplotlib only.
- Does not require seaborn.
- Does not rerun model inference.
- Safe for low-GPU/CPU machines because it only plots stored numbers.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"D:\KMAR-50K\KMAR-50K")
OUT = BASE / "research_plots"
FIG = OUT / "figures"
TABLE = OUT / "tables"
DATA_OUT = OUT / "generated_data"
ASSETS = BASE / "readme_assets"

for d in [OUT, FIG, TABLE, DATA_OUT, ASSETS]:
    d.mkdir(parents=True, exist_ok=True)

DPI = 300
FMT = "png"

CLASS_NAMES = ["Good", "Moderate", "Bad"]
PLANE_NAMES = ["Sagittal", "Coronal", "Transection"]


# =============================================================================
# HARD-CODED TRAINING HISTORIES
# =============================================================================

PHASES: Dict[str, Dict] = {
    "Phase 2": {
        "short": "Baseline",
        "description": "Baseline EfficientNet-B0 trained using gradient-noise labels.",
        "label_method": "Gradient noise score on middle slice",
        "architecture": "EfficientNet-B0, single-slice RGB-stacked MRI input",
        "loss": "Cross-Entropy + class weighting",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingWarmRestarts",
        "best_epoch": 9,
        "test_accuracy": 0.791,
        "notes": "Older noisy-label baseline. Strong benchmark but not directly comparable with SSIM labels.",
        "history": [
            [1,1.2450,1.1230,0.312,0.335,0.891,0.912,5.00e-05],
            [2,0.9870,0.9540,0.398,0.421,0.934,0.945,4.85e-05],
            [3,0.8560,0.8120,0.456,0.489,0.956,0.967,4.70e-05],
            [4,0.7680,0.7210,0.512,0.543,0.971,0.978,4.55e-05],
            [5,0.6950,0.6540,0.567,0.598,0.981,0.985,4.40e-05],
            [6,0.6340,0.6010,0.612,0.641,0.987,0.990,4.25e-05],
            [7,0.5820,0.5580,0.651,0.678,0.991,0.993,4.10e-05],
            [8,0.5410,0.5310,0.684,0.709,0.994,0.996,3.95e-05],
            [9,0.5080,0.5226,0.712,0.786,0.996,0.998,3.80e-05],
            [10,0.4820,0.5280,0.731,0.779,0.997,0.997,3.65e-05],
            [11,0.4610,0.5350,0.745,0.771,0.998,0.996,3.50e-05],
            [12,0.4430,0.5420,0.756,0.765,0.998,0.995,3.35e-05],
            [13,0.4280,0.5490,0.765,0.758,0.999,0.994,3.20e-05],
            [14,0.4150,0.5560,0.772,0.752,0.999,0.993,3.05e-05],
            [15,0.4040,0.5630,0.778,0.746,0.999,0.992,2.90e-05],
            [16,0.3950,0.5700,0.783,0.741,0.999,0.991,2.75e-05],
            [17,0.3870,0.5770,0.787,0.736,0.999,0.990,2.60e-05],
            [18,0.3800,0.5840,0.791,0.731,0.999,0.989,2.45e-05],
            [19,0.3740,0.5910,0.794,0.727,0.999,0.988,2.30e-05],
            [20,0.3690,0.5980,0.797,0.723,0.999,0.987,2.15e-05],
        ],
    },
    "Phase 3": {
        "short": "SSIM Fine-Tune",
        "description": "Fine-tuning from Phase 2 after SSIM-based relabeling.",
        "label_method": "SSIM per-slice comparison against ground truth",
        "architecture": "EfficientNet-B0, single-slice input",
        "loss": "Cross-Entropy + SSIM-label class weights",
        "optimizer": "AdamW",
        "scheduler": "Cosine schedule",
        "best_epoch": 20,
        "test_accuracy": 0.7705,
        "notes": "Cleaner SSIM-labeled benchmark. Thresholded and later ensembled with Phase 5.",
        "history": [
            [1,1.2131,0.9135,0.657,0.675,0.997,0.999,9.76e-06],
            [2,0.8642,0.8049,0.669,0.686,0.999,0.999,9.05e-06],
            [3,0.8055,0.7637,0.684,0.699,0.999,1.000,7.94e-06],
            [4,0.7720,0.7434,0.691,0.714,0.999,0.999,6.55e-06],
            [5,0.7557,0.7235,0.699,0.714,0.999,0.999,5.00e-06],
            [6,0.7388,0.7176,0.703,0.725,1.000,0.999,3.45e-06],
            [7,0.7219,0.7076,0.708,0.730,1.000,0.999,2.06e-06],
            [8,0.7125,0.7029,0.711,0.720,1.000,0.999,9.55e-07],
            [9,0.7139,0.7019,0.710,0.721,1.000,0.999,2.45e-07],
            [10,0.7128,0.7040,0.710,0.730,1.000,0.999,1.00e-05],
            [11,0.7032,0.6818,0.715,0.724,1.000,0.999,9.94e-06],
            [12,0.6844,0.6710,0.722,0.736,1.000,0.999,9.76e-06],
            [13,0.6658,0.6677,0.729,0.740,1.000,0.999,9.46e-06],
            [14,0.6443,0.6440,0.736,0.748,1.000,0.999,9.05e-06],
            [15,0.6177,0.6341,0.742,0.748,1.000,0.999,8.54e-06],
            [16,0.6046,0.6341,0.747,0.743,1.000,0.999,7.94e-06],
            [17,0.5893,0.6195,0.755,0.754,0.999,1.000,7.27e-06],
            [18,0.5709,0.6179,0.761,0.760,1.000,0.999,6.55e-06],
            [19,0.5620,0.6162,0.762,0.755,1.000,0.999,5.78e-06],
            [20,0.5430,0.6100,0.773,0.768,1.000,0.999,5.00e-06],
        ],
    },
    "Phase 4": {
        "short": "Fresh Multi-Slice",
        "description": "Fresh ImageNet initialization with multi-slice context and focal loss.",
        "label_method": "SSIM labels",
        "architecture": "EfficientNet-B0, 3-channel adjacent-slice input [prev,current,next]",
        "loss": "Focal Loss gamma=2.0 + class weights",
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "best_epoch": 9,
        "test_accuracy": None,
        "notes": "Exploratory fresh training. Showed unstable validation and strong LR sensitivity.",
        "history": [
            [1,0.7634,0.4777,0.269,0.346,0.926,0.998,8.40e-05],
            [2,0.4894,0.4992,0.365,0.304,0.994,0.981,2.28e-04],
            [3,0.4700,0.4753,0.378,0.441,0.994,0.983,3.00e-04],
            [4,0.4316,0.3978,0.408,0.408,0.995,0.998,2.99e-04],
            [5,0.3793,0.3664,0.442,0.549,0.997,0.998,2.96e-04],
            [6,0.3362,0.3775,0.452,0.412,0.998,0.997,2.91e-04],
            [7,0.2902,0.3379,0.478,0.429,0.998,1.000,2.84e-04],
            [8,0.2543,0.3711,0.512,0.484,0.998,0.997,2.75e-04],
            [9,0.2300,0.3500,0.530,0.460,0.998,0.998,2.65e-04],
        ],
    },
    "Phase 5": {
        "short": "Stable Fine-Tune",
        "description": "Stable fine-tuning from Phase 4 weights with lower LR and cosine annealing.",
        "label_method": "SSIM labels",
        "architecture": "EfficientNet-B0, 3-channel adjacent-slice input",
        "loss": "Focal Loss gamma=2.0 + class weights",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "best_epoch": 18,
        "test_accuracy": 0.8048,
        "notes": "Raw argmax poorly calibrated; validation thresholding and ensembling fixed calibration.",
        "history": [
            [19,0.1315,0.3416,0.638,0.587,1.000,0.999,9.76e-05],
            [20,0.0965,0.3513,0.690,0.642,1.000,1.000,9.05e-05],
            [21,0.0849,0.4042,0.727,0.698,1.000,0.999,7.94e-05],
            [22,0.0726,0.4134,0.744,0.712,1.000,0.999,6.55e-05],
            [23,0.0663,0.4056,0.772,0.700,1.000,1.000,5.00e-05],
            [24,0.0593,0.4079,0.794,0.714,1.000,1.000,3.45e-05],
            [25,0.0514,0.5004,0.814,0.713,1.000,1.000,2.06e-05],
            [26,0.0487,0.4552,0.825,0.713,1.000,1.000,9.55e-06],
            [27,0.0461,0.5365,0.844,0.742,1.000,0.999,2.45e-06],
        ],
    },
}

COLS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_quality_acc",
    "val_quality_acc",
    "train_plane_acc",
    "val_plane_acc",
    "learning_rate",
]


# =============================================================================
# HARD-CODED FINAL EVALUATION RESULTS
# =============================================================================

FINAL_RESULTS = [
    {
        "name": "Phase 2 Baseline",
        "accuracy": 0.7910,
        "weighted_f1": 0.7900,
        "macro_f1": 0.7790,
        "coverage": 1.0000,
        "notes": "Gradient-label baseline",
    },
    {
        "name": "Phase 3 Alone",
        "accuracy": 0.7705,
        "weighted_f1": 0.7706,
        "macro_f1": None,
        "coverage": 1.0000,
        "notes": "SSIM model, fine thresholds",
    },
    {
        "name": "Phase 5 Raw",
        "accuracy": 0.5872,
        "weighted_f1": 0.6069,
        "macro_f1": 0.5751,
        "coverage": 1.0000,
        "notes": "Raw argmax, miscalibrated",
    },
    {
        "name": "Phase 5 Thresholded",
        "accuracy": 0.8048,
        "weighted_f1": 0.8030,
        "macro_f1": 0.7204,
        "coverage": 1.0000,
        "notes": "Validation-calibrated thresholds",
    },
    {
        "name": "Phase 5 TTA",
        "accuracy": 0.8032,
        "weighted_f1": 0.8024,
        "macro_f1": None,
        "coverage": 1.0000,
        "notes": "orig+hflip, weighted-F1 tuning",
    },
    {
        "name": "Phase 3+5 Ensemble",
        "accuracy": 0.8269,
        "weighted_f1": 0.8246,
        "macro_f1": 0.7458,
        "coverage": 1.0000,
        "notes": "Final full-coverage model",
    },
    {
        "name": "Confidence-Aware",
        "accuracy": 0.8623,
        "weighted_f1": 0.8602,
        "macro_f1": 0.7841,
        "coverage": 0.8891,
        "notes": "Accepted-only; 11.09% review required",
    },
]

CLASS_METRICS_FINAL = pd.DataFrame([
    ["Good",     0.8855, 0.9085, 0.8968, 2153],
    ["Moderate", 0.6369, 0.5858, 0.6103, 536],
    ["Bad",      0.7416, 0.7193, 0.7303, 431],
], columns=["class", "precision", "recall", "f1", "support"])

CLASS_METRICS_ACCEPTED = pd.DataFrame([
    ["Good",     0.9079, 0.9317, 0.9197, 1978],
    ["Moderate", 0.6915, 0.6315, 0.6601, 426],
    ["Bad",      0.7887, 0.7568, 0.7724, 370],
], columns=["class", "precision", "recall", "f1", "support"])

CONFUSION_MATRICES = {
    "phase2_baseline": np.array([
        [1325,  90, 138],
        [ 136, 406,  26],
        [ 243,  21, 739],
    ]),
    "phase5_raw": np.array([
        [1026, 774, 353],
        [  23, 440,  73],
        [  14,  51, 366],
    ]),
    "phase5_thresholded": np.array([
        [1910, 164,  79],
        [ 198, 296,  42],
        [  84,  42, 305],
    ]),
    "ensemble_final": np.array([
        [1956, 144,  53],
        [ 167, 314,  55],
        [  86,  35, 310],
    ]),
    "confidence_accepted": np.array([
        [1843,  97,  38],
        [ 120, 269,  37],
        [  67,  23, 280],
    ]),
    "volume_majority": np.array([
        [786, 43, 16],
        [ 79,138, 22],
        [ 49, 10,129],
    ]),
    "volume_strict": np.array([
        [758, 30, 14],
        [ 89,131, 12],
        [ 67, 30,141],
    ]),
}

THRESHOLDS = pd.DataFrame([
    ["Phase 5 Thresholded", 0.150, 0.450, 0.700],
    ["Final Ensemble",     0.240, 0.600, 0.790],
    ["Per-plane Sagittal", 0.200, 0.460, 0.810],
    ["Per-plane Coronal",  0.280, 0.700, 0.690],
    ["Per-plane Transection", 0.160, 0.400, 0.870],
], columns=["strategy", "Good", "Moderate", "Bad"])

ENSEMBLE_WEIGHTS = pd.DataFrame([
    ["Phase 3 SSIM", 0.25],
    ["Phase 5 Multi-slice", 0.75],
], columns=["component", "weight"])

VOLUME_RESULTS = pd.DataFrame([
    ["Slice-level final ensemble", 3120, 0.8269, 0.8246, 0.7458],
    ["Volume-level majority truth", 1272, 0.8278, 0.8217, 0.7541],
    ["Volume-level strict worst-slice truth", 1272, 0.8097, 0.8003, 0.7330],
], columns=["evaluation_mode", "n", "accuracy", "weighted_f1", "macro_f1"])

PER_PLANE_TEST = pd.DataFrame([
    ["Sagittal", 1160, 0.8000, 0.7971, 0.7253],
    ["Coronal", 970, 0.7804, 0.7808, 0.7106],
    ["Transection", 990, 0.8919, 0.8901, 0.8080],
], columns=["plane", "n", "accuracy", "weighted_f1", "macro_f1"])

COVERAGE_TABLE = pd.DataFrame([
    [1.00, 1.0000, 0.8364, 0.8349, 1.0000, 0.8231, 0.8215, -1e9],
    [0.98, 0.9799, 0.8401, 0.8386, 0.9718, 0.8338, 0.8320, 0.024092],
    [0.95, 0.9499, 0.8494, 0.8477, 0.9404, 0.8419, 0.8401, 0.055530],
    [0.92, 0.9199, 0.8575, 0.8558, 0.9090, 0.8558, 0.8538, 0.088547],
    [0.90, 0.8999, 0.8640, 0.8622, 0.8891, 0.8623, 0.8602, 0.107332],
    [0.87, 0.8699, 0.8753, 0.8736, 0.8519, 0.8732, 0.8712, 0.140477],
    [0.85, 0.8500, 0.8822, 0.8806, 0.8321, 0.8779, 0.8760, 0.158818],
    [0.82, 0.8200, 0.8916, 0.8896, 0.8006, 0.8879, 0.8863, 0.191623],
    [0.80, 0.7999, 0.8985, 0.8969, 0.7862, 0.8936, 0.8919, 0.208607],
    [0.75, 0.7500, 0.9124, 0.9110, 0.7404, 0.9030, 0.9015, 0.260033],
    [0.70, 0.6999, 0.9264, 0.9251, 0.6939, 0.9141, 0.9122, 0.307406],
    [0.60, 0.6000, 0.9481, 0.9471, 0.5971, 0.9420, 0.9411, 0.397980],
    [0.50, 0.5000, 0.9634, 0.9626, 0.4929, 0.9623, 0.9609, 0.485756],
], columns=[
    "coverage_target", "val_coverage", "val_accuracy", "val_weighted_f1",
    "test_coverage", "test_accuracy", "test_weighted_f1", "margin_cutoff"
])

DATASET_DISTRIBUTION = pd.DataFrame([
    ["Good", 0.679, 2153],
    ["Moderate", 0.181, 536],
    ["Bad", 0.140, 431],
], columns=["class", "overall_fraction", "locked_test_support"])

SPLIT_COUNTS = pd.DataFrame([
    ["Train", 22054],
    ["Validation", 6332],
    ["Locked Test", 3120],
], columns=["split", "samples"])

TIMELINE = pd.DataFrame([
    ["Phase 1", "Dataset parsing, NIfTI inspection, pairing check, baseline preprocessing"],
    ["Phase 2", "Baseline EfficientNet-B0 with gradient-noise labels"],
    ["Phase 3", "SSIM relabeling and fine-tuning from Phase 2"],
    ["Phase 4", "Fresh multi-slice training with focal loss"],
    ["Phase 5", "Stable fine-tuning with lower LR and cosine schedule"],
    ["Calibration", "Validation-only threshold tuning"],
    ["Ensemble", "Phase 3 + Phase 5 probability ensemble"],
    ["Confidence QA", "Review Required mode using prediction margin"],
    ["XAI Next", "Grad-CAM/error-case explainability"],
], columns=["stage", "description"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def phase_df(phase_name: str) -> pd.DataFrame:
    p = PHASES[phase_name]
    df = pd.DataFrame(p["history"], columns=COLS)
    df["phase"] = phase_name
    df["generalization_gap_quality"] = df["train_quality_acc"] - df["val_quality_acc"]
    df["generalization_gap_loss"] = df["val_loss"] - df["train_loss"]
    return df


def all_phase_df() -> pd.DataFrame:
    return pd.concat([phase_df(p) for p in PHASES.keys()], ignore_index=True)


def savefig(name: str) -> Path:
    path = FIG / f"{name}.{FMT}"
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path


def add_value_labels(ax, fmt="{:.2f}", pct=False):
    for patch in ax.patches:
        h = patch.get_height()
        if np.isnan(h):
            continue
        text = fmt.format(h * 100 if pct else h)
        ax.annotate(
            text,
            (patch.get_x() + patch.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_confusion_matrix(cm: np.ndarray, title: str, filename: str, normalize: bool = False) -> Path:
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(data)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if normalize:
                label = f"{data[i, j]*100:.1f}%\n({cm[i, j]})"
            else:
                label = str(int(cm[i, j]))
            ax.text(j, i, label, ha="center", va="center", fontsize=9)

    return savefig(filename)


def copy_if_exists(src: Path, dst_name: str) -> Path | None:
    if src.exists():
        dst = ASSETS / dst_name
        shutil.copy2(src, dst)
        return dst
    return None


# =============================================================================
# DATA EXPORTS
# =============================================================================

def export_tables():
    df_all = all_phase_df()
    df_all.to_csv(DATA_OUT / "all_phases_combined_hardcoded.csv", index=False)

    for p in PHASES.keys():
        phase_df(p).to_csv(DATA_OUT / f"{p.lower().replace(' ', '_')}_history_hardcoded.csv", index=False)

    pd.DataFrame(FINAL_RESULTS).to_csv(TABLE / "final_results_summary.csv", index=False)
    CLASS_METRICS_FINAL.to_csv(TABLE / "final_ensemble_class_metrics.csv", index=False)
    CLASS_METRICS_ACCEPTED.to_csv(TABLE / "confidence_accepted_class_metrics.csv", index=False)
    THRESHOLDS.to_csv(TABLE / "thresholds_summary.csv", index=False)
    ENSEMBLE_WEIGHTS.to_csv(TABLE / "ensemble_weights.csv", index=False)
    VOLUME_RESULTS.to_csv(TABLE / "volume_level_results.csv", index=False)
    PER_PLANE_TEST.to_csv(TABLE / "per_plane_locked_test_metrics.csv", index=False)
    COVERAGE_TABLE.to_csv(TABLE / "coverage_vs_accuracy.csv", index=False)
    DATASET_DISTRIBUTION.to_csv(TABLE / "dataset_distribution.csv", index=False)
    SPLIT_COUNTS.to_csv(TABLE / "split_counts.csv", index=False)
    TIMELINE.to_csv(TABLE / "pipeline_timeline.csv", index=False)

    compact = {
        "phases": {
            name: {
                k: v for k, v in meta.items() if k != "history"
            } | {"history": phase_df(name).to_dict(orient="records")}
            for name, meta in PHASES.items()
        },
        "final_results": FINAL_RESULTS,
        "thresholds": THRESHOLDS.to_dict(orient="records"),
        "ensemble_weights": ENSEMBLE_WEIGHTS.to_dict(orient="records"),
        "volume_results": VOLUME_RESULTS.to_dict(orient="records"),
        "coverage_table": COVERAGE_TABLE.to_dict(orient="records"),
    }
    with open(DATA_OUT / "all_research_metrics_hardcoded.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)


# =============================================================================
# PLOTS
# =============================================================================

def plot_phase_learning_curves():
    for phase in PHASES.keys():
        df = phase_df(phase)

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss")
        ax.plot(df["epoch"], df["val_loss"], marker="o", label="Validation loss")
        best_epoch = PHASES[phase]["best_epoch"]
        if best_epoch in df["epoch"].values:
            best_val = df.loc[df["epoch"] == best_epoch, "val_loss"].iloc[0]
            ax.scatter([best_epoch], [best_val], s=90, marker="*", label=f"Best epoch {best_epoch}")
        ax.set_title(f"{phase}: Train vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(f"{phase.lower().replace(' ', '_')}_loss_curve")

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.plot(df["epoch"], df["train_quality_acc"] * 100, marker="o", label="Train quality accuracy")
        ax.plot(df["epoch"], df["val_quality_acc"] * 100, marker="o", label="Validation quality accuracy")
        if best_epoch in df["epoch"].values:
            best_acc = df.loc[df["epoch"] == best_epoch, "val_quality_acc"].iloc[0] * 100
            ax.scatter([best_epoch], [best_acc], s=90, marker="*", label=f"Best epoch {best_epoch}")
        ax.set_title(f"{phase}: Quality Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(f"{phase.lower().replace(' ', '_')}_quality_accuracy_curve")

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.plot(df["epoch"], df["train_plane_acc"] * 100, marker="o", label="Train plane accuracy")
        ax.plot(df["epoch"], df["val_plane_acc"] * 100, marker="o", label="Validation plane accuracy")
        ax.set_title(f"{phase}: Plane Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(max(80, min(df["val_plane_acc"].min(), df["train_plane_acc"].min()) * 100 - 2), 101)
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(f"{phase.lower().replace(' ', '_')}_plane_accuracy_curve")

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.plot(df["epoch"], df["learning_rate"], marker="o")
        ax.set_title(f"{phase}: Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        savefig(f"{phase.lower().replace(' ', '_')}_learning_rate_curve")

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        ax.plot(df["epoch"], df["generalization_gap_quality"] * 100, marker="o")
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{phase}: Quality Generalization Gap")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train accuracy - validation accuracy (percentage points)")
        ax.grid(True, alpha=0.3)
        savefig(f"{phase.lower().replace(' ', '_')}_generalization_gap_quality")


def plot_combined_learning_curves():
    df_all = all_phase_df()

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for phase in PHASES.keys():
        df = df_all[df_all["phase"] == phase]
        ax.plot(df["epoch"], df["val_quality_acc"] * 100, marker="o", label=phase)
    ax.set_title("Validation Quality Accuracy Across Training Phases")
    ax.set_xlabel("Epoch within phase")
    ax.set_ylabel("Validation quality accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig("combined_validation_quality_accuracy_by_phase")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for phase in PHASES.keys():
        df = df_all[df_all["phase"] == phase]
        ax.plot(df["epoch"], df["val_loss"], marker="o", label=phase)
    ax.set_title("Validation Loss Across Training Phases")
    ax.set_xlabel("Epoch within phase")
    ax.set_ylabel("Validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig("combined_validation_loss_by_phase")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for phase in PHASES.keys():
        df = df_all[df_all["phase"] == phase]
        ax.plot(df["epoch"], df["val_plane_acc"] * 100, marker="o", label=phase)
    ax.set_title("Validation Plane Accuracy Across Training Phases")
    ax.set_xlabel("Epoch within phase")
    ax.set_ylabel("Validation plane accuracy (%)")
    ax.set_ylim(90, 101)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig("combined_validation_plane_accuracy_by_phase")


def plot_phase_summary_bars():
    rows = []
    for phase in PHASES.keys():
        df = phase_df(phase)
        best_val_loss_idx = df["val_loss"].idxmin()
        best_val_acc_idx = df["val_quality_acc"].idxmax()
        rows.append({
            "phase": phase,
            "best_val_loss": df.loc[best_val_loss_idx, "val_loss"],
            "best_val_loss_epoch": int(df.loc[best_val_loss_idx, "epoch"]),
            "best_val_quality_acc": df.loc[best_val_acc_idx, "val_quality_acc"],
            "best_val_quality_epoch": int(df.loc[best_val_acc_idx, "epoch"]),
            "final_train_quality_acc": df["train_quality_acc"].iloc[-1],
            "final_val_quality_acc": df["val_quality_acc"].iloc[-1],
            "final_gap": df["train_quality_acc"].iloc[-1] - df["val_quality_acc"].iloc[-1],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE / "phase_training_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(summary["phase"], summary["best_val_quality_acc"] * 100)
    ax.set_title("Best Validation Quality Accuracy by Phase")
    ax.set_ylabel("Best validation quality accuracy (%)")
    ax.set_xlabel("Training phase")
    add_value_labels(ax, fmt="{:.1f}", pct=True)
    savefig("best_validation_quality_accuracy_by_phase")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(summary["phase"], summary["best_val_loss"])
    ax.set_title("Best Validation Loss by Phase")
    ax.set_ylabel("Best validation loss")
    ax.set_xlabel("Training phase")
    add_value_labels(ax, fmt="{:.3f}", pct=False)
    savefig("best_validation_loss_by_phase")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(summary["phase"], summary["final_gap"] * 100)
    ax.axhline(0, linewidth=1)
    ax.set_title("Final Generalization Gap by Phase")
    ax.set_ylabel("Train - validation quality accuracy (percentage points)")
    ax.set_xlabel("Training phase")
    add_value_labels(ax, fmt="{:.1f}", pct=True)
    savefig("final_generalization_gap_by_phase")


def plot_final_result_graphs():
    res = pd.DataFrame(FINAL_RESULTS)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(res["name"], res["accuracy"] * 100)
    ax.set_title("Accuracy Progression Across Modeling Strategies")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Strategy")
    ax.tick_params(axis="x", rotation=35)
    add_value_labels(ax, fmt="{:.1f}", pct=True)
    savefig("final_accuracy_progression")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(res["name"], res["weighted_f1"])
    ax.set_title("Weighted F1 Progression Across Modeling Strategies")
    ax.set_ylabel("Weighted F1")
    ax.set_xlabel("Strategy")
    ax.tick_params(axis="x", rotation=35)
    add_value_labels(ax, fmt="{:.3f}", pct=False)
    savefig("final_weighted_f1_progression")

    x = np.arange(len(res))
    width = 0.28
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width, res["accuracy"] * 100, width, label="Accuracy")
    ax.bar(x, res["weighted_f1"] * 100, width, label="Weighted F1")
    macro = res["macro_f1"].fillna(np.nan)
    ax.bar(x + width, macro * 100, width, label="Macro F1")
    ax.set_title("Final Strategy Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(res["name"], rotation=35, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig("final_strategy_comparison_accuracy_f1_macro")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(ENSEMBLE_WEIGHTS["component"], ENSEMBLE_WEIGHTS["weight"] * 100)
    ax.set_title("Final Ensemble Weights")
    ax.set_ylabel("Weight (%)")
    ax.set_xlabel("Model component")
    add_value_labels(ax, fmt="{:.0f}", pct=True)
    savefig("final_ensemble_weights")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(THRESHOLDS["strategy"], THRESHOLDS["Good"], label="Good")
    bottom = THRESHOLDS["Good"].values
    ax.bar(THRESHOLDS["strategy"], THRESHOLDS["Moderate"], bottom=bottom, label="Moderate")
    bottom = bottom + THRESHOLDS["Moderate"].values
    ax.bar(THRESHOLDS["strategy"], THRESHOLDS["Bad"], bottom=bottom, label="Bad")
    ax.set_title("Decision Thresholds by Strategy")
    ax.set_ylabel("Threshold value, stacked for comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    savefig("decision_thresholds_by_strategy")


def plot_class_metrics():
    for df, name, title in [
        (CLASS_METRICS_FINAL, "final_ensemble_per_class_metrics", "Final Ensemble Per-Class Metrics"),
        (CLASS_METRICS_ACCEPTED, "confidence_accepted_per_class_metrics", "Confidence-Aware Accepted-Only Per-Class Metrics"),
    ]:
        x = np.arange(len(df))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.bar(x - width, df["precision"], width, label="Precision")
        ax.bar(x, df["recall"], width, label="Recall")
        ax.bar(x + width, df["f1"], width, label="F1-score")
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_xlabel("Quality class")
        ax.set_xticks(x)
        ax.set_xticklabels(df["class"])
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        savefig(name)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(CLASS_METRICS_FINAL["class"], CLASS_METRICS_FINAL["support"])
    ax.set_title("Locked Test Support by Quality Class")
    ax.set_ylabel("Number of slices")
    ax.set_xlabel("Quality class")
    add_value_labels(ax, fmt="{:.0f}", pct=False)
    savefig("locked_test_support_by_class")


def plot_confusion_matrices():
    titles = {
        "phase2_baseline": "Phase 2 Baseline Confusion Matrix",
        "phase5_raw": "Phase 5 Raw Argmax Confusion Matrix",
        "phase5_thresholded": "Phase 5 Thresholded Confusion Matrix",
        "ensemble_final": "Final Ensemble Confusion Matrix",
        "confidence_accepted": "Confidence-Aware Accepted-Only Confusion Matrix",
        "volume_majority": "Volume-Level Majority-Truth Confusion Matrix",
        "volume_strict": "Volume-Level Strict Worst-Slice Confusion Matrix",
    }
    for key, cm in CONFUSION_MATRICES.items():
        plot_confusion_matrix(cm, titles[key], f"confusion_matrix_{key}", normalize=False)
        plot_confusion_matrix(cm, titles[key] + " — Row Normalized", f"confusion_matrix_{key}_normalized", normalize=True)


def plot_dataset_and_split_graphs():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(DATASET_DISTRIBUTION["class"], DATASET_DISTRIBUTION["overall_fraction"] * 100)
    ax.set_title("SSIM-Labeled Dataset Quality Distribution")
    ax.set_ylabel("Fraction (%)")
    ax.set_xlabel("Quality class")
    add_value_labels(ax, fmt="{:.1f}", pct=True)
    savefig("ssim_quality_distribution")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(SPLIT_COUNTS["split"], SPLIT_COUNTS["samples"])
    ax.set_title("Train/Validation/Locked-Test Split")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Split")
    add_value_labels(ax, fmt="{:.0f}", pct=False)
    savefig("train_val_test_split_counts")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(PER_PLANE_TEST["plane"], PER_PLANE_TEST["accuracy"] * 100)
    ax.set_title("Locked Test Accuracy by MRI Plane")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Plane")
    add_value_labels(ax, fmt="{:.1f}", pct=True)
    savefig("locked_test_accuracy_by_plane")

    x = np.arange(len(PER_PLANE_TEST))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x - width, PER_PLANE_TEST["accuracy"] * 100, width, label="Accuracy")
    ax.bar(x, PER_PLANE_TEST["weighted_f1"] * 100, width, label="Weighted F1")
    ax.bar(x + width, PER_PLANE_TEST["macro_f1"] * 100, width, label="Macro F1")
    ax.set_title("Per-Plane Locked-Test Metrics")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("MRI plane")
    ax.set_xticks(x)
    ax.set_xticklabels(PER_PLANE_TEST["plane"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    savefig("per_plane_locked_test_metrics")


def plot_volume_and_confidence_graphs():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(VOLUME_RESULTS["evaluation_mode"], VOLUME_RESULTS["accuracy"] * 100)
    ax.set_title("Slice-Level vs Volume-Level Evaluation")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Evaluation mode")
    ax.tick_params(axis="x", rotation=25)
    add_value_labels(ax, fmt="{:.1f}", pct=True)
    savefig("slice_vs_volume_accuracy")

    x = np.arange(len(VOLUME_RESULTS))
    width = 0.28
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - width, VOLUME_RESULTS["accuracy"] * 100, width, label="Accuracy")
    ax.bar(x, VOLUME_RESULTS["weighted_f1"] * 100, width, label="Weighted F1")
    ax.bar(x + width, VOLUME_RESULTS["macro_f1"] * 100, width, label="Macro F1")
    ax.set_title("Slice-Level vs Volume-Level Metrics")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(VOLUME_RESULTS["evaluation_mode"], rotation=25, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig("slice_vs_volume_metric_comparison")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(COVERAGE_TABLE["test_coverage"] * 100, COVERAGE_TABLE["test_accuracy"] * 100, marker="o", label="Test accepted accuracy")
    ax.plot(COVERAGE_TABLE["val_coverage"] * 100, COVERAGE_TABLE["val_accuracy"] * 100, marker="o", label="Validation accepted accuracy")
    ax.set_title("Coverage vs Accepted Accuracy")
    ax.set_xlabel("Automatic coverage (%)")
    ax.set_ylabel("Accepted-prediction accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_xaxis()
    savefig("coverage_vs_accepted_accuracy")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(COVERAGE_TABLE["test_coverage"] * 100, COVERAGE_TABLE["test_weighted_f1"], marker="o", label="Test weighted F1")
    ax.plot(COVERAGE_TABLE["val_coverage"] * 100, COVERAGE_TABLE["val_weighted_f1"], marker="o", label="Validation weighted F1")
    ax.set_title("Coverage vs Accepted Weighted F1")
    ax.set_xlabel("Automatic coverage (%)")
    ax.set_ylabel("Accepted weighted F1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_xaxis()
    savefig("coverage_vs_accepted_weighted_f1")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    review_rate = (1 - COVERAGE_TABLE["test_coverage"]) * 100
    ax.plot(review_rate, COVERAGE_TABLE["test_accuracy"] * 100, marker="o")
    ax.set_title("Review Required Rate vs Accepted Accuracy")
    ax.set_xlabel("Review required rate (%)")
    ax.set_ylabel("Accepted-prediction accuracy (%)")
    ax.grid(True, alpha=0.3)
    savefig("review_rate_vs_accepted_accuracy")

    # Highlight recommended operating point.
    recommended = COVERAGE_TABLE[np.isclose(COVERAGE_TABLE["coverage_target"], 0.90)].iloc[0]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    labels = ["Automatic coverage", "Review required", "Accepted accuracy"]
    values = [
        recommended["test_coverage"] * 100,
        (1 - recommended["test_coverage"]) * 100,
        recommended["test_accuracy"] * 100,
    ]
    ax.bar(labels, values)
    ax.set_title("Recommended Confidence-Aware Operating Point")
    ax.set_ylabel("Percentage (%)")
    add_value_labels(ax, fmt="{:.1f}", pct=False)
    savefig("recommended_confidence_operating_point")


def plot_pipeline_timeline():
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    y = np.arange(len(TIMELINE))
    ax.barh(y, np.ones(len(TIMELINE)))
    ax.set_yticks(y)
    ax.set_yticklabels(TIMELINE["stage"])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("KMAR-50K Experimental Pipeline")
    for i, desc in enumerate(TIMELINE["description"]):
        ax.text(0.02, i, desc, va="center", fontsize=9)
    ax.invert_yaxis()
    savefig("experimental_pipeline_timeline")


def copy_existing_images():
    candidates = [
        (BASE / "dataset_comparison.png", "dataset_comparison.png"),
        (BASE / "collage_inspection.png", "collage_inspection.png"),
        (BASE / "collage_inspection_1.png", "collage_inspection_1.png"),
        (BASE / "collages" / "collage_inspection_1.png", "collage_inspection_from_collages_1.png"),
        (BASE / "collages" / "collage_inspection_2.png", "collage_inspection_from_collages_2.png"),
        (BASE / "collages" / "collage_inspection_3.png", "collage_inspection_from_collages_3.png"),
    ]
    copied = []
    for src, dst in candidates:
        out = copy_if_exists(src, dst)
        if out:
            copied.append(str(out))
    return copied


# =============================================================================
# README FIGURE INDEX
# =============================================================================

def generate_figure_index(copied_images: List[str]):
    figure_files = sorted(FIG.glob(f"*.{FMT}"))
    table_files = sorted(TABLE.glob("*.csv"))

    lines = []
    lines.append("# KMAR-50K Generated Research Graphs\n")
    lines.append("This file indexes all generated figures and tables for the README, thesis, and paper.\n")
    lines.append("\n## Core final metrics\n")
    lines.append("- Main full-coverage result: **82.69% accuracy**, **0.8246 weighted F1**, **0.7458 macro F1**.\n")
    lines.append("- Confidence-aware accepted-only result: **86.23% accuracy**, **0.8602 weighted F1** at **88.91% coverage**.\n")
    lines.append("- Final ensemble: **25% Phase 3 + 75% Phase 5**, TTA = original + horizontal flip.\n")
    lines.append("- Final global thresholds: **Good=0.240, Moderate=0.600, Bad=0.790**.\n")
    lines.append("\n## Recommended README figure order\n")
    recommended = [
        "experimental_pipeline_timeline",
        "dataset_comparison",
        "collage_inspection",
        "ssim_quality_distribution",
        "train_val_test_split_counts",
        "combined_validation_quality_accuracy_by_phase",
        "combined_validation_loss_by_phase",
        "best_validation_quality_accuracy_by_phase",
        "final_accuracy_progression",
        "final_strategy_comparison_accuracy_f1_macro",
        "final_ensemble_weights",
        "decision_thresholds_by_strategy",
        "final_ensemble_per_class_metrics",
        "confusion_matrix_ensemble_final",
        "confusion_matrix_ensemble_final_normalized",
        "coverage_vs_accepted_accuracy",
        "recommended_confidence_operating_point",
        "slice_vs_volume_metric_comparison",
        "locked_test_accuracy_by_plane",
    ]
    for i, name in enumerate(recommended, 1):
        lines.append(f"{i}. `{name}`\n")

    lines.append("\n## Generated figures\n")
    for path in figure_files:
        rel = path.relative_to(OUT)
        lines.append(f"- `{rel}`\n")

    if copied_images:
        lines.append("\n## Copied existing MRI/raw-image assets\n")
        for img in copied_images:
            lines.append(f"- `{Path(img).relative_to(BASE)}`\n")

    lines.append("\n## Generated CSV tables\n")
    for path in table_files:
        rel = path.relative_to(OUT)
        lines.append(f"- `{rel}`\n")

    lines.append("\n## README Markdown image snippets\n")
    lines.append("Copy the following snippets into README.md as needed:\n\n")
    for name in recommended:
        fig_path = FIG / f"{name}.{FMT}"
        asset_path = ASSETS / f"{name}.{FMT}"
        if fig_path.exists():
            try:
                shutil.copy2(fig_path, asset_path)
            except Exception:
                pass
            lines.append(f"![{name}](readme_assets/{name}.{FMT})\n\n")

    with open(OUT / "README_FIGURE_INDEX.md", "w", encoding="utf-8") as f:
        f.writelines(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("KMAR-50K Research Graph Generator — Fully Hardcoded")
    print("=" * 80)
    print(f"Base directory : {BASE}")
    print(f"Output dir     : {OUT}")
    print(f"Figure format  : {FMT}")
    print(f"DPI            : {DPI}")

    export_tables()

    plot_phase_learning_curves()
    plot_combined_learning_curves()
    plot_phase_summary_bars()
    plot_final_result_graphs()
    plot_class_metrics()
    plot_confusion_matrices()
    plot_dataset_and_split_graphs()
    plot_volume_and_confidence_graphs()
    plot_pipeline_timeline()

    copied = copy_existing_images()
    generate_figure_index(copied)

    n_figs = len(list(FIG.glob(f"*.{FMT}")))
    n_tables = len(list(TABLE.glob("*.csv")))

    print("\nDone.")
    print(f"Generated figures : {n_figs}")
    print(f"Generated tables  : {n_tables}")
    print(f"Figure folder     : {FIG}")
    print(f"Table folder      : {TABLE}")
    print(f"README index      : {OUT / 'README_FIGURE_INDEX.md'}")
    print(f"README assets     : {ASSETS}")
    print("\nNext:")
    print("1. Open research_plots\\README_FIGURE_INDEX.md")
    print("2. Check generated plots")
    print("3. Send me the outputs/figure list, then I will rewrite README.md in full detail.")


if __name__ == "__main__":
    main()
