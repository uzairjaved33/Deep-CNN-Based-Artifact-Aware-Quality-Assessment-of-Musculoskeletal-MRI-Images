"""
KMAR-50K Binary Artifact vs Non-Artifact Evaluator
==================================================

Purpose
-------
Creates separate binary results and confusion matrices for client requirement:

    Non-artifact = clean / ground-truth / Good
    Artifact     = artifact / Bad
    Moderate     = skipped from strict binary SSIM evaluation

This script does NOT retrain. It reuses the final frozen ensemble probabilities:
    final_probs = 0.25 * Phase3 + 0.75 * Phase5

It creates two binary reports:

1) SSIM strict binary, skip Moderate:
   - true Good  -> Non-artifact
   - true Bad   -> Artifact
   - true Moderate excluded
   - model prediction forced between Good-vs-Bad using final adjusted scores

2) Path-origin binary:
   - path containing GroundTruthData_part1 -> Non-artifact
   - path containing ArtifactData_part1    -> Artifact
   - uses all samples that can be mapped by path
   - model prediction forced between Good-vs-Bad using final adjusted scores

Outputs:
    checkpoints/binary_artifact_nonartifact/
    ├── binary_ssim_skip_moderate_report.txt
    ├── binary_ssim_skip_moderate_confusion_matrix.png
    ├── binary_path_origin_report.txt
    ├── binary_path_origin_confusion_matrix.png
    ├── binary_all_predictions.csv
    └── binary_summary.json

Run:
    cd /d D:\KMAR-50K\KMAR-50K
    python evaluate_binary_artifact_nonartifact.py

If prediction_cache is missing:
    python build_xai_prediction_cache_only.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(os.environ.get("KMAR_BASE", r"D:\KMAR-50K\KMAR-50K"))
CKPT_DIR = BASE / "checkpoints"
PRED_CACHE_DIR = CKPT_DIR / "prediction_cache"
OUT_DIR = CKPT_DIR / "binary_artifact_nonartifact"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_JSON = CKPT_DIR / "test_samples_ssim.json"
STRATEGY_JSON = CKPT_DIR / "best_strategy_tta_ensemble.json"

CLASS_NAMES = ["Good", "Moderate", "Bad"]
BIN_NAMES = ["Non-artifact", "Artifact"]


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest(pattern: str) -> Path:
    files = glob.glob(str(PRED_CACHE_DIR / pattern))
    if not files:
        raise FileNotFoundError(
            f"No cache found for pattern {pattern} in {PRED_CACHE_DIR}\n"
            "Run: python build_xai_prediction_cache_only.py"
        )
    return Path(sorted(files, key=os.path.getmtime, reverse=True)[0])


def compute_final_scores() -> Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = load_json(TEST_JSON)
    strategy = load_json(STRATEGY_JSON)

    p3_path = find_latest("phase3_test_best_model_ssim_orig_*.npz")
    p5_path = find_latest("phase5_test_best_model_phase5_orig_hflip_*.npz")

    print(f"Using Phase 3 cache: {p3_path}")
    print(f"Using Phase 5 cache: {p5_path}")

    p3 = np.load(str(p3_path))
    p5 = np.load(str(p5_path))

    p3_probs = np.asarray(p3["probs"], dtype=np.float32)
    p5_probs = np.asarray(p5["probs"], dtype=np.float32)

    if len(samples) != len(p3_probs) or len(samples) != len(p5_probs):
        raise RuntimeError(
            f"Length mismatch: samples={len(samples)}, p3={len(p3_probs)}, p5={len(p5_probs)}"
        )

    q_true = np.asarray(p5["q_true"] if "q_true" in p5.files else [s["quality"] for s in samples], dtype=np.int64)

    w3 = float(strategy.get("phase3_weight", 0.25))
    w5 = float(strategy.get("phase5_weight", 0.75))
    thresholds_dict = strategy.get("thresholds", {"Good": 0.240, "Moderate": 0.600, "Bad": 0.790})
    thresholds = np.array(
        [
            float(thresholds_dict.get("Good", 0.240)),
            float(thresholds_dict.get("Moderate", 0.600)),
            float(thresholds_dict.get("Bad", 0.790)),
        ],
        dtype=np.float32,
    ).reshape(1, 3)

    probs = (w3 * p3_probs) + (w5 * p5_probs)
    adjusted = probs / thresholds

    pred3 = np.argmax(adjusted, axis=1).astype(np.int64)

    nonartifact_score = adjusted[:, 0]
    artifact_score = adjusted[:, 2]
    pred_bin = (artifact_score > nonartifact_score).astype(np.int64)

    denom = nonartifact_score + artifact_score + 1e-12
    prob_nonartifact = nonartifact_score / denom
    prob_artifact = artifact_score / denom
    binary_confidence = np.maximum(prob_nonartifact, prob_artifact)
    binary_margin = np.abs(prob_artifact - prob_nonartifact)

    return samples, q_true, pred3, pred_bin, np.stack([prob_nonartifact, prob_artifact, binary_confidence, binary_margin], axis=1)


def path_origin_label(path: str) -> Optional[int]:
    p = str(path).replace("\\", "/").lower()
    if "groundtruthdata_part1" in p or "testing _groundtruthdata" in p or "testing_groundtruthdata" in p:
        return 0
    if "artifactdata_part1" in p:
        return 1
    return None


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    n = tn + fp + fn + tp

    def div(a, b):
        return float(a) / float(b) if b else 0.0

    acc = div(tp + tn, n)

    prec_non = div(tn, tn + fn)
    rec_non = div(tn, tn + fp)
    f1_non = div(2 * prec_non * rec_non, prec_non + rec_non)

    prec_art = div(tp, tp + fp)
    rec_art = div(tp, tp + fn)
    f1_art = div(2 * prec_art * rec_art, prec_art + rec_art)

    support_non = tn + fp
    support_art = fn + tp

    macro_f1 = (f1_non + f1_art) / 2.0
    weighted_f1 = div((f1_non * support_non) + (f1_art * support_art), n)

    return {
        "n": n,
        "accuracy": acc,
        "nonartifact_precision": prec_non,
        "nonartifact_recall": rec_non,
        "nonartifact_f1": f1_non,
        "nonartifact_support": support_non,
        "artifact_precision": prec_art,
        "artifact_recall": rec_art,
        "artifact_f1": f1_art,
        "artifact_support": support_art,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def make_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion(cm: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(BIN_NAMES, rotation=15, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(BIN_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    max_val = cm.max() if cm.size else 1
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            color = "white" if val > max_val * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(name: str, cm: np.ndarray, metrics: Dict[str, float], out_path: Path, note: str):
    lines = []
    lines.append("=" * 78 + "\n")
    lines.append(f"KMAR-50K Binary Artifact vs Non-Artifact Report - {name}\n")
    lines.append("=" * 78 + "\n\n")
    lines.append(note.strip() + "\n\n")
    lines.append("Label mapping:\n")
    lines.append("  0 = Non-artifact\n")
    lines.append("  1 = Artifact\n\n")
    lines.append("Confusion matrix rows=true, cols=pred [Non-artifact, Artifact]:\n")
    lines.append(str(cm.tolist()) + "\n\n")
    lines.append(f"Samples       : {metrics['n']}\n")
    lines.append(f"Accuracy      : {metrics['accuracy']*100:.2f}%\n")
    lines.append(f"Macro F1      : {metrics['macro_f1']:.4f}\n")
    lines.append(f"Weighted F1   : {metrics['weighted_f1']:.4f}\n\n")
    lines.append("Per-class metrics:\n")
    lines.append(f"  Non-artifact precision: {metrics['nonartifact_precision']:.4f}\n")
    lines.append(f"  Non-artifact recall   : {metrics['nonartifact_recall']:.4f}\n")
    lines.append(f"  Non-artifact F1       : {metrics['nonartifact_f1']:.4f}\n")
    lines.append(f"  Non-artifact support  : {metrics['nonartifact_support']}\n\n")
    lines.append(f"  Artifact precision    : {metrics['artifact_precision']:.4f}\n")
    lines.append(f"  Artifact recall       : {metrics['artifact_recall']:.4f}\n")
    lines.append(f"  Artifact F1           : {metrics['artifact_f1']:.4f}\n")
    lines.append(f"  Artifact support      : {metrics['artifact_support']}\n\n")
    lines.append("Binary error interpretation:\n")
    lines.append(f"  True non-artifact predicted artifact: {metrics['fp']}\n")
    lines.append(f"  True artifact predicted non-artifact: {metrics['fn']}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def save_predictions_csv(samples, q_true, pred3, pred_bin, binary_stats):
    out = OUT_DIR / "binary_all_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "path",
            "ssim_true_quality",
            "ssim_true_quality_name",
            "final_3class_pred",
            "final_3class_pred_name",
            "path_origin_binary_true",
            "path_origin_binary_true_name",
            "binary_pred",
            "binary_pred_name",
            "binary_prob_nonartifact",
            "binary_prob_artifact",
            "binary_confidence",
            "binary_margin",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, s in enumerate(samples):
            path = s.get("path", "")
            origin = path_origin_label(path)
            w.writerow({
                "index": i,
                "path": path,
                "ssim_true_quality": int(q_true[i]),
                "ssim_true_quality_name": CLASS_NAMES[int(q_true[i])] if 0 <= int(q_true[i]) <= 2 else "",
                "final_3class_pred": int(pred3[i]),
                "final_3class_pred_name": CLASS_NAMES[int(pred3[i])],
                "path_origin_binary_true": "" if origin is None else int(origin),
                "path_origin_binary_true_name": "" if origin is None else BIN_NAMES[int(origin)],
                "binary_pred": int(pred_bin[i]),
                "binary_pred_name": BIN_NAMES[int(pred_bin[i])],
                "binary_prob_nonartifact": float(binary_stats[i, 0]),
                "binary_prob_artifact": float(binary_stats[i, 1]),
                "binary_confidence": float(binary_stats[i, 2]),
                "binary_margin": float(binary_stats[i, 3]),
            })

    print(f"Saved predictions CSV: {out}")


def main():
    print("=" * 78)
    print("KMAR-50K Binary Artifact vs Non-Artifact Evaluator")
    print("=" * 78)
    print(f"BASE    : {BASE}")
    print(f"OUT_DIR : {OUT_DIR}")

    samples, q_true, pred3, pred_bin, binary_stats = compute_final_scores()
    save_predictions_csv(samples, q_true, pred3, pred_bin, binary_stats)

    summary = {}

    mask_ssim = np.isin(q_true, [0, 2])
    y_true_ssim = np.where(q_true[mask_ssim] == 2, 1, 0).astype(np.int64)
    y_pred_ssim = pred_bin[mask_ssim].astype(np.int64)

    cm_ssim = make_confusion(y_true_ssim, y_pred_ssim)
    metrics_ssim = metrics_from_confusion(cm_ssim)

    plot_confusion(
        cm_ssim,
        "Binary Artifact vs Non-Artifact\nSSIM strict evaluation - true Moderate skipped",
        OUT_DIR / "binary_ssim_skip_moderate_confusion_matrix.png",
    )
    write_report(
        "SSIM strict binary, Moderate skipped",
        cm_ssim,
        metrics_ssim,
        OUT_DIR / "binary_ssim_skip_moderate_report.txt",
        note=(
            "This report maps true Good to Non-artifact and true Bad to Artifact. "
            "True Moderate cases are excluded. Prediction is forced between Good-vs-Bad "
            "using the final ensemble adjusted scores."
        ),
    )
    summary["ssim_skip_moderate"] = metrics_ssim

    origin_true = []
    origin_pred = []
    for i, s in enumerate(samples):
        label = path_origin_label(s.get("path", ""))
        if label is None:
            continue
        origin_true.append(label)
        origin_pred.append(int(pred_bin[i]))

    y_true_origin = np.asarray(origin_true, dtype=np.int64)
    y_pred_origin = np.asarray(origin_pred, dtype=np.int64)

    cm_origin = make_confusion(y_true_origin, y_pred_origin)
    metrics_origin = metrics_from_confusion(cm_origin)

    plot_confusion(
        cm_origin,
        "Binary Artifact vs Non-Artifact\nPath-origin evaluation",
        OUT_DIR / "binary_path_origin_confusion_matrix.png",
    )
    write_report(
        "Path-origin binary",
        cm_origin,
        metrics_origin,
        OUT_DIR / "binary_path_origin_report.txt",
        note=(
            "This report maps GroundTruthData_part1/Testing_GroundTruthData paths to Non-artifact "
            "and ArtifactData_part1 paths to Artifact. Prediction is forced between Good-vs-Bad "
            "using the final ensemble adjusted scores."
        ),
    )
    summary["path_origin_binary"] = metrics_origin

    (OUT_DIR / "binary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print("\nGenerated:")
    for p in [
        "binary_ssim_skip_moderate_report.txt",
        "binary_ssim_skip_moderate_confusion_matrix.png",
        "binary_path_origin_report.txt",
        "binary_path_origin_confusion_matrix.png",
        "binary_all_predictions.csv",
        "binary_summary.json",
    ]:
        print(f"  {OUT_DIR / p}")

    print("\nQuick result:")
    print(f"  SSIM skip-Moderate accuracy : {metrics_ssim['accuracy']*100:.2f}% | N={metrics_ssim['n']}")
    print(f"  Path-origin binary accuracy : {metrics_origin['accuracy']*100:.2f}% | N={metrics_origin['n']}")
    print("\nOpen folder:")
    print(f"  start {OUT_DIR}")


if __name__ == "__main__":
    main()
