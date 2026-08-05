"""
KMAR-50K — Per-Plane Thresholds + Selective Prediction / Review Mode

Purpose
-------
Uses already-saved prediction caches from apply_thresholds_tta_ensemble.py.
No model inference is run here, so this is fast and 4GB-GPU safe.

What it does
------------
1. Loads Phase 3 + Phase 5 ensemble prediction caches.
2. Rebuilds the final ensemble probabilities using best_strategy_tta_ensemble.json.
3. Tunes separate class thresholds for each MRI plane on VALIDATION only.
4. Evaluates the frozen per-plane thresholds on the locked test set.
5. Builds coverage-vs-accuracy tables using confidence/margin based Review Required mode.
6. Saves error rows for later Grad-CAM/XAI inspection.

Expected prerequisite
---------------------
Run this first if prediction caches are missing:
    python apply_thresholds_tta_ensemble.py

Environment knobs
-----------------
KMAR_BASE                 default D:\\KMAR-50K\\KMAR-50K
KMAR_PLANE_STEP           default 0.01     # use 0.02 if slow
KMAR_TARGET_ACCEPTED_ACC  default 0.86     # target accuracy for accepted/high-confidence samples
KMAR_CONFIDENCE_SCORE     default margin   # options: margin, confidence
"""

import os
import json
import glob
import csv
import pickle
from collections import defaultdict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------
# Config
# -------------------------
BASE = os.environ.get("KMAR_BASE", r"D:\KMAR-50K\KMAR-50K")
CKPT_DIR = os.path.join(BASE, "checkpoints")
PRED_CACHE_DIR = os.path.join(CKPT_DIR, "prediction_cache")
VAL_CACHE_INDEX = os.path.join(BASE, "val_cache", "index.json")
TEST_JSON = os.path.join(CKPT_DIR, "test_samples_ssim.json")
BEST_STRATEGY_JSON = os.path.join(CKPT_DIR, "best_strategy_tta_ensemble.json")

STEP = float(os.environ.get("KMAR_PLANE_STEP", "0.01"))
TARGET_ACCEPTED_ACC = float(os.environ.get("KMAR_TARGET_ACCEPTED_ACC", "0.86"))
CONFIDENCE_SCORE = os.environ.get("KMAR_CONFIDENCE_SCORE", "margin").strip().lower()

CLASS_NAMES = ["Good", "Moderate", "Bad"]
PLANE_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Transection"}

# Conservative ranges around your current winning global thresholds:
# Good=0.240, Moderate=0.600, Bad=0.790.
# These can be widened later if needed.
GOOD_RANGE = (0.10, 0.35)
MOD_RANGE = (0.40, 0.75)
BAD_RANGE = (0.55, 0.90)

COVERAGE_TARGETS = [1.00, 0.98, 0.95, 0.92, 0.90, 0.87, 0.85, 0.82, 0.80, 0.75, 0.70, 0.60, 0.50]


def banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def find_latest_cache(pattern: str) -> str:
    files = glob.glob(os.path.join(PRED_CACHE_DIR, pattern))
    if not files:
        raise FileNotFoundError(
            f"No prediction cache found for pattern: {pattern}\n"
            f"Looked in: {PRED_CACHE_DIR}\n"
            "Run first: python apply_thresholds_tta_ensemble.py"
        )
    return sorted(files, key=os.path.getmtime, reverse=True)[0]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_npz_array(npz, names):
    for name in names:
        if name in npz.files:
            return npz[name]
    return None


def samples_to_labels(samples):
    q = np.array([int(s.get("quality")) for s in samples], dtype=np.int64)
    p = np.array([int(s.get("plane")) for s in samples], dtype=np.int64)
    return q, p


def load_probs_and_labels(cache_path: str, samples, split_name: str):
    data = np.load(cache_path)
    probs = get_npz_array(data, ["probs", "probabilities", "quality_probs"])
    if probs is None:
        raise KeyError(f"No probs array found in {cache_path}. Keys: {data.files}")

    q_true = get_npz_array(data, ["q_true", "quality_true", "y_true", "labels"])
    plane_true = get_npz_array(data, ["plane_true", "p_true", "planes", "plane"])

    sample_q, sample_p = samples_to_labels(samples)
    if q_true is None:
        q_true = sample_q
    if plane_true is None:
        plane_true = sample_p

    q_true = np.asarray(q_true, dtype=np.int64)
    plane_true = np.asarray(plane_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)

    if len(probs) != len(samples):
        raise RuntimeError(
            f"{split_name}: cache length mismatch. probs={len(probs)}, samples={len(samples)}\n"
            f"Cache: {cache_path}"
        )
    if len(q_true) != len(samples) or len(plane_true) != len(samples):
        raise RuntimeError(f"{split_name}: label length mismatch.")

    return probs, q_true, plane_true


def make_range(lo, hi, step):
    # Include hi despite floating-point drift.
    return np.round(np.arange(lo, hi + step * 0.5, step), 6).astype(np.float32)


def fast_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return {"acc": float("nan"), "weighted_f1": float("nan"), "macro_f1": float("nan")}

    cm = np.bincount(3 * y_true + y_pred, minlength=9).reshape(3, 3).astype(np.float64)
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else float("nan")

    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    tp = np.diag(cm)

    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() else float("nan")
    macro_f1 = float(f1.mean())
    return {"acc": acc, "weighted_f1": weighted_f1, "macro_f1": macro_f1}


def apply_global_thresholds(probs, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, 3)
    adjusted = probs / thresholds
    pred = np.argmax(adjusted, axis=1).astype(np.int64)
    adjusted_norm = adjusted / (adjusted.sum(axis=1, keepdims=True) + 1e-12)
    sorted_adj = np.sort(adjusted_norm, axis=1)
    confidence = sorted_adj[:, -1]
    margin = sorted_adj[:, -1] - sorted_adj[:, -2]
    return pred, confidence, margin, adjusted_norm


def apply_per_plane_thresholds(probs, planes, threshold_map):
    thresholds = np.zeros((len(probs), 3), dtype=np.float32)
    for plane in [0, 1, 2]:
        mask = planes == plane
        thresholds[mask] = np.asarray(threshold_map[str(plane)]["thresholds"], dtype=np.float32)

    adjusted = probs / thresholds
    pred = np.argmax(adjusted, axis=1).astype(np.int64)
    adjusted_norm = adjusted / (adjusted.sum(axis=1, keepdims=True) + 1e-12)
    sorted_adj = np.sort(adjusted_norm, axis=1)
    confidence = sorted_adj[:, -1]
    margin = sorted_adj[:, -1] - sorted_adj[:, -2]
    return pred, confidence, margin, adjusted_norm


def tune_thresholds_for_subset(probs, y_true, metric="weighted_f1", step=STEP):
    good_values = make_range(GOOD_RANGE[0], GOOD_RANGE[1], step)
    mod_values = make_range(MOD_RANGE[0], MOD_RANGE[1], step)
    bad_values = make_range(BAD_RANGE[0], BAD_RANGE[1], step)

    best = {
        "thresholds": None,
        "acc": -1.0,
        "weighted_f1": -1.0,
        "macro_f1": -1.0,
        "selection_metric": metric,
    }
    total = len(good_values) * len(mod_values) * len(bad_values)
    checked = 0

    for g in good_values:
        for m in mod_values:
            # Vectorize across all Bad thresholds for speed enough, but keep readable.
            for b in bad_values:
                checked += 1
                thresholds = np.array([g, m, b], dtype=np.float32)
                pred, _, _, _ = apply_global_thresholds(probs, thresholds)
                met = fast_metrics(y_true, pred)
                score = met[metric]

                # Tie break: selected metric, then accuracy, then weighted F1, then macro F1.
                best_score = best[metric]
                better = (
                    score > best_score + 1e-12
                    or (
                        abs(score - best_score) <= 1e-12
                        and (
                            met["acc"], met["weighted_f1"], met["macro_f1"]
                        ) > (
                            best["acc"], best["weighted_f1"], best["macro_f1"]
                        )
                    )
                )
                if better:
                    best = {
                        "thresholds": thresholds.tolist(),
                        "acc": met["acc"],
                        "weighted_f1": met["weighted_f1"],
                        "macro_f1": met["macro_f1"],
                        "selection_metric": metric,
                    }

        # Lightweight progress per Good threshold.
        if checked % max(1, total // 5) < len(mod_values) * len(bad_values):
            pass

    return best


def tune_per_plane_thresholds(probs_val, y_val, planes_val, metric="weighted_f1"):
    threshold_map = {}
    print(f"\nTuning per-plane thresholds on validation only | metric={metric} | step={STEP}")
    print(f"Ranges: Good={GOOD_RANGE}, Moderate={MOD_RANGE}, Bad={BAD_RANGE}")

    for plane in [0, 1, 2]:
        mask = planes_val == plane
        if mask.sum() == 0:
            raise RuntimeError(f"No validation samples for plane {plane}")
        print(f"\nPlane {plane} / {PLANE_NAMES[plane]} | validation samples={mask.sum()}")
        best = tune_thresholds_for_subset(probs_val[mask], y_val[mask], metric=metric, step=STEP)
        threshold_map[str(plane)] = {
            "plane_name": PLANE_NAMES[plane],
            **best,
            "validation_samples": int(mask.sum()),
        }
        th = best["thresholds"]
        print(
            f"  Best th=[{th[0]:.3f},{th[1]:.3f},{th[2]:.3f}] | "
            f"acc={best['acc']*100:.2f}% | wf1={best['weighted_f1']:.4f} | macro={best['macro_f1']:.4f}"
        )

    return threshold_map


def print_eval_block(title, y_true, y_pred):
    met = fast_metrics(y_true, y_pred)
    banner(title)
    print(f"Accuracy    : {met['acc'] * 100:.2f}%")
    print(f"Weighted F1 : {met['weighted_f1']:.4f}")
    print(f"Macro F1    : {met['macro_f1']:.4f}")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix [Good, Moderate, Bad]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    return met


def per_plane_report(y_true, y_pred, planes, split_name):
    rows = []
    print(f"\nPer-plane {split_name} metrics:")
    for plane in [0, 1, 2]:
        mask = planes == plane
        met = fast_metrics(y_true[mask], y_pred[mask])
        rows.append({"plane": plane, "plane_name": PLANE_NAMES[plane], "n": int(mask.sum()), **met})
        print(
            f"  {PLANE_NAMES[plane]:11s} n={mask.sum():4d} | "
            f"acc={met['acc']*100:6.2f}% | wf1={met['weighted_f1']:.4f} | macro={met['macro_f1']:.4f}"
        )
    return rows


def coverage_table(y_val, pred_val, conf_val, y_test, pred_test, conf_test):
    rows = []
    if CONFIDENCE_SCORE not in {"margin", "confidence"}:
        raise ValueError("KMAR_CONFIDENCE_SCORE must be margin or confidence")

    sorted_val = np.sort(conf_val)[::-1]
    n_val = len(sorted_val)

    print(f"\nSelective prediction / Review Required mode using score={CONFIDENCE_SCORE}")
    print("Thresholds are selected from validation coverage targets only.")
    print("\ncoverage_target | val_cov val_acc val_wf1 | test_cov test_acc test_wf1 | cutoff")
    print("-" * 88)

    for target in COVERAGE_TARGETS:
        if target >= 0.9999:
            cutoff = -1e9
        else:
            keep_n = max(1, int(np.floor(target * n_val)))
            cutoff = float(sorted_val[keep_n - 1])

        val_mask = conf_val >= cutoff
        test_mask = conf_test >= cutoff

        val_met = fast_metrics(y_val[val_mask], pred_val[val_mask])
        test_met = fast_metrics(y_test[test_mask], pred_test[test_mask])

        row = {
            "coverage_target": target,
            "cutoff": cutoff,
            "validation_coverage": float(val_mask.mean()),
            "validation_accuracy": val_met["acc"],
            "validation_weighted_f1": val_met["weighted_f1"],
            "validation_macro_f1": val_met["macro_f1"],
            "validation_accepted": int(val_mask.sum()),
            "validation_rejected": int((~val_mask).sum()),
            "test_coverage": float(test_mask.mean()),
            "test_accuracy": test_met["acc"],
            "test_weighted_f1": test_met["weighted_f1"],
            "test_macro_f1": test_met["macro_f1"],
            "test_accepted": int(test_mask.sum()),
            "test_rejected": int((~test_mask).sum()),
        }
        rows.append(row)

        print(
            f"{target:14.2f} | "
            f"{row['validation_coverage']*100:6.2f}% {row['validation_accuracy']*100:6.2f}% {row['validation_weighted_f1']:.4f} | "
            f"{row['test_coverage']*100:6.2f}% {row['test_accuracy']*100:6.2f}% {row['test_weighted_f1']:.4f} | "
            f"{cutoff:.6f}"
        )

    # Choose highest validation coverage that reaches target accepted accuracy.
    qualifying = [r for r in rows if r["validation_accuracy"] >= TARGET_ACCEPTED_ACC]
    if qualifying:
        selected = max(qualifying, key=lambda r: (r["validation_coverage"], r["validation_weighted_f1"]))
        reason = f"highest validation coverage with validation accepted accuracy >= {TARGET_ACCEPTED_ACC*100:.2f}%"
    else:
        selected = max(rows, key=lambda r: (r["validation_accuracy"], r["validation_coverage"]))
        reason = f"no coverage point reached {TARGET_ACCEPTED_ACC*100:.2f}% validation accuracy; selected best validation accuracy"

    print("\nRecommended review operating point:")
    print(f"  Reason         : {reason}")
    print(f"  Cutoff         : {selected['cutoff']:.6f}")
    print(f"  Val coverage   : {selected['validation_coverage']*100:.2f}%")
    print(f"  Val accuracy   : {selected['validation_accuracy']*100:.2f}%")
    print(f"  Test coverage  : {selected['test_coverage']*100:.2f}%")
    print(f"  Test accuracy  : {selected['test_accuracy']*100:.2f}%")
    print(f"  Test weightedF1: {selected['test_weighted_f1']:.4f}")

    return rows, selected


def save_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_error_rows(samples, y_true, y_pred, planes, conf, margin, probs_adj_norm, split_name, max_rows=None):
    rows = []
    for idx, (sample, yt, yp, pl, c, mg) in enumerate(zip(samples, y_true, y_pred, planes, conf, margin)):
        if int(yt) == int(yp):
            continue
        row = {
            "split": split_name,
            "index": idx,
            "path": sample.get("path", ""),
            "slice": sample.get("slice", ""),
            "plane": int(pl),
            "plane_name": PLANE_NAMES.get(int(pl), str(pl)),
            "true": int(yt),
            "true_name": CLASS_NAMES[int(yt)],
            "pred": int(yp),
            "pred_name": CLASS_NAMES[int(yp)],
            "confidence": float(c),
            "margin": float(mg),
            "adj_good": float(probs_adj_norm[idx, 0]),
            "adj_moderate": float(probs_adj_norm[idx, 1]),
            "adj_bad": float(probs_adj_norm[idx, 2]),
        }
        rows.append(row)

    # Put highest-confidence mistakes first: best for XAI/failure analysis.
    rows.sort(key=lambda r: r["confidence"], reverse=True)
    return rows[:max_rows] if max_rows else rows


def build_review_rows(samples, y_true, y_pred, planes, conf, margin, probs_adj_norm, cutoff, split_name):
    rows = []
    for idx, (sample, yt, yp, pl, c, mg) in enumerate(zip(samples, y_true, y_pred, planes, conf, margin)):
        if c >= cutoff:
            continue
        rows.append({
            "split": split_name,
            "index": idx,
            "path": sample.get("path", ""),
            "slice": sample.get("slice", ""),
            "plane": int(pl),
            "plane_name": PLANE_NAMES.get(int(pl), str(pl)),
            "true": int(yt),
            "true_name": CLASS_NAMES[int(yt)],
            "pred_if_forced": int(yp),
            "pred_if_forced_name": CLASS_NAMES[int(yp)],
            "forced_correct": int(int(yt) == int(yp)),
            "confidence": float(c),
            "margin": float(mg),
            "adj_good": float(probs_adj_norm[idx, 0]),
            "adj_moderate": float(probs_adj_norm[idx, 1]),
            "adj_bad": float(probs_adj_norm[idx, 2]),
        })
    rows.sort(key=lambda r: r["confidence"])
    return rows


def main():
    banner("KMAR-50K Per-Plane Threshold + Review Required Evaluation")
    print(f"BASE                 : {BASE}")
    print(f"Prediction cache dir : {PRED_CACHE_DIR}")
    print(f"Plane threshold step : {STEP}")
    print(f"Confidence score     : {CONFIDENCE_SCORE}")
    print(f"Target accepted acc  : {TARGET_ACCEPTED_ACC*100:.2f}%")

    if not os.path.exists(BEST_STRATEGY_JSON):
        raise FileNotFoundError(f"Missing {BEST_STRATEGY_JSON}. Run apply_thresholds_tta_ensemble.py first.")

    strategy = load_json(BEST_STRATEGY_JSON)
    w3 = float(strategy.get("phase3_weight", 0.25))
    w5 = float(strategy.get("phase5_weight", 0.75))
    global_thresholds = np.array([
        strategy["thresholds"]["Good"],
        strategy["thresholds"]["Moderate"],
        strategy["thresholds"]["Bad"],
    ], dtype=np.float32)

    print("\nLoaded best ensemble strategy:")
    print(f"  Strategy       : {strategy.get('strategy')}")
    print(f"  Phase3 weight  : {w3}")
    print(f"  Phase5 weight  : {w5}")
    print(f"  Global th      : [{global_thresholds[0]:.3f},{global_thresholds[1]:.3f},{global_thresholds[2]:.3f}]")

    val_samples = load_json(VAL_CACHE_INDEX)
    test_samples = load_json(TEST_JSON)

    # Cache names generated by apply_thresholds_tta_ensemble.py.
    p3_val_cache = find_latest_cache("phase3_val_best_model_ssim_orig_*.npz")
    p3_test_cache = find_latest_cache("phase3_test_best_model_ssim_orig_*.npz")
    p5_val_cache = find_latest_cache("phase5_val_best_model_phase5_orig_hflip_*.npz")
    p5_test_cache = find_latest_cache("phase5_test_best_model_phase5_orig_hflip_*.npz")

    print("\nUsing prediction caches:")
    print(f"  Phase3 val  : {p3_val_cache}")
    print(f"  Phase3 test : {p3_test_cache}")
    print(f"  Phase5 val  : {p5_val_cache}")
    print(f"  Phase5 test : {p5_test_cache}")

    p3_val, y_val, plane_val = load_probs_and_labels(p3_val_cache, val_samples, "validation phase3")
    p5_val, y_val_5, plane_val_5 = load_probs_and_labels(p5_val_cache, val_samples, "validation phase5")
    p3_test, y_test, plane_test = load_probs_and_labels(p3_test_cache, test_samples, "test phase3")
    p5_test, y_test_5, plane_test_5 = load_probs_and_labels(p5_test_cache, test_samples, "test phase5")

    if not np.array_equal(y_val, y_val_5):
        raise RuntimeError("Validation labels differ between Phase3 and Phase5 caches.")
    if not np.array_equal(y_test, y_test_5):
        raise RuntimeError("Test labels differ between Phase3 and Phase5 caches.")

    # Use labels/planes from metadata/cache. Plane accuracy is ~99.9%, but these are evaluation labels.
    # Operational inference can use the predicted plane head or known series plane metadata.
    ens_val = (w3 * p3_val) + (w5 * p5_val)
    ens_test = (w3 * p3_test) + (w5 * p5_test)

    # Baseline global strategy sanity check.
    pred_val_global, conf_val_global, margin_val_global, adj_val_global = apply_global_thresholds(ens_val, global_thresholds)
    pred_test_global, conf_test_global, margin_test_global, adj_test_global = apply_global_thresholds(ens_test, global_thresholds)
    print_eval_block("VALIDATION — CURRENT GLOBAL ENSEMBLE THRESHOLDS", y_val, pred_val_global)
    print_eval_block("LOCKED TEST — CURRENT GLOBAL ENSEMBLE THRESHOLDS", y_test, pred_test_global)

    # Per-plane thresholds.
    threshold_map = tune_per_plane_thresholds(ens_val, y_val, plane_val, metric="weighted_f1")

    pred_val_plane, conf_val, margin_val, adj_val = apply_per_plane_thresholds(ens_val, plane_val, threshold_map)
    pred_test_plane, conf_test, margin_test, adj_test = apply_per_plane_thresholds(ens_test, plane_test, threshold_map)

    val_plane_met = print_eval_block("VALIDATION — PER-PLANE THRESHOLDS", y_val, pred_val_plane)
    test_plane_met = print_eval_block("LOCKED TEST — PER-PLANE THRESHOLDS", y_test, pred_test_plane)

    val_plane_rows = per_plane_report(y_val, pred_val_plane, plane_val, "validation")
    test_plane_rows = per_plane_report(y_test, pred_test_plane, plane_test, "locked test")

    # Select score for abstention.
    if CONFIDENCE_SCORE == "confidence":
        score_val = conf_val
        score_test = conf_test
    else:
        score_val = margin_val
        score_test = margin_test

    cov_rows, selected_review = coverage_table(
        y_val, pred_val_plane, score_val,
        y_test, pred_test_plane, score_test,
    )

    # Report selected review point classification on accepted test samples.
    cutoff = selected_review["cutoff"]
    accept_mask_test = score_test >= cutoff
    review_mask_test = ~accept_mask_test
    print_eval_block(
        f"LOCKED TEST — ACCEPTED ONLY / REVIEW REQUIRED CUTOFF={cutoff:.6f}",
        y_test[accept_mask_test],
        pred_test_plane[accept_mask_test],
    )
    print(f"Accepted samples : {int(accept_mask_test.sum())}/{len(y_test)} ({accept_mask_test.mean()*100:.2f}%)")
    print(f"Review required  : {int(review_mask_test.sum())}/{len(y_test)} ({review_mask_test.mean()*100:.2f}%)")

    # Save artifacts.
    out_threshold_json = os.path.join(CKPT_DIR, "per_plane_thresholds.json")
    out_threshold_pkl = os.path.join(CKPT_DIR, "per_plane_thresholds.pkl")
    out_coverage_csv = os.path.join(CKPT_DIR, "coverage_vs_accuracy_per_plane.csv")
    out_error_csv = os.path.join(CKPT_DIR, "per_plane_error_analysis_rows.csv")
    out_review_csv = os.path.join(CKPT_DIR, "review_required_rows.csv")
    out_summary_json = os.path.join(CKPT_DIR, "per_plane_abstention_summary.json")

    summary = {
        "base": BASE,
        "source_strategy": strategy,
        "phase3_weight": w3,
        "phase5_weight": w5,
        "global_thresholds": global_thresholds.tolist(),
        "per_plane_thresholds": threshold_map,
        "validation_per_plane_metrics": val_plane_met,
        "test_per_plane_metrics": test_plane_met,
        "validation_plane_rows": val_plane_rows,
        "test_plane_rows": test_plane_rows,
        "coverage_rows": cov_rows,
        "selected_review_operating_point": selected_review,
        "confidence_score": CONFIDENCE_SCORE,
        "note": "Per-plane thresholds and review cutoff are selected from validation only. Locked test metrics are reported after selection.",
    }

    with open(out_threshold_json, "w", encoding="utf-8") as f:
        json.dump(threshold_map, f, indent=2)
    with open(out_threshold_pkl, "wb") as f:
        pickle.dump(threshold_map, f)
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_csv(out_coverage_csv, cov_rows)

    error_rows = build_error_rows(
        test_samples, y_test, pred_test_plane, plane_test,
        conf_test, margin_test, adj_test,
        split_name="locked_test",
    )
    save_csv(out_error_csv, error_rows)

    review_rows = build_review_rows(
        test_samples, y_test, pred_test_plane, plane_test,
        score_test, margin_test, adj_test,
        cutoff=cutoff,
        split_name="locked_test",
    )
    save_csv(out_review_csv, review_rows)

    print("\nSaved outputs:")
    print(f"  Per-plane thresholds : {out_threshold_json}")
    print(f"  Per-plane thresholds : {out_threshold_pkl}")
    print(f"  Summary              : {out_summary_json}")
    print(f"  Coverage table        : {out_coverage_csv}")
    print(f"  Error rows for XAI    : {out_error_csv}")
    print(f"  Review rows           : {out_review_csv}")

    print("\nFinal decision guide:")
    print("  1. If LOCKED TEST — PER-PLANE THRESHOLDS > 82.69%, use per-plane thresholds as new main result.")
    print("  2. If accepted-only accuracy is high at useful coverage, report it as confidence-aware QA, not full-coverage accuracy.")
    print("  3. Use per_plane_error_analysis_rows.csv and review_required_rows.csv for Grad-CAM/XAI case selection.")


if __name__ == "__main__":
    main()
