"""
KMAR-50K Final Ensemble XAI / Grad-CAM Generator
================================================

Purpose
-------
Generates XAI proof images for the frozen final KMAR-50K MRI quality classifier.

Final system:
    Phase 3 single-slice EfficientNet-B0   weight = 0.25
    Phase 5 multi-slice EfficientNet-B0    weight = 0.75
    Thresholds: Good=0.240, Moderate=0.600, Bad=0.790

What this script produces:
    xai_outputs/
    ├── correct_good/
    ├── correct_moderate/
    ├── correct_bad/
    ├── good_to_moderate_errors/
    ├── good_to_bad_errors/
    ├── moderate_to_good_errors/
    ├── moderate_to_bad_errors/
    ├── bad_to_good_errors/
    ├── bad_to_moderate_errors/
    ├── review_required/
    ├── xai_case_table.csv
    ├── xai_summary.md
    └── xai_overview_contact_sheet.png

Important:
- This is inference/XAI only. No training.
- Uses batch size 1.
- Uses CUDA if available, CPU fallback otherwise.
- Uses existing prediction caches for case selection.
- Does not upload/copy dataset.
- Grad-CAM is generated separately for:
    1) Phase 3 single-slice model
    2) Phase 5 multi-slice model
- The final prediction shown in each panel is the frozen ensemble decision.

Prerequisites:
    D:\\KMAR-50K\\KMAR-50K\\checkpoints\\best_model_ssim.pt
    D:\\KMAR-50K\\KMAR-50K\\checkpoints\\best_model_phase5.pt
    D:\\KMAR-50K\\KMAR-50K\\checkpoints\\best_strategy_tta_ensemble.json
    D:\\KMAR-50K\\KMAR-50K\\checkpoints\\test_samples_ssim.json
    D:\\KMAR-50K\\KMAR-50K\\checkpoints\\prediction_cache\\phase3_test_*.npz
    D:\\KMAR-50K\\KMAR-50K\\checkpoints\\prediction_cache\\phase5_test_*orig_hflip*.npz

Run:
    cd /d D:\\KMAR-50K\\KMAR-50K
    python generate_xai_gradcam_final_ensemble.py

Optional environment variables:
    set KMAR_XAI_PER_CATEGORY=5
    set KMAR_XAI_MAX_TOTAL=60
    set KMAR_XAI_DEVICE=cuda
"""

from __future__ import annotations

import csv
import glob
import json
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

BASE = Path(os.environ.get("KMAR_BASE", r"D:\KMAR-50K\KMAR-50K"))
CKPT_DIR = BASE / "checkpoints"
PRED_CACHE_DIR = CKPT_DIR / "prediction_cache"

PHASE3_MODEL = CKPT_DIR / "best_model_ssim.pt"
PHASE5_MODEL = CKPT_DIR / "best_model_phase5.pt"
BEST_STRATEGY_JSON = CKPT_DIR / "best_strategy_tta_ensemble.json"
TEST_JSON = CKPT_DIR / "test_samples_ssim.json"

ERROR_ROWS_CSV = CKPT_DIR / "per_plane_error_analysis_rows.csv"
REVIEW_ROWS_CSV = CKPT_DIR / "review_required_rows.csv"
ABSTENTION_SUMMARY_JSON = CKPT_DIR / "per_plane_abstention_summary.json"

OUT_DIR = BASE / "xai_outputs"
README_ASSETS_DIR = BASE / "readme_assets"

IMG_SIZE = 224
PER_CATEGORY = int(os.environ.get("KMAR_XAI_PER_CATEGORY", "5"))
MAX_TOTAL = int(os.environ.get("KMAR_XAI_MAX_TOTAL", "80"))

DEVICE_NAME = os.environ.get("KMAR_XAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE_NAME if DEVICE_NAME == "cpu" or torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Good", "Moderate", "Bad"]
PLANE_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Transection"}

CATEGORY_ORDER = [
    "correct_good",
    "correct_moderate",
    "correct_bad",
    "good_to_moderate_errors",
    "good_to_bad_errors",
    "moderate_to_good_errors",
    "moderate_to_bad_errors",
    "bad_to_good_errors",
    "bad_to_moderate_errors",
    "review_required",
]

for d in [OUT_DIR, README_ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL
# =============================================================================

class KMARMultiTask(nn.Module):
    """Same architecture used by Phase 3 and Phase 5 checkpoints."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.head_quality = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )
        self.head_plane = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.head_quality(f), self.head_plane(f)


def load_model(path: Path, label: str) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")

    print(f"Loading {label}: {path}")
    model = KMARMultiTask()
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def find_target_layer(model: KMARMultiTask) -> nn.Module:
    """
    For timm EfficientNet-B0, conv_head is a stable late convolution layer.
    Fallback: last Conv2d in the backbone.
    """
    if hasattr(model.backbone, "conv_head"):
        return model.backbone.conv_head

    last_conv = None
    for _, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model: KMARMultiTask, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, x: torch.Tensor, target_class: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            heatmap: HxW float [0,1]
            q_probs: 3-class quality probabilities
            p_probs: 3-class plane probabilities
        """
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        q_logits, p_logits = self.model(x)
        q_probs = torch.softmax(q_logits, dim=1)
        p_probs = torch.softmax(p_logits, dim=1)

        score = q_logits[:, int(target_class)].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations/gradients: [B, C, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_np = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        cam_np = cam_np - cam_np.min()
        denom = cam_np.max() + 1e-8
        cam_np = cam_np / denom

        return (
            cam_np.astype(np.float32),
            q_probs[0].detach().cpu().numpy().astype(np.float32),
            p_probs[0].detach().cpu().numpy().astype(np.float32),
        )


# =============================================================================
# DATA / PREPROCESSING
# =============================================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_slice(proxy, idx: int) -> np.ndarray:
    idx = int(idx)
    data = proxy.dataobj
    if len(proxy.shape) == 4:
        return np.asarray(data[:, :, idx, 0], dtype=np.float32)
    return np.asarray(data[:, :, idx], dtype=np.float32)


def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    return (img - mn) / (mx - mn + 1e-8)


def prepare_inputs(sample: dict) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """
    Returns:
        phase3_input: [1,3,224,224] tensor
        phase5_input: [1,3,224,224] tensor
        display_img: [224,224] current slice normalized [0,1]
        meta: extra info
    """
    path = sample["path"]
    sl_idx = int(sample["slice"])

    proxy = nib.load(path)
    n_slices = int(proxy.shape[2])
    sl_idx = max(0, min(sl_idx, n_slices - 1))

    prev_sl = normalize01(read_slice(proxy, max(0, sl_idx - 1)))
    curr_sl = normalize01(read_slice(proxy, sl_idx))
    next_sl = normalize01(read_slice(proxy, min(n_slices - 1, sl_idx + 1)))

    curr_224 = cv2.resize(curr_sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Phase 3: current slice repeated.
    p3 = np.stack([curr_224, curr_224, curr_224], axis=0).astype(np.float32)

    # Phase 5: prev/current/next.
    prev_224 = cv2.resize(prev_sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    next_224 = cv2.resize(next_sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    p5 = np.stack([prev_224, curr_224, next_224], axis=0).astype(np.float32)

    # Normalize to training convention: (x - 0.5) / 0.5
    p3 = (p3 - 0.5) / 0.5
    p5 = (p5 - 0.5) / 0.5

    p3_t = torch.tensor(p3, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    p5_t = torch.tensor(p5, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    meta = {
        "path": path,
        "slice": sl_idx,
        "n_slices": n_slices,
    }

    return p3_t, p5_t, curr_224.astype(np.float32), meta


def find_latest_cache(pattern: str) -> Path:
    files = glob.glob(str(PRED_CACHE_DIR / pattern))
    if not files:
        raise FileNotFoundError(
            f"No prediction cache found for pattern: {pattern}\n"
            f"Looked in: {PRED_CACHE_DIR}\n\n"
            "Run first:\n"
            "    set KMAR_EVAL_BATCH_SIZE=2\n"
            "    set KMAR_TTA_MODES=orig,hflip\n"
            "    python apply_thresholds_tta_ensemble.py"
        )
    return Path(sorted(files, key=os.path.getmtime, reverse=True)[0])


def load_prediction_caches(test_samples: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p3_cache = find_latest_cache("phase3_test_best_model_ssim_orig_*.npz")
    p5_cache = find_latest_cache("phase5_test_best_model_phase5_orig_hflip_*.npz")

    print(f"Using Phase 3 cache: {p3_cache}")
    print(f"Using Phase 5 cache: {p5_cache}")

    p3 = np.load(str(p3_cache))
    p5 = np.load(str(p5_cache))

    p3_probs = np.asarray(p3["probs"], dtype=np.float32)
    p5_probs = np.asarray(p5["probs"], dtype=np.float32)

    if len(p3_probs) != len(test_samples) or len(p5_probs) != len(test_samples):
        raise RuntimeError(
            f"Prediction cache length mismatch. test={len(test_samples)}, "
            f"phase3={len(p3_probs)}, phase5={len(p5_probs)}"
        )

    q_true = np.asarray(p5["q_true"] if "q_true" in p5.files else [s["quality"] for s in test_samples], dtype=np.int64)
    plane_true = np.asarray(p5["p_true"] if "p_true" in p5.files else [s["plane"] for s in test_samples], dtype=np.int64)
    return p3_probs, p5_probs, q_true, plane_true


def ensemble_decisions(
    p3_probs: np.ndarray,
    p5_probs: np.ndarray,
    strategy: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w3 = float(strategy.get("phase3_weight", 0.25))
    w5 = float(strategy.get("phase5_weight", 0.75))
    thresholds = np.array([
        strategy["thresholds"]["Good"],
        strategy["thresholds"]["Moderate"],
        strategy["thresholds"]["Bad"],
    ], dtype=np.float32).reshape(1, 3)

    probs = (w3 * p3_probs) + (w5 * p5_probs)
    adjusted = probs / thresholds
    adjusted_norm = adjusted / (adjusted.sum(axis=1, keepdims=True) + 1e-12)

    pred = np.argmax(adjusted, axis=1).astype(np.int64)
    sorted_adj = np.sort(adjusted_norm, axis=1)
    confidence = sorted_adj[:, -1]
    margin = sorted_adj[:, -1] - sorted_adj[:, -2]

    return probs, pred, confidence.astype(np.float32), margin.astype(np.float32)


# =============================================================================
# CASE SELECTION
# =============================================================================

def category_name(true_label: int, pred_label: int, accepted_correct: Optional[bool] = None) -> str:
    if int(true_label) == int(pred_label):
        if int(true_label) == 0:
            return "correct_good"
        if int(true_label) == 1:
            return "correct_moderate"
        if int(true_label) == 2:
            return "correct_bad"

    mapping = {
        (0, 1): "good_to_moderate_errors",
        (0, 2): "good_to_bad_errors",
        (1, 0): "moderate_to_good_errors",
        (1, 2): "moderate_to_bad_errors",
        (2, 0): "bad_to_good_errors",
        (2, 1): "bad_to_moderate_errors",
    }
    return mapping.get((int(true_label), int(pred_label)), "other_errors")


def select_cases(
    test_samples: List[dict],
    y_true: np.ndarray,
    plane_true: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
) -> List[dict]:
    rows = []

    for idx, sample in enumerate(test_samples):
        true_i = int(y_true[idx])
        pred_i = int(pred[idx])
        cat = category_name(true_i, pred_i)

        rows.append({
            "source": "final_global_ensemble",
            "category": cat,
            "index": int(idx),
            "path": sample["path"],
            "slice": int(sample["slice"]),
            "plane": int(plane_true[idx]),
            "plane_name": PLANE_NAMES.get(int(plane_true[idx]), str(plane_true[idx])),
            "true": true_i,
            "true_name": CLASS_NAMES[true_i],
            "pred": pred_i,
            "pred_name": CLASS_NAMES[pred_i],
            "confidence": float(confidence[idx]),
            "margin": float(margin[idx]),
            "sort_score": float(margin[idx] if true_i == pred_i else confidence[idx]),
        })

    selected = []
    for cat in CATEGORY_ORDER:
        if cat == "review_required":
            continue
        cat_rows = [r for r in rows if r["category"] == cat]
        if cat.startswith("correct_"):
            cat_rows.sort(key=lambda r: (r["margin"], r["confidence"]), reverse=True)
        else:
            # High-confidence mistakes are useful for failure analysis.
            cat_rows.sort(key=lambda r: (r["confidence"], r["margin"]), reverse=True)
        selected.extend(cat_rows[:PER_CATEGORY])

    # Add review-required cases from saved CSV if available.
    if REVIEW_ROWS_CSV.exists():
        review_df = pd.read_csv(REVIEW_ROWS_CSV)
        review_df = review_df.sort_values(["margin", "confidence"], ascending=[True, True])
        for _, rr in review_df.head(PER_CATEGORY).iterrows():
            selected.append({
                "source": "review_required_rows_csv",
                "category": "review_required",
                "index": int(rr["index"]),
                "path": str(rr["path"]),
                "slice": int(rr["slice"]),
                "plane": int(rr["plane"]),
                "plane_name": str(rr.get("plane_name", PLANE_NAMES.get(int(rr["plane"]), ""))),
                "true": int(rr["true"]),
                "true_name": str(rr["true_name"]),
                "pred": int(rr["pred_if_forced"]),
                "pred_name": str(rr["pred_if_forced_name"]),
                "confidence": float(rr["confidence"]),
                "margin": float(rr["margin"]),
                "sort_score": -float(rr["margin"]),
            })
    else:
        # Fallback: lowest margin from final global ensemble.
        low_margin = sorted(rows, key=lambda r: r["margin"])[:PER_CATEGORY]
        for r in low_margin:
            r = dict(r)
            r["category"] = "review_required"
            r["source"] = "lowest_margin_fallback"
            selected.append(r)

    # Remove duplicate exact sample/category pairs while preserving order.
    seen = set()
    deduped = []
    for r in selected:
        key = (r["category"], r["path"], int(r["slice"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
        if len(deduped) >= MAX_TOTAL:
            break

    return deduped


# =============================================================================
# VISUALIZATION
# =============================================================================

def colorize_cam(cam: np.ndarray) -> np.ndarray:
    cam_u8 = np.uint8(np.clip(cam, 0, 1) * 255)
    colored = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return colored


def overlay_cam(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    heat = colorize_cam(cam)
    out = (1 - alpha) * gray_rgb + alpha * heat
    return np.clip(out, 0, 1)


def safe_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:180]


def save_case_panel(
    row: dict,
    display_img: np.ndarray,
    cam3: np.ndarray,
    cam5: np.ndarray,
    p3_probs: np.ndarray,
    p5_probs: np.ndarray,
    p3_plane_probs: np.ndarray,
    p5_plane_probs: np.ndarray,
    out_path: Path,
):
    overlay3 = overlay_cam(display_img, cam3)
    overlay5 = overlay_cam(display_img, cam5)
    heat3 = colorize_cam(cam3)
    heat5 = colorize_cam(cam5)

    fig = plt.figure(figsize=(15.5, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(display_img, cmap="gray")
    ax0.set_title("Current MRI slice")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(heat3)
    ax1.set_title("Phase 3 Grad-CAM")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(overlay3)
    ax2.set_title("Phase 3 overlay")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    info = (
        f"Category: {row['category']}\n"
        f"True: {row['true_name']} | Pred: {row['pred_name']}\n"
        f"Plane: {row['plane_name']}\n"
        f"Slice: {row['slice']}\n"
        f"Confidence: {row['confidence']:.4f}\n"
        f"Margin: {row['margin']:.4f}\n\n"
        f"Phase 3 q-probs:\n"
        f"  Good={p3_probs[0]:.3f}\n"
        f"  Moderate={p3_probs[1]:.3f}\n"
        f"  Bad={p3_probs[2]:.3f}\n\n"
        f"Phase 5 q-probs:\n"
        f"  Good={p5_probs[0]:.3f}\n"
        f"  Moderate={p5_probs[1]:.3f}\n"
        f"  Bad={p5_probs[2]:.3f}"
    )
    ax3.text(0, 1, info, va="top", ha="left", fontsize=10, family="monospace")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(display_img, cmap="gray")
    ax4.set_title("Current slice repeated/center")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(heat5)
    ax5.set_title("Phase 5 Grad-CAM")
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(overlay5)
    ax6.set_title("Phase 5 overlay")
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis("off")
    path_short = str(row["path"])
    if len(path_short) > 68:
        path_short = "..." + path_short[-68:]

    info2 = (
        "Final ensemble:\n"
        "  Phase3 weight=0.25\n"
        "  Phase5 weight=0.75\n"
        "  Thresholds=[0.240,0.600,0.790]\n\n"
        f"Phase 3 plane probs:\n"
        f"  Sag={p3_plane_probs[0]:.3f}\n"
        f"  Cor={p3_plane_probs[1]:.3f}\n"
        f"  Tran={p3_plane_probs[2]:.3f}\n\n"
        f"Phase 5 plane probs:\n"
        f"  Sag={p5_plane_probs[0]:.3f}\n"
        f"  Cor={p5_plane_probs[1]:.3f}\n"
        f"  Tran={p5_plane_probs[2]:.3f}\n\n"
        f"File:\n{path_short}"
    )
    ax7.text(0, 1, info2, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(
        f"KMAR-50K XAI: {row['category']} | True={row['true_name']} | Pred={row['pred_name']}",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(image_paths: List[Path], out_path: Path, max_images: int = 12):
    paths = image_paths[:max_images]
    if not paths:
        return

    thumbs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (420, 230), interpolation=cv2.INTER_AREA)
        thumbs.append((p, img))

    if not thumbs:
        return

    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (p, img) in zip(axes.ravel(), thumbs):
        ax.imshow(img)
        ax.set_title(p.parent.name + "/" + p.name[:45], fontsize=8)
        ax.axis("off")

    fig.suptitle("KMAR-50K XAI Overview Contact Sheet", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("KMAR-50K Final Ensemble Grad-CAM / XAI")
    print("=" * 80)
    print(f"BASE       : {BASE}")
    print(f"DEVICE     : {DEVICE}")
    print(f"Output dir : {OUT_DIR}")
    print(f"Per category: {PER_CATEGORY}")

    required = [
        PHASE3_MODEL,
        PHASE5_MODEL,
        BEST_STRATEGY_JSON,
        TEST_JSON,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    test_samples = load_json(TEST_JSON)
    strategy = load_json(BEST_STRATEGY_JSON)

    p3_probs_cache, p5_probs_cache, y_true, plane_true = load_prediction_caches(test_samples)
    ens_probs, pred, confidence, margin = ensemble_decisions(p3_probs_cache, p5_probs_cache, strategy)

    selected_cases = select_cases(test_samples, y_true, plane_true, pred, confidence, margin)
    if not selected_cases:
        raise RuntimeError("No XAI cases selected.")

    print(f"Selected XAI cases: {len(selected_cases)}")

    # Load models after case selection to avoid unnecessary GPU memory use if selection fails.
    phase3 = load_model(PHASE3_MODEL, "Phase 3 SSIM single-slice")
    phase5 = load_model(PHASE5_MODEL, "Phase 5 multi-slice")

    cam3_runner = GradCAM(phase3, find_target_layer(phase3))
    cam5_runner = GradCAM(phase5, find_target_layer(phase5))

    case_table = []
    generated_images = []

    try:
        for i, row in enumerate(selected_cases, 1):
            cat = row["category"]
            cat_dir = OUT_DIR / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{i:02d}/{len(selected_cases):02d}] {cat} | true={row['true_name']} pred={row['pred_name']} slice={row['slice']}")

            sample = {
                "path": row["path"],
                "slice": int(row["slice"]),
                "quality": int(row["true"]),
                "plane": int(row["plane"]),
            }

            try:
                p3_input, p5_input, display_img, meta = prepare_inputs(sample)

                target_class = int(row["pred"])
                cam3, p3_q_probs, p3_plane_probs = cam3_runner(p3_input, target_class)
                phase3.zero_grad(set_to_none=True)

                cam5, p5_q_probs, p5_plane_probs = cam5_runner(p5_input, target_class)
                phase5.zero_grad(set_to_none=True)

                filename = safe_filename(
                    f"{i:02d}_{cat}_true-{row['true_name']}_pred-{row['pred_name']}_idx-{row['index']}_slice-{row['slice']}.png"
                )
                out_path = cat_dir / filename

                save_case_panel(
                    row=row,
                    display_img=display_img,
                    cam3=cam3,
                    cam5=cam5,
                    p3_probs=p3_q_probs,
                    p5_probs=p5_q_probs,
                    p3_plane_probs=p3_plane_probs,
                    p5_plane_probs=p5_plane_probs,
                    out_path=out_path,
                )

                generated_images.append(out_path)

                case_table.append({
                    "case_id": i,
                    "category": cat,
                    "source": row["source"],
                    "index": row["index"],
                    "path": row["path"],
                    "slice": row["slice"],
                    "plane": row["plane"],
                    "plane_name": row["plane_name"],
                    "true": row["true"],
                    "true_name": row["true_name"],
                    "pred": row["pred"],
                    "pred_name": row["pred_name"],
                    "confidence": row["confidence"],
                    "margin": row["margin"],
                    "image": str(out_path.relative_to(BASE)),
                    "phase3_good": float(p3_q_probs[0]),
                    "phase3_moderate": float(p3_q_probs[1]),
                    "phase3_bad": float(p3_q_probs[2]),
                    "phase5_good": float(p5_q_probs[0]),
                    "phase5_moderate": float(p5_q_probs[1]),
                    "phase5_bad": float(p5_q_probs[2]),
                })

                del p3_input, p5_input
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"  WARNING: failed case {i}: {e}")
                case_table.append({
                    "case_id": i,
                    "category": cat,
                    "source": row.get("source", ""),
                    "index": row.get("index", ""),
                    "path": row.get("path", ""),
                    "slice": row.get("slice", ""),
                    "plane": row.get("plane", ""),
                    "plane_name": row.get("plane_name", ""),
                    "true": row.get("true", ""),
                    "true_name": row.get("true_name", ""),
                    "pred": row.get("pred", ""),
                    "pred_name": row.get("pred_name", ""),
                    "confidence": row.get("confidence", ""),
                    "margin": row.get("margin", ""),
                    "image": "",
                    "error": str(e),
                })

    finally:
        cam3_runner.close()
        cam5_runner.close()

    case_csv = OUT_DIR / "xai_case_table.csv"
    pd.DataFrame(case_table).to_csv(case_csv, index=False)

    contact_sheet = OUT_DIR / "xai_overview_contact_sheet.png"
    make_contact_sheet(generated_images, contact_sheet, max_images=12)

    # Copy overview to README assets for easy README/thesis reference.
    if contact_sheet.exists():
        try:
            import shutil
            shutil.copy2(contact_sheet, README_ASSETS_DIR / "xai_overview_contact_sheet.png")
        except Exception:
            pass

    summary_md = OUT_DIR / "xai_summary.md"
    counts = pd.DataFrame(case_table).groupby("category").size().to_dict() if case_table else {}

    lines = []
    lines.append("# KMAR-50K XAI / Grad-CAM Summary\n\n")
    lines.append("## Final model explained\n\n")
    lines.append("- Final full-coverage model: Phase 3 + Phase 5 probability ensemble\n")
    lines.append("- Phase 3 weight: 0.25\n")
    lines.append("- Phase 5 weight: 0.75\n")
    lines.append("- TTA used for final metric: original + horizontal flip\n")
    lines.append("- Thresholds: Good=0.240, Moderate=0.600, Bad=0.790\n")
    lines.append("- Full-coverage locked-test accuracy: 82.69%\n")
    lines.append("- Confidence-aware accepted-only accuracy: 86.23% at 88.91% coverage\n\n")
    lines.append("## Generated XAI case counts\n\n")
    for cat in CATEGORY_ORDER:
        lines.append(f"- {cat}: {counts.get(cat, 0)}\n")
    lines.append("\n## Outputs\n\n")
    lines.append(f"- Case table: `{case_csv.relative_to(BASE)}`\n")
    if contact_sheet.exists():
        lines.append(f"- Overview contact sheet: `{contact_sheet.relative_to(BASE)}`\n")
    lines.append("\n## Suggested README snippet\n\n")
    lines.append("```markdown\n")
    lines.append("## XAI / Grad-CAM Evidence\n\n")
    lines.append("![XAI overview](readme_assets/xai_overview_contact_sheet.png)\n\n")
    lines.append("Grad-CAM was generated for representative correct predictions, high-confidence errors, and Review Required cases. ")
    lines.append("For each selected sample, Phase 3 and Phase 5 heatmaps were generated separately to explain the final ensemble decision.\n")
    lines.append("```\n")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\nDone.")
    print(f"Generated XAI images : {len(generated_images)}")
    print(f"Case table           : {case_csv}")
    print(f"Summary              : {summary_md}")
    if contact_sheet.exists():
        print(f"Contact sheet        : {contact_sheet}")
        print(f"README asset copied  : {README_ASSETS_DIR / 'xai_overview_contact_sheet.png'}")

    print("\nNext:")
    print("1. Inspect xai_outputs\\xai_overview_contact_sheet.png")
    print("2. Inspect xai_outputs\\xai_case_table.csv")
    print("3. If images look good, run:")
    print("   Copy-Item .\\xai_outputs\\xai_overview_contact_sheet.png .\\readme_assets\\ -Force")
    print("4. Then update README with the generated XAI evidence.")


if __name__ == "__main__":
    main()
