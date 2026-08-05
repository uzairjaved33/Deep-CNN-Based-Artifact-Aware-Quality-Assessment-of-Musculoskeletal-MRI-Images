"""
apply_thresholds_tta_ensemble.py — KMAR-50K Phase 5 improvement evaluator

Purpose
-------
Safely tries the next non-training improvements:
1) Fine threshold search for Phase 5.
2) Phase 5 TTA, default: original + horizontal flip.
3) Optional Phase 3 + Phase 5 probability ensemble if best_model_ssim.pt exists.

Rules
-----
- Thresholds are tuned on VALIDATION ONLY.
- Locked test set is evaluated only after validation selection.
- No full NIfTI volume is loaded. Only the needed slice(s) are read with nibabel dataobj.
- Safe defaults for 4GB CUDA GPUs: batch_size=4, num_workers=0, fp16 inference.
- Prediction cache is saved under checkpoints/prediction_cache so reruns are faster.

Expected paths
--------------
D:\KMAR-50K\KMAR-50K\samples_ssim_labeled.json
D:\KMAR-50K\KMAR-50K\val_cache\index.json
D:\KMAR-50K\KMAR-50K\checkpoints\test_samples_ssim.json
D:\KMAR-50K\KMAR-50K\checkpoints\best_model_phase5.pt
D:\KMAR-50K\KMAR-50K\checkpoints\best_model_ssim.pt        # optional Phase 3

Useful environment overrides
----------------------------
set KMAR_EVAL_BATCH_SIZE=4
set KMAR_TTA_MODES=orig,hflip
set KMAR_REBUILD_PRED_CACHE=1
set KMAR_ENSEMBLE_WEIGHT_STEP=0.05

TTA modes supported: orig,hflip,rot5,rot-5
Keep default orig,hflip first. Rotation is slower.
"""

import os
import json
import gc
import cv2
import pickle
import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import timm
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try:
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    HAS_TORCHVISION_TTA = True
except Exception:
    HAS_TORCHVISION_TTA = False

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────
BASE = r"D:\KMAR-50K\KMAR-50K"
CKPT_DIR = os.path.join(BASE, "checkpoints")
VAL_CACHE_DIR = os.path.join(BASE, "val_cache")
LABELED_JSON = os.path.join(BASE, "samples_ssim_labeled.json")
TEST_JSON = os.path.join(CKPT_DIR, "test_samples_ssim.json")

PHASE5_MODEL_PATH = os.path.join(CKPT_DIR, "best_model_phase5.pt")
PHASE3_MODEL_PATH = os.path.join(CKPT_DIR, "best_model_ssim.pt")

PRED_CACHE_DIR = os.path.join(CKPT_DIR, "prediction_cache")
REPORT_PATH = os.path.join(CKPT_DIR, "tta_ensemble_evaluation_report.txt")
BEST_JSON_PATH = os.path.join(CKPT_DIR, "best_strategy_tta_ensemble.json")
BEST_PKL_PATH = os.path.join(CKPT_DIR, "optimal_thresholds_tta_ensemble.pkl")
BEST_THRESH_JSON_PATH = os.path.join(CKPT_DIR, "optimal_thresholds_tta_ensemble.json")

IMG_SIZE = 224
BATCH_SIZE = int(os.environ.get("KMAR_EVAL_BATCH_SIZE", "4"))
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
REBUILD_CACHE = os.environ.get("KMAR_REBUILD_PRED_CACHE", "0") == "1"
CLASS_NAMES = ["Good", "Moderate", "Bad"]
PLANE_NAMES = ["Sagittal", "Coronal", "Transection"]

# Fine threshold window around your recovered Phase 5 best thresholds:
# Good=0.15, Moderate=0.45, Bad=0.70
GOOD_RANGE = (0.08, 0.25)
MOD_RANGE = (0.35, 0.60)
BAD_RANGE = (0.55, 0.85)
FINE_STEP = float(os.environ.get("KMAR_FINE_THRESHOLD_STEP", "0.01"))
COARSE_STEP = float(os.environ.get("KMAR_COARSE_THRESHOLD_STEP", "0.05"))
ENSEMBLE_WEIGHT_STEP = float(os.environ.get("KMAR_ENSEMBLE_WEIGHT_STEP", "0.05"))
TTA_MODES = [m.strip() for m in os.environ.get("KMAR_TTA_MODES", "orig,hflip").split(",") if m.strip()]

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PRED_CACHE_DIR, exist_ok=True)

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ── Models ───────────────────────────────────────────────────
class KMARMultiTask(nn.Module):
    """Architecture used by both Phase 3 and Phase 5 checkpoints."""
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


def load_model(model_path: str, name: str) -> Tuple[nn.Module, dict]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{name} model not found: {model_path}")
    print(f"🔄 Loading {name}: {model_path}")
    model = KMARMultiTask().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


# ── Datasets ─────────────────────────────────────────────────
class Phase5DiskDataset(Dataset):
    """Validation cache from Phase 5 preprocessing: already 3x224x224 normalized."""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Validation cache index not found: {index_path}")
        with open(index_path, "r") as f:
            self.samples = json.load(f)
        print(f"📂 Loaded Phase 5 validation cache: {len(self.samples):,} samples from {cache_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        data = np.load(os.path.join(self.cache_dir, s["file"]), allow_pickle=True).item()
        return (
            torch.tensor(data["img"], dtype=torch.float32),
            torch.tensor(int(data["quality"]), dtype=torch.long),
            torch.tensor(int(data["plane"]), dtype=torch.long),
        )


class Phase5RawMultiSliceDataset(Dataset):
    """Phase 5 raw loader: [previous, current, next] slice as 3 channels."""
    def __init__(self, samples: List[dict], img_size: int = IMG_SIZE):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_slice(proxy, idx: int) -> np.ndarray:
        data = proxy.dataobj
        if len(proxy.shape) == 4:
            return np.asarray(data[:, :, int(idx), 0], dtype=np.float32)
        return np.asarray(data[:, :, int(idx)], dtype=np.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        proxy = nib.load(s["path"])
        n_slices = int(proxy.shape[2])
        sl_idx = max(0, min(int(s["slice"]), n_slices - 1))

        prev_sl = self._read_slice(proxy, max(0, sl_idx - 1))
        curr_sl = self._read_slice(proxy, sl_idx)
        next_sl = self._read_slice(proxy, min(n_slices - 1, sl_idx + 1))
        slc = np.stack([prev_sl, curr_sl, next_sl], axis=0)

        for c in range(3):
            mn, mx = float(slc[c].min()), float(slc[c].max())
            slc[c] = (slc[c] - mn) / (mx - mn + 1e-8)

        out = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
        for c in range(3):
            out[c] = cv2.resize(slc[c], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        out = (out - 0.5) / 0.5
        return (
            torch.tensor(out, dtype=torch.float32),
            torch.tensor(int(s["quality"]), dtype=torch.long),
            torch.tensor(int(s["plane"]), dtype=torch.long),
        )


class Phase3SingleSliceDataset(Dataset):
    """Phase 3 loader: current slice repeated into 3 channels."""
    def __init__(self, samples: List[dict], img_size: int = IMG_SIZE):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_slice(proxy, idx: int) -> np.ndarray:
        data = proxy.dataobj
        if len(proxy.shape) == 4:
            return np.asarray(data[:, :, int(idx), 0], dtype=np.float32)
        return np.asarray(data[:, :, int(idx)], dtype=np.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        proxy = nib.load(s["path"])
        n_slices = int(proxy.shape[2])
        sl_idx = max(0, min(int(s["slice"]), n_slices - 1))
        img = self._read_slice(proxy, sl_idx)

        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn + 1e-8)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        out = np.stack([img, img, img], axis=0).astype(np.float32)
        out = (out - 0.5) / 0.5

        return (
            torch.tensor(out, dtype=torch.float32),
            torch.tensor(int(s["quality"]), dtype=torch.long),
            torch.tensor(int(s["plane"]), dtype=torch.long),
        )


# ── TTA / Prediction collection ──────────────────────────────
def apply_tta_tensor(imgs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "orig":
        return imgs
    if mode == "hflip":
        return torch.flip(imgs, dims=[3])
    if mode in ("rot5", "rot-5"):
        if not HAS_TORCHVISION_TTA:
            raise RuntimeError("torchvision rotation TTA unavailable. Use KMAR_TTA_MODES=orig,hflip")
        angle = 5.0 if mode == "rot5" else -5.0
        return TF.rotate(imgs, angle=angle, interpolation=InterpolationMode.BILINEAR)
    raise ValueError(f"Unsupported TTA mode: {mode}")


def collect_probs(
    model: nn.Module,
    loader: DataLoader,
    desc: str,
    tta_modes: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_probs, all_q_true, all_p_true, all_p_pred = [], [], [], []
    autocast_enabled = DEVICE.type == "cuda"

    with torch.inference_mode():
        for i, (imgs, q_labels, p_labels) in enumerate(tqdm(loader, desc=desc, leave=True)):
            imgs = imgs.to(DEVICE, non_blocking=True)
            prob_sum = None
            plane_logits_sum = None

            # One TTA forward at a time. This avoids large concatenated batches and is 4GB-GPU friendly.
            for mode in tta_modes:
                x_aug = apply_tta_tensor(imgs, mode)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                    q_out, p_out = model(x_aug)
                probs = torch.softmax(q_out, dim=1)
                prob_sum = probs if prob_sum is None else prob_sum + probs
                plane_logits_sum = p_out if plane_logits_sum is None else plane_logits_sum + p_out
                del x_aug, q_out, p_out, probs

            avg_probs = (prob_sum / float(len(tta_modes))).detach().cpu().numpy()
            p_pred = plane_logits_sum.argmax(1).detach().cpu().numpy()

            all_probs.append(avg_probs)
            all_q_true.append(q_labels.numpy())
            all_p_true.append(p_labels.numpy())
            all_p_pred.append(p_pred)

            del imgs, prob_sum, plane_logits_sum
            if DEVICE.type == "cuda" and i % 25 == 0:
                torch.cuda.empty_cache()
            if i % 50 == 0:
                gc.collect()

    return (
        np.vstack(all_probs),
        np.concatenate(all_q_true),
        np.concatenate(all_p_true),
        np.concatenate(all_p_pred),
    )


def model_cache_key(model_path: str, dataset_name: str, tta_modes: List[str], extra: str = "") -> str:
    mtime = int(os.path.getmtime(model_path)) if os.path.exists(model_path) else 0
    raw = f"{os.path.basename(model_path)}|{mtime}|{dataset_name}|{','.join(tta_modes)}|{extra}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"{dataset_name}_{os.path.basename(model_path).replace('.pt','')}_{'_'.join(tta_modes)}_{digest}.npz"


def cached_collect(
    model: nn.Module,
    model_path: str,
    dataset_name: str,
    loader: DataLoader,
    tta_modes: List[str],
    desc: str,
    extra: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_file = os.path.join(PRED_CACHE_DIR, model_cache_key(model_path, dataset_name, tta_modes, extra))
    if os.path.exists(cache_file) and not REBUILD_CACHE:
        print(f"⚡ Loading prediction cache: {cache_file}")
        data = np.load(cache_file)
        return data["probs"], data["q_true"], data["p_true"], data["p_pred"]

    probs, q_true, p_true, p_pred = collect_probs(model, loader, desc=desc, tta_modes=tta_modes)
    np.savez_compressed(cache_file, probs=probs, q_true=q_true, p_true=p_true, p_pred=p_pred)
    print(f"💾 Saved prediction cache: {cache_file}")
    return probs, q_true, p_true, p_pred


# ── Threshold / metric utilities ─────────────────────────────
def threshold_values(lo: float, hi: float, step: float) -> np.ndarray:
    return np.round(np.arange(lo, hi + 1e-9, step), 4)


def apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return np.argmax(probabilities / thresholds, axis=1)


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    cm = np.bincount(3 * y_true + y_pred, minlength=9).reshape(3, 3).astype(np.float64)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    tp = np.diag(cm)

    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

    total = support.sum()
    acc = tp.sum() / total if total > 0 else 0.0
    weighted_f1 = (f1 * support).sum() / total if total > 0 else 0.0
    macro_f1 = f1.mean()
    return {"accuracy": float(acc), "weighted_f1": float(weighted_f1), "macro_f1": float(macro_f1)}


@dataclass
class TuneResult:
    name: str
    thresholds: np.ndarray
    metric_name: str
    metric_value: float
    accuracy: float
    weighted_f1: float
    macro_f1: float


def tune_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    name: str,
    metric_name: str = "weighted_f1",
    step: float = FINE_STEP,
    ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (GOOD_RANGE, MOD_RANGE, BAD_RANGE),
) -> TuneResult:
    good_vals = threshold_values(ranges[0][0], ranges[0][1], step)
    mod_vals = threshold_values(ranges[1][0], ranges[1][1], step)
    bad_vals = threshold_values(ranges[2][0], ranges[2][1], step)
    total = len(good_vals) * len(mod_vals) * len(bad_vals)

    best = TuneResult(
        name=name,
        thresholds=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        metric_name=metric_name,
        metric_value=-1.0,
        accuracy=-1.0,
        weighted_f1=-1.0,
        macro_f1=-1.0,
    )

    checked = 0
    print(f"🔎 Tuning {name} on validation | metric={metric_name} | step={step:.3f} | combos={total:,}")
    for g in good_vals:
        for m in mod_vals:
            # inner loop small; keep simple and stable
            for b in bad_vals:
                th = np.array([g, m, b], dtype=np.float32)
                pred = apply_thresholds(probs, th)
                metrics = fast_metrics(y_true, pred)
                score = metrics[metric_name]

                # Tie-breaker: prefer higher accuracy, then higher weighted F1.
                if (
                    score > best.metric_value
                    or (abs(score - best.metric_value) < 1e-12 and metrics["accuracy"] > best.accuracy)
                    or (abs(score - best.metric_value) < 1e-12 and abs(metrics["accuracy"] - best.accuracy) < 1e-12 and metrics["weighted_f1"] > best.weighted_f1)
                ):
                    best = TuneResult(
                        name=name,
                        thresholds=th.copy(),
                        metric_name=metric_name,
                        metric_value=float(score),
                        accuracy=metrics["accuracy"],
                        weighted_f1=metrics["weighted_f1"],
                        macro_f1=metrics["macro_f1"],
                    )

                checked += 1
                if checked % 10000 == 0:
                    print(
                        f"   checked {checked:,}/{total:,} | "
                        f"best {metric_name}={best.metric_value:.4f} acc={best.accuracy:.4f} "
                        f"th={best.thresholds.tolist()}"
                    )

    print(
        f"✅ Best {name}: th={best.thresholds.tolist()} | "
        f"acc={best.accuracy:.4f} | weighted-F1={best.weighted_f1:.4f} | macro-F1={best.macro_f1:.4f}"
    )
    return best


def evaluate_probs(probs: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray) -> Dict[str, object]:
    pred = apply_thresholds(probs, thresholds)
    metrics = fast_metrics(y_true, pred)
    report = classification_report(y_true, pred, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    return {"pred": pred, "metrics": metrics, "report": report, "cm": cm}


def format_eval_block(title: str, eval_obj: Dict[str, object]) -> str:
    m = eval_obj["metrics"]
    cm = eval_obj["cm"]
    return "\n".join([
        "=" * 70,
        title,
        "=" * 70,
        f"Quality Accuracy : {m['accuracy']*100:.2f}%",
        f"Weighted F1      : {m['weighted_f1']:.4f}",
        f"Macro F1         : {m['macro_f1']:.4f}",
        "",
        str(eval_obj["report"]),
        "Confusion matrix [Good, Moderate, Bad]:",
        str(cm),
        "",
    ])


# ── Ensemble search ──────────────────────────────────────────
def choose_best_ensemble_weight(
    p3_val: np.ndarray,
    p5_val: np.ndarray,
    y_val: np.ndarray,
    metric_name: str = "weighted_f1",
) -> Tuple[float, TuneResult]:
    weights = np.round(np.arange(0.0, 1.0 + 1e-9, ENSEMBLE_WEIGHT_STEP), 4)
    # Coarse ranges first to avoid excessive runtime.
    coarse_ranges = ((0.10, 0.25), (0.35, 0.60), (0.55, 0.85))
    best_weight = 0.0
    best_tune: Optional[TuneResult] = None

    print(f"🔀 Scanning Phase3/Phase5 ensemble weights | step={ENSEMBLE_WEIGHT_STEP}")
    print("   Formula: ensemble_probs = w*Phase3 + (1-w)*Phase5")

    for w in weights:
        ens_val = (w * p3_val) + ((1.0 - w) * p5_val)
        tune = tune_thresholds(
            ens_val,
            y_val,
            name=f"Ensemble coarse w_phase3={w:.2f}",
            metric_name=metric_name,
            step=COARSE_STEP,
            ranges=coarse_ranges,
        )
        if best_tune is None or tune.metric_value > best_tune.metric_value or (
            abs(tune.metric_value - best_tune.metric_value) < 1e-12 and tune.accuracy > best_tune.accuracy
        ):
            best_weight = float(w)
            best_tune = tune

    assert best_tune is not None
    print(f"✅ Best coarse ensemble weight: w_phase3={best_weight:.2f}")

    # Fine-tune thresholds only for the best ensemble weight.
    ens_val = (best_weight * p3_val) + ((1.0 - best_weight) * p5_val)
    fine_tune = tune_thresholds(
        ens_val,
        y_val,
        name=f"Ensemble FINE w_phase3={best_weight:.2f}",
        metric_name=metric_name,
        step=FINE_STEP,
        ranges=(GOOD_RANGE, MOD_RANGE, BAD_RANGE),
    )
    return best_weight, fine_tune


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("KMAR-50K TTA + Fine Threshold + Ensemble Evaluator")
    print("=" * 70)
    print(f"Device       : {DEVICE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Workers      : {NUM_WORKERS}")
    print(f"TTA modes    : {TTA_MODES}")
    print(f"Fine step    : {FINE_STEP}")
    print(f"Cache rebuild: {REBUILD_CACHE}")
    print("=" * 70)

    if not os.path.exists(LABELED_JSON):
        raise FileNotFoundError(f"Missing labeled JSON: {LABELED_JSON}")
    if not os.path.exists(TEST_JSON):
        raise FileNotFoundError(f"Missing locked test JSON: {TEST_JSON}")

    with open(LABELED_JSON, "r") as f:
        all_samples = json.load(f)
    train_s, temp_s = train_test_split(all_samples, test_size=0.30, random_state=42)
    val_s, reconstructed_test_s = train_test_split(temp_s, test_size=0.33, random_state=42)

    with open(TEST_JSON, "r") as f:
        test_s = json.load(f)

    print(f"📂 Reconstructed split → Train: {len(train_s):,} | Val: {len(val_s):,} | Test: {len(reconstructed_test_s):,}")
    print(f"📂 Locked test JSON    → Test: {len(test_s):,}")
    print("   Validation is used for selection. Locked test is used only after selection.")

    # Load Phase 5 model and predictions.
    phase5_model, phase5_ckpt = load_model(PHASE5_MODEL_PATH, "Phase 5")

    # Prefer val_cache for Phase 5 validation speed. It must match val_s ordering.
    p5_val_loader = DataLoader(
        Phase5DiskDataset(VAL_CACHE_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    p5_test_loader = DataLoader(
        Phase5RawMultiSliceDataset(test_s),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    p5_val_probs_tta, p5_val_true, p5_val_plane_true, p5_val_plane_pred = cached_collect(
        phase5_model,
        PHASE5_MODEL_PATH,
        dataset_name="phase5_val",
        loader=p5_val_loader,
        tta_modes=TTA_MODES,
        desc="Phase 5 validation inference",
        extra="val_cache",
    )
    p5_test_probs_tta, p5_test_true, p5_test_plane_true, p5_test_plane_pred = cached_collect(
        phase5_model,
        PHASE5_MODEL_PATH,
        dataset_name="phase5_test",
        loader=p5_test_loader,
        tta_modes=TTA_MODES,
        desc="Phase 5 locked test inference",
        extra="raw_test",
    )

    # Also get Phase 5 without TTA if TTA modes are not just orig.
    if TTA_MODES == ["orig"]:
        p5_val_probs_orig = p5_val_probs_tta
        p5_test_probs_orig = p5_test_probs_tta
    else:
        p5_val_probs_orig, p5_val_true_orig, _, _ = cached_collect(
            phase5_model,
            PHASE5_MODEL_PATH,
            dataset_name="phase5_val",
            loader=p5_val_loader,
            tta_modes=["orig"],
            desc="Phase 5 validation inference no-TTA",
            extra="val_cache_orig",
        )
        p5_test_probs_orig, p5_test_true_orig, _, _ = cached_collect(
            phase5_model,
            PHASE5_MODEL_PATH,
            dataset_name="phase5_test",
            loader=p5_test_loader,
            tta_modes=["orig"],
            desc="Phase 5 locked test inference no-TTA",
            extra="raw_test_orig",
        )
        if not np.array_equal(p5_val_true, p5_val_true_orig):
            raise RuntimeError("Phase 5 validation labels mismatch between TTA and no-TTA caches.")
        if not np.array_equal(p5_test_true, p5_test_true_orig):
            raise RuntimeError("Phase 5 test labels mismatch between TTA and no-TTA caches.")

    del phase5_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    candidates = []

    # Candidate 1: Phase 5 no-TTA fine thresholds.
    for metric_name in ["weighted_f1", "accuracy"]:
        tune = tune_thresholds(p5_val_probs_orig, p5_val_true, f"Phase5 no-TTA fine ({metric_name})", metric_name=metric_name)
        test_eval = evaluate_probs(p5_test_probs_orig, p5_test_true, tune.thresholds)
        candidates.append({
            "strategy": f"phase5_no_tta_{metric_name}",
            "kind": "phase5",
            "phase3_weight": None,
            "tta_modes": ["orig"],
            "tune": tune,
            "test_eval": test_eval,
            "test_probs": p5_test_probs_orig,
        })

    # Candidate 2: Phase 5 TTA fine thresholds.
    if TTA_MODES != ["orig"]:
        for metric_name in ["weighted_f1", "accuracy"]:
            tune = tune_thresholds(p5_val_probs_tta, p5_val_true, f"Phase5 TTA fine ({metric_name})", metric_name=metric_name)
            test_eval = evaluate_probs(p5_test_probs_tta, p5_test_true, tune.thresholds)
            candidates.append({
                "strategy": f"phase5_tta_{metric_name}",
                "kind": "phase5_tta",
                "phase3_weight": None,
                "tta_modes": TTA_MODES,
                "tune": tune,
                "test_eval": test_eval,
                "test_probs": p5_test_probs_tta,
            })

    # Optional Phase 3 + Phase 5 ensemble.
    phase3_available = os.path.exists(PHASE3_MODEL_PATH)
    if phase3_available:
        phase3_model, phase3_ckpt = load_model(PHASE3_MODEL_PATH, "Phase 3 SSIM")

        p3_val_loader = DataLoader(
            Phase3SingleSliceDataset(val_s),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        p3_test_loader = DataLoader(
            Phase3SingleSliceDataset(test_s),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        p3_val_probs, p3_val_true, _, _ = cached_collect(
            phase3_model,
            PHASE3_MODEL_PATH,
            dataset_name="phase3_val",
            loader=p3_val_loader,
            tta_modes=["orig"],
            desc="Phase 3 validation inference",
            extra="single_slice",
        )
        p3_test_probs, p3_test_true, _, _ = cached_collect(
            phase3_model,
            PHASE3_MODEL_PATH,
            dataset_name="phase3_test",
            loader=p3_test_loader,
            tta_modes=["orig"],
            desc="Phase 3 locked test inference",
            extra="single_slice",
        )

        del phase3_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if not np.array_equal(p3_val_true, p5_val_true):
            print("⚠️ Validation label order mismatch between Phase 3 reconstructed val_s and Phase 5 val_cache.")
            print("   Skipping ensemble to avoid invalid probability averaging.")
        elif not np.array_equal(p3_test_true, p5_test_true):
            print("⚠️ Test label order mismatch between Phase 3 and Phase 5 loaders. Skipping ensemble.")
        else:
            # Phase 3 alone is also useful as a sanity check.
            for metric_name in ["weighted_f1", "accuracy"]:
                tune = tune_thresholds(p3_val_probs, p3_val_true, f"Phase3 alone fine ({metric_name})", metric_name=metric_name)
                test_eval = evaluate_probs(p3_test_probs, p3_test_true, tune.thresholds)
                candidates.append({
                    "strategy": f"phase3_alone_{metric_name}",
                    "kind": "phase3",
                    "phase3_weight": 1.0,
                    "tta_modes": ["orig"],
                    "tune": tune,
                    "test_eval": test_eval,
                    "test_probs": p3_test_probs,
                })

            # Ensemble with Phase 5 TTA probabilities if available; otherwise Phase 5 orig.
            p5_val_for_ensemble = p5_val_probs_tta if TTA_MODES != ["orig"] else p5_val_probs_orig
            p5_test_for_ensemble = p5_test_probs_tta if TTA_MODES != ["orig"] else p5_test_probs_orig
            ens_tta_modes = TTA_MODES if TTA_MODES != ["orig"] else ["orig"]

            for metric_name in ["weighted_f1", "accuracy"]:
                w3, ens_tune = choose_best_ensemble_weight(
                    p3_val_probs,
                    p5_val_for_ensemble,
                    p5_val_true,
                    metric_name=metric_name,
                )
                ens_test_probs = (w3 * p3_test_probs) + ((1.0 - w3) * p5_test_for_ensemble)
                test_eval = evaluate_probs(ens_test_probs, p5_test_true, ens_tune.thresholds)
                candidates.append({
                    "strategy": f"ensemble_phase3_phase5_{metric_name}",
                    "kind": "ensemble_phase3_phase5",
                    "phase3_weight": w3,
                    "phase5_weight": 1.0 - w3,
                    "tta_modes": ens_tta_modes,
                    "tune": ens_tune,
                    "test_eval": test_eval,
                    "test_probs": ens_test_probs,
                })
    else:
        print(f"⚠️ Phase 3 checkpoint not found, skipping ensemble: {PHASE3_MODEL_PATH}")

    # Pick winner by validation weighted-F1 first, then validation accuracy, then test accuracy only for display.
    # This preserves validation-only selection. It does not pick by test.
    def val_selection_key(c):
        t = c["tune"]
        return (t.weighted_f1, t.accuracy, t.macro_f1)

    best = max(candidates, key=val_selection_key)

    # Save best threshold and strategy metadata.
    best_tune: TuneResult = best["tune"]
    best_meta = {
        "strategy": best["strategy"],
        "kind": best["kind"],
        "thresholds": {
            "Good": float(best_tune.thresholds[0]),
            "Moderate": float(best_tune.thresholds[1]),
            "Bad": float(best_tune.thresholds[2]),
        },
        "selection_metric": best_tune.metric_name,
        "validation_accuracy": float(best_tune.accuracy),
        "validation_weighted_f1": float(best_tune.weighted_f1),
        "validation_macro_f1": float(best_tune.macro_f1),
        "test_accuracy": float(best["test_eval"]["metrics"]["accuracy"]),
        "test_weighted_f1": float(best["test_eval"]["metrics"]["weighted_f1"]),
        "test_macro_f1": float(best["test_eval"]["metrics"]["macro_f1"]),
        "phase3_weight": None if best.get("phase3_weight") is None else float(best.get("phase3_weight")),
        "phase5_weight": None if best.get("phase5_weight") is None else float(best.get("phase5_weight")),
        "tta_modes": best.get("tta_modes", []),
        "phase5_model": PHASE5_MODEL_PATH,
        "phase3_model": PHASE3_MODEL_PATH if phase3_available else None,
        "note": "Winner selected by validation metrics only. Locked test metrics are reported after selection.",
    }

    with open(BEST_JSON_PATH, "w") as f:
        json.dump(best_meta, f, indent=2)
    with open(BEST_THRESH_JSON_PATH, "w") as f:
        json.dump(best_meta["thresholds"], f, indent=2)
    with open(BEST_PKL_PATH, "wb") as f:
        pickle.dump({
            "thresholds": best_tune.thresholds,
            "strategy": best_meta,
        }, f)

    # Report all candidates.
    lines = []
    lines.append("=" * 70)
    lines.append("KMAR-50K TTA + FINE THRESHOLD + ENSEMBLE EVALUATION")
    lines.append("=" * 70)
    lines.append(f"Device              : {DEVICE}")
    lines.append(f"Batch size          : {BATCH_SIZE}")
    lines.append(f"TTA modes           : {TTA_MODES}")
    lines.append(f"Phase 5 checkpoint  : {PHASE5_MODEL_PATH}")
    lines.append(f"Phase 3 checkpoint  : {PHASE3_MODEL_PATH if phase3_available else 'NOT FOUND'}")
    lines.append(f"Validation samples  : {len(p5_val_true):,}")
    lines.append(f"Locked test samples : {len(p5_test_true):,}")
    lines.append("Selection rule      : validation weighted-F1, then validation accuracy")
    lines.append("")
    lines.append("CANDIDATE SUMMARY")
    lines.append("-" * 70)

    for c in sorted(candidates, key=val_selection_key, reverse=True):
        t = c["tune"]
        tm = c["test_eval"]["metrics"]
        lines.append(
            f"{c['strategy']:<36} | "
            f"VAL acc={t.accuracy*100:6.2f}% wf1={t.weighted_f1:.4f} | "
            f"TEST acc={tm['accuracy']*100:6.2f}% wf1={tm['weighted_f1']:.4f} | "
            f"th=[{t.thresholds[0]:.3f},{t.thresholds[1]:.3f},{t.thresholds[2]:.3f}] | "
            f"w3={c.get('phase3_weight')}"
        )

    lines.append("")
    lines.append("BEST SELECTED STRATEGY")
    lines.append("-" * 70)
    lines.append(json.dumps(best_meta, indent=2))
    lines.append("")
    lines.append(format_eval_block(f"LOCKED TEST — {best['strategy']}", best["test_eval"]))

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n✅ Saved report        → {REPORT_PATH}")
    print(f"✅ Saved best strategy → {BEST_JSON_PATH}")
    print(f"✅ Saved thresholds    → {BEST_PKL_PATH}")
    print(f"✅ Saved thresholds JS → {BEST_THRESH_JSON_PATH}")


if __name__ == "__main__":
    main()
