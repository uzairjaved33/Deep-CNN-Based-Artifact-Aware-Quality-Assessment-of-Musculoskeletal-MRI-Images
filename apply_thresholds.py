"""
apply_thresholds.py — Phase 5 validation-calibrated threshold evaluator

Purpose:
- Loads UPDATED Phase 5 model: checkpoints/best_model_phase5.pt
- Tunes class thresholds on VALIDATION ONLY using val_cache/index.json
- Applies the selected thresholds to the LOCKED test set
- Does NOT tune on test. Test is used only after thresholds are frozen.
- Safe for 4GB CUDA GPUs: batch_size=4, num_workers=0, fp16 inference.

Outputs:
- checkpoints/optimal_thresholds_phase5.pkl
- checkpoints/optimal_thresholds_phase5.json
- checkpoints/optimal_thresholds.pkl      # updated compatibility copy
- checkpoints/thresholded_evaluation_report_phase5.txt
"""

import os
import json
import gc
import cv2
import pickle
import torch
import timm
import nibabel as nib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ── Config ───────────────────────────────────────────────────
BASE = r"D:\KMAR-50K\KMAR-50K"
CKPT_DIR = os.path.join(BASE, "checkpoints")
VAL_CACHE_DIR = os.path.join(BASE, "val_cache")
MODEL_PATH = os.path.join(CKPT_DIR, "best_model_phase5.pt")
TEST_JSON = os.path.join(CKPT_DIR, "test_samples_ssim.json")

THRESHOLD_PKL = os.path.join(CKPT_DIR, "optimal_thresholds_phase5.pkl")
THRESHOLD_JSON = os.path.join(CKPT_DIR, "optimal_thresholds_phase5.json")
COMPAT_PKL = os.path.join(CKPT_DIR, "optimal_thresholds.pkl")
REPORT_PATH = os.path.join(CKPT_DIR, "thresholded_evaluation_report_phase5.txt")

IMG_SIZE = 224
BATCH_SIZE = int(os.environ.get("KMAR_EVAL_BATCH_SIZE", "4"))  # safe for 4GB VRAM
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
CLASS_NAMES = ["Good", "Moderate", "Bad"]
PLANE_NAMES = ["Sagittal", "Coronal", "Transection"]

# Threshold grid. 0.05 is fast and stable; use KMAR_THRESHOLD_STEP=0.02 for a finer pass.
THRESHOLD_STEP = float(os.environ.get("KMAR_THRESHOLD_STEP", "0.05"))
GRID_MIN = float(os.environ.get("KMAR_THRESHOLD_MIN", "0.10"))
GRID_MAX = float(os.environ.get("KMAR_THRESHOLD_MAX", "0.90"))

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ── Model: must match Phase 5 architecture ───────────────────
class KMARMultiTask(nn.Module):
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


# ── Validation dataset: uses already preprocessed val_cache ───
class DiskDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        index_path = os.path.join(cache_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Validation cache index not found: {index_path}")
        with open(index_path, "r") as f:
            self.samples = json.load(f)
        print(f"📂 Loaded validation cache: {len(self.samples):,} samples from {cache_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = np.load(os.path.join(self.cache_dir, s["file"]), allow_pickle=True).item()
        # data['img'] is already 3x224x224 and normalized exactly as Phase 5 training used.
        return (
            torch.tensor(data["img"], dtype=torch.float32),
            torch.tensor(int(data["quality"]), dtype=torch.long),
            torch.tensor(int(data["plane"]), dtype=torch.long),
        )


# ── Test dataset: raw memory-safe multi-slice loading ─────────
class RawMultiSliceDataset(Dataset):
    def __init__(self, samples, img_size=224):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_slice(proxy, idx):
        idx = int(idx)
        data = proxy.dataobj
        if len(proxy.shape) == 4:
            return np.asarray(data[:, :, idx, 0], dtype=np.float32)
        return np.asarray(data[:, :, idx], dtype=np.float32)

    def __getitem__(self, idx):
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


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = KMARMultiTask().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def collect_probs(model, loader, desc):
    all_probs = []
    all_q_true = []
    all_p_true = []
    all_p_pred = []

    autocast_enabled = DEVICE.type == "cuda"
    with torch.inference_mode():
        for i, (imgs, q_labels, p_labels) in enumerate(tqdm(loader, desc=desc, leave=True)):
            imgs = imgs.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                q_out, p_out = model(imgs)

            probs = torch.softmax(q_out, dim=1).detach().cpu().numpy()
            p_pred = p_out.argmax(1).detach().cpu().numpy()

            all_probs.append(probs)
            all_q_true.append(q_labels.numpy())
            all_p_true.append(p_labels.numpy())
            all_p_pred.append(p_pred)

            del imgs, q_out, p_out
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


def apply_thresholds(probabilities, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    adjusted = probabilities / thresholds.reshape(1, -1)
    return np.argmax(adjusted, axis=1)


def tune_thresholds_on_validation(val_probs, val_true):
    values = np.round(np.arange(GRID_MIN, GRID_MAX + 1e-9, THRESHOLD_STEP), 4)
    best = {
        "thresholds": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "weighted_f1": -1.0,
        "accuracy": -1.0,
    }

    print(f"🔎 Tuning thresholds on validation only | grid={GRID_MIN:.2f}..{GRID_MAX:.2f}, step={THRESHOLD_STEP:.2f}")
    total = len(values) ** 3
    checked = 0

    for tg in values:
        for tm in values:
            for tb in values:
                th = np.array([tg, tm, tb], dtype=np.float32)
                pred = apply_thresholds(val_probs, th)
                wf1 = f1_score(val_true, pred, average="weighted", zero_division=0)
                acc = accuracy_score(val_true, pred)

                # Primary: weighted F1. Tie-break: accuracy.
                if (wf1 > best["weighted_f1"]) or (np.isclose(wf1, best["weighted_f1"]) and acc > best["accuracy"]):
                    best = {"thresholds": th.copy(), "weighted_f1": float(wf1), "accuracy": float(acc)}

                checked += 1
                if checked % 1000 == 0:
                    print(f"   checked {checked:,}/{total:,} | best weighted-F1={best['weighted_f1']:.4f} acc={best['accuracy']:.4f}", end="\r")

    print()
    return best


def metric_block(title, y_true, y_pred, p_true=None, p_pred=None):
    lines = []
    lines.append("=" * 70)
    lines.append(title)
    lines.append("=" * 70)
    lines.append(f"Quality Accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    lines.append(f"Weighted F1      : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    if p_true is not None and p_pred is not None:
        lines.append(f"Plane Accuracy   : {accuracy_score(p_true, p_pred) * 100:.2f}%")
    lines.append("")
    lines.append(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    lines.append("Confusion matrix [Good, Moderate, Bad]:")
    lines.append(str(confusion_matrix(y_true, y_pred, labels=[0, 1, 2])))
    return "\n".join(lines)


def save_thresholds(best):
    payload = {
        "model": MODEL_PATH,
        "tuned_on": VAL_CACHE_DIR,
        "metric": "weighted_f1",
        "thresholds": [float(x) for x in best["thresholds"]],
        "val_weighted_f1": float(best["weighted_f1"]),
        "val_accuracy": float(best["accuracy"]),
        "class_order": CLASS_NAMES,
    }

    with open(THRESHOLD_PKL, "wb") as f:
        pickle.dump(payload, f)
    with open(COMPAT_PKL, "wb") as f:
        pickle.dump(payload, f)
    with open(THRESHOLD_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    print(f"⚡ Device: {DEVICE}")
    print(f"📦 Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"🔄 Loading model: {MODEL_PATH}")
    model, ckpt = load_model(MODEL_PATH)

    # 1) Validation probabilities from val_cache — this is where thresholds are tuned.
    val_loader = DataLoader(
        DiskDataset(VAL_CACHE_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_probs, val_true, val_plane_true, val_plane_pred = collect_probs(model, val_loader, "Validation inference")
    val_raw_pred = val_probs.argmax(axis=1)

    best = tune_thresholds_on_validation(val_probs, val_true)
    thresholds = best["thresholds"]
    val_tuned_pred = apply_thresholds(val_probs, thresholds)
    payload = save_thresholds(best)

    print(f"\n✅ Best thresholds saved:")
    print(f"   Good={thresholds[0]:.4f} | Moderate={thresholds[1]:.4f} | Bad={thresholds[2]:.4f}")
    print(f"   {THRESHOLD_PKL}")
    print(f"   {THRESHOLD_JSON}")

    # 2) Locked test set — thresholds are already frozen here.
    print(f"\n📂 Loading locked test set: {TEST_JSON}")
    with open(TEST_JSON, "r") as f:
        test_samples = json.load(f)
    print(f"   Loaded test samples: {len(test_samples):,}")

    test_loader = DataLoader(
        RawMultiSliceDataset(test_samples, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_probs, test_true, test_plane_true, test_plane_pred = collect_probs(model, test_loader, "Locked test inference")
    test_raw_pred = test_probs.argmax(axis=1)
    test_tuned_pred = apply_thresholds(test_probs, thresholds)

    sections = []
    sections.append("PHASE 5 THRESHOLD EVALUATION")
    sections.append(f"Model: {MODEL_PATH}")
    sections.append(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown') if isinstance(ckpt, dict) else 'unknown'}")
    sections.append(f"Checkpoint best_loss: {ckpt.get('best_loss', 'unknown') if isinstance(ckpt, dict) else 'unknown'}")
    sections.append(f"Thresholds tuned on validation cache only: {VAL_CACHE_DIR}")
    sections.append(f"Selected thresholds: Good={thresholds[0]:.4f}, Moderate={thresholds[1]:.4f}, Bad={thresholds[2]:.4f}")
    sections.append(f"Validation weighted-F1 used for selection: {best['weighted_f1']:.4f}")
    sections.append(f"Validation accuracy at selection: {best['accuracy'] * 100:.2f}%")
    sections.append("")
    sections.append(metric_block("VALIDATION — RAW ARGMAX", val_true, val_raw_pred, val_plane_true, val_plane_pred))
    sections.append(metric_block("VALIDATION — THRESHOLD-TUNED", val_true, val_tuned_pred, val_plane_true, val_plane_pred))
    sections.append(metric_block("LOCKED TEST — RAW ARGMAX", test_true, test_raw_pred, test_plane_true, test_plane_pred))
    sections.append(metric_block("LOCKED TEST — VALIDATION-THRESHOLD-TUNED", test_true, test_tuned_pred, test_plane_true, test_plane_pred))

    text = "\n\n".join(sections)
    print("\n" + text)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ Saved threshold report → {REPORT_PATH}")
