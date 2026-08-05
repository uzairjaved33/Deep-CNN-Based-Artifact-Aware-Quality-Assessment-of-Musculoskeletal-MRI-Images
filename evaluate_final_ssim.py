"""
evaluate_final_ssim.py — Phase 5 raw evaluator

Purpose:
- Evaluates the UPDATED Phase 5 model: checkpoints/best_model_phase5.pt
- Uses the same multi-slice input used during recent training: [slice-1, slice, slice+1]
- Does NOT use thresholds. This is the raw argmax test result.
- Does NOT preload volumes. It reads only 3 slices per sample using nibabel dataobj.
- Safe for 4GB CUDA GPUs: batch_size=4, num_workers=0, fp16 inference.
"""

import os
import json
import gc
import cv2
import torch
import timm
import nibabel as nib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Config ───────────────────────────────────────────────────
BASE = r"D:\KMAR-50K\KMAR-50K"
CKPT_DIR = os.path.join(BASE, "checkpoints")
MODEL_PATH = os.path.join(CKPT_DIR, "best_model_phase5.pt")
TEST_JSON = os.path.join(CKPT_DIR, "test_samples_ssim.json")
REPORT_PATH = os.path.join(CKPT_DIR, "final_evaluation_report_phase5_raw.txt")

IMG_SIZE = 224
BATCH_SIZE = int(os.environ.get("KMAR_EVAL_BATCH_SIZE", "4"))  # safe for 4GB VRAM
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
CLASS_NAMES = ["Good", "Moderate", "Bad"]
PLANE_NAMES = ["Sagittal", "Coronal", "Transection"]

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


# ── Dataset: raw test set, memory-safe multi-slice loading ────
class RawMultiSliceDataset(Dataset):
    def __init__(self, samples, img_size=224):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _read_slice(proxy, idx):
        """Read one 2D slice only; supports 3D and simple 4D NIfTI."""
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

        # Same logic as Phase 5 preprocessing: [prev, current, next]
        slc = np.stack([prev_sl, curr_sl, next_sl], axis=0)

        # Per-channel min-max normalization
        for c in range(3):
            mn, mx = float(slc[c].min()), float(slc[c].max())
            slc[c] = (slc[c] - mn) / (mx - mn + 1e-8)

        # Resize each channel to 224x224
        out = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
        for c in range(3):
            out[c] = cv2.resize(slc[c], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Same final normalization used when cache was written: (x - 0.5) / 0.5
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

    # Handles DataParallel-style checkpoints if ever produced
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def predict(model, loader):
    all_q_true, all_q_pred = [], []
    all_p_true, all_p_pred = [], []

    autocast_enabled = DEVICE.type == "cuda"
    with torch.inference_mode():
        for i, (imgs, q_labels, p_labels) in enumerate(tqdm(loader, desc="Evaluating", leave=True)):
            imgs = imgs.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                q_out, p_out = model(imgs)

            q_pred = q_out.argmax(1).detach().cpu().numpy()
            p_pred = p_out.argmax(1).detach().cpu().numpy()

            all_q_true.extend(q_labels.numpy())
            all_q_pred.extend(q_pred)
            all_p_true.extend(p_labels.numpy())
            all_p_pred.extend(p_pred)

            del imgs, q_out, p_out
            if DEVICE.type == "cuda" and i % 25 == 0:
                torch.cuda.empty_cache()
            if i % 50 == 0:
                gc.collect()

    return (
        np.array(all_q_true),
        np.array(all_q_pred),
        np.array(all_p_true),
        np.array(all_p_pred),
    )


def build_report(q_true, q_pred, p_true, p_pred, ckpt_info):
    q_acc = accuracy_score(q_true, q_pred)
    p_acc = accuracy_score(p_true, p_pred)
    cm = confusion_matrix(q_true, q_pred, labels=[0, 1, 2])
    report = classification_report(q_true, q_pred, target_names=CLASS_NAMES, digits=4)

    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 5 RAW TEST EVALUATION — SSIM-LABELED LOCKED TEST SET")
    lines.append("=" * 70)
    lines.append(f"Model path      : {MODEL_PATH}")
    lines.append(f"Test JSON       : {TEST_JSON}")
    lines.append(f"Device          : {DEVICE}")
    lines.append(f"Batch size      : {BATCH_SIZE}")
    lines.append(f"Checkpoint epoch: {ckpt_info.get('epoch', 'unknown') if isinstance(ckpt_info, dict) else 'unknown'}")
    lines.append(f"Checkpoint loss : {ckpt_info.get('best_loss', 'unknown') if isinstance(ckpt_info, dict) else 'unknown'}")
    lines.append("")
    lines.append(f"Quality Accuracy: {q_acc * 100:.2f}%")
    lines.append(f"Plane Accuracy  : {p_acc * 100:.2f}%")
    lines.append("")
    lines.append("Quality classification report:")
    lines.append(report)
    lines.append("Confusion matrix [Good, Moderate, Bad]:")
    lines.append(str(cm))
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"⚡ Device: {DEVICE}")
    print(f"📦 Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"📂 Loading test samples: {TEST_JSON}")

    with open(TEST_JSON, "r") as f:
        test_samples = json.load(f)
    print(f"   Loaded test samples: {len(test_samples):,}")

    print(f"🔄 Loading model: {MODEL_PATH}")
    model, ckpt_info = load_model(MODEL_PATH)

    loader = DataLoader(
        RawMultiSliceDataset(test_samples, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    q_true, q_pred, p_true, p_pred = predict(model, loader)
    text = build_report(q_true, q_pred, p_true, p_pred, ckpt_info)

    print("\n" + text)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ Saved report → {REPORT_PATH}")
