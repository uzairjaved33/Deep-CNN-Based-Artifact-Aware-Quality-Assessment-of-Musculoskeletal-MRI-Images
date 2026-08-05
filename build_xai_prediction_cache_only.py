"""
KMAR-50K XAI Prediction Cache Builder
=====================================

Purpose:
    Rebuild only the locked-test prediction caches needed by
    generate_xai_gradcam_final_ensemble.py.

Why:
    apply_thresholds_tta_ensemble.py needs val_cache/index.json.
    If val_cache was deleted, XAI cache generation can still be done directly
    from checkpoints/test_samples_ssim.json and the original .nii.gz files.

Creates:
    checkpoints/prediction_cache/phase3_test_best_model_ssim_orig_<hash>.npz
    checkpoints/prediction_cache/phase5_test_best_model_phase5_orig_hflip_<hash>.npz

Run:
    cd /d D:\\KMAR-50K\\KMAR-50K
    python build_xai_prediction_cache_only.py

Optional:
    set KMAR_XAI_CACHE_BATCH=2
    set KMAR_XAI_CACHE_DEVICE=cuda
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm


BASE = Path(os.environ.get("KMAR_BASE", r"D:\KMAR-50K\KMAR-50K"))
CKPT_DIR = BASE / "checkpoints"
PRED_CACHE_DIR = CKPT_DIR / "prediction_cache"
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEST_JSON = CKPT_DIR / "test_samples_ssim.json"
PHASE3_MODEL = CKPT_DIR / "best_model_ssim.pt"
PHASE5_MODEL = CKPT_DIR / "best_model_phase5.pt"

IMG_SIZE = 224
BATCH_SIZE = int(os.environ.get("KMAR_XAI_CACHE_BATCH", "2"))
NUM_WORKERS = 0
DEVICE_NAME = os.environ.get("KMAR_XAI_CACHE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE_NAME if DEVICE_NAME == "cpu" or torch.cuda.is_available() else "cpu")


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


def load_model(path: Path, label: str) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    print(f"Loading {label}: {path}")
    model = KMARMultiTask()
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def read_slice(proxy, idx: int) -> np.ndarray:
    idx = int(idx)
    data = proxy.dataobj
    if len(proxy.shape) == 4:
        return np.asarray(data[:, :, idx, 0], dtype=np.float32)
    return np.asarray(data[:, :, idx], dtype=np.float32)


def normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(img.min())
    mx = float(img.max())
    return (img - mn) / (mx - mn + 1e-8)


class TestNiftiDataset(Dataset):
    def __init__(self, samples: List[dict], mode: str, hflip: bool = False):
        assert mode in {"phase3", "phase5"}
        self.samples = samples
        self.mode = mode
        self.hflip = hflip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        path = s["path"]
        slice_idx = int(s["slice"])

        proxy = nib.load(path)
        n_slices = int(proxy.shape[2])
        slice_idx = max(0, min(slice_idx, n_slices - 1))

        curr = normalize01(read_slice(proxy, slice_idx))
        curr = cv2.resize(curr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        if self.mode == "phase3":
            arr = np.stack([curr, curr, curr], axis=0).astype(np.float32)
        else:
            prev = normalize01(read_slice(proxy, max(0, slice_idx - 1)))
            nxt = normalize01(read_slice(proxy, min(n_slices - 1, slice_idx + 1)))
            prev = cv2.resize(prev, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            nxt = cv2.resize(nxt, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            arr = np.stack([prev, curr, nxt], axis=0).astype(np.float32)

        if self.hflip:
            arr = arr[:, :, ::-1].copy()

        arr = (arr - 0.5) / 0.5

        q = int(s.get("quality", s.get("quality_label", -1)))
        p = int(s.get("plane", s.get("plane_label", -1)))

        return (
            torch.tensor(arr, dtype=torch.float32),
            torch.tensor(q, dtype=torch.long),
            torch.tensor(p, dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


@torch.no_grad()
def infer_probs(model: nn.Module, dataset: Dataset, label: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    probs_all = []
    q_true_all = []
    p_true_all = []
    indices_all = []

    use_amp = DEVICE.type == "cuda"

    for x, q, p, idx in tqdm(loader, desc=label):
        x = x.to(DEVICE, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                q_logits, _p_logits = model(x)
        else:
            q_logits, _p_logits = model(x)

        probs = torch.softmax(q_logits.float(), dim=1)

        probs_all.append(probs.cpu().numpy().astype(np.float32))
        q_true_all.append(q.numpy().astype(np.int64))
        p_true_all.append(p.numpy().astype(np.int64))
        indices_all.append(idx.numpy().astype(np.int64))

    return (
        np.concatenate(probs_all, axis=0),
        np.concatenate(q_true_all, axis=0),
        np.concatenate(p_true_all, axis=0),
        np.concatenate(indices_all, axis=0),
    )


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def save_npz(path: Path, probs: np.ndarray, q_true: np.ndarray, p_true: np.ndarray, indices: np.ndarray):
    np.savez_compressed(
        str(path),
        probs=probs.astype(np.float32),
        q_true=q_true.astype(np.int64),
        p_true=p_true.astype(np.int64),
        indices=indices.astype(np.int64),
    )
    print(f"Saved: {path}")
    print(f"  probs shape: {probs.shape}")


def main():
    print("=" * 80)
    print("KMAR-50K XAI Prediction Cache Builder")
    print("=" * 80)
    print(f"BASE       : {BASE}")
    print(f"DEVICE     : {DEVICE}")
    print(f"BATCH_SIZE : {BATCH_SIZE}")
    print(f"TEST_JSON  : {TEST_JSON}")

    if not TEST_JSON.exists():
        raise FileNotFoundError(TEST_JSON)
    if not PHASE3_MODEL.exists():
        raise FileNotFoundError(PHASE3_MODEL)
    if not PHASE5_MODEL.exists():
        raise FileNotFoundError(PHASE5_MODEL)

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded locked-test samples: {len(samples)}")

    phase3_out = PRED_CACHE_DIR / f"phase3_test_best_model_ssim_orig_{short_hash(str(PHASE3_MODEL)+str(TEST_JSON))}.npz"
    phase5_out = PRED_CACHE_DIR / f"phase5_test_best_model_phase5_orig_hflip_{short_hash(str(PHASE5_MODEL)+str(TEST_JSON)+'orig_hflip')}.npz"

    if phase3_out.exists() and phase5_out.exists():
        print("Both required XAI caches already exist:")
        print(f"  {phase3_out}")
        print(f"  {phase5_out}")
        return

    phase3 = load_model(PHASE3_MODEL, "Phase 3 model")
    probs3, q3, p3, idx3 = infer_probs(
        phase3,
        TestNiftiDataset(samples, mode="phase3", hflip=False),
        "Phase 3 test orig",
    )
    save_npz(phase3_out, probs3, q3, p3, idx3)
    del phase3
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    phase5 = load_model(PHASE5_MODEL, "Phase 5 model")

    probs5_orig, q5, p5, idx5 = infer_probs(
        phase5,
        TestNiftiDataset(samples, mode="phase5", hflip=False),
        "Phase 5 test orig",
    )

    probs5_hflip, q5b, p5b, idx5b = infer_probs(
        phase5,
        TestNiftiDataset(samples, mode="phase5", hflip=True),
        "Phase 5 test hflip",
    )

    if not np.array_equal(idx5, idx5b):
        raise RuntimeError("Phase 5 orig/hflip index mismatch.")
    if not np.array_equal(q5, q5b):
        raise RuntimeError("Phase 5 orig/hflip q_true mismatch.")
    if not np.array_equal(p5, p5b):
        raise RuntimeError("Phase 5 orig/hflip p_true mismatch.")

    probs5 = ((probs5_orig + probs5_hflip) / 2.0).astype(np.float32)
    save_npz(phase5_out, probs5, q5, p5, idx5)

    print("\nDone. Required XAI prediction caches are ready.")
    print(f"Prediction cache folder: {PRED_CACHE_DIR}")
    print("\nNow run:")
    print("    set KMAR_XAI_PER_CATEGORY=8")
    print("    set KMAR_XAI_MAX_TOTAL=80")
    print("    set KMAR_XAI_DEVICE=cuda")
    print("    python generate_xai_gradcam_final_ensemble.py")


if __name__ == "__main__":
    main()
