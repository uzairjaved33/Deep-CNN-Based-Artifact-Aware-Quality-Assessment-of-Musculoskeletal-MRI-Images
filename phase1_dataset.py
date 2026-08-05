"""
KMAR-50K Phase 1 — Dataset Class + Dataloader
Multi-task: Quality (3-class) + Plane (3-class)
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────
BASE         = r'D:\KMAR-50K\KMAR-50K'
ART_DIR      = os.path.join(BASE, 'ArtifactData_part1')
GT_DIR       = os.path.join(BASE, 'GroundTruthData_part1')
TRAIN_CSV    = os.path.join(BASE, 'TrainingCohort.csv')
IMG_SIZE     = 224
BATCH_SIZE   = 4    # Safe for 4GB VRAM
NUM_WORKERS  = 2

# ── Label Maps ───────────────────────────────────────────────
PLANE_MAP = {'sagittal': 0, 'coronal': 1, 'transection': 2}

# Quality: 0=Good(GT), 1=Moderate(artifact mild), 2=Bad(artifact severe)
# We split artifact → moderate/bad via noise score threshold
NOISE_THRESHOLD = 0.00065  # tune after running compute_noise_scores()


# ── Noise Score ───────────────────────────────────────────────
def gradient_noise_score(slice_2d):
    s  = slice_2d.astype(np.float32)
    s  = (s - s.min()) / (s.max() - s.min() + 1e-8)
    Gx = np.gradient(s, axis=1)
    Gy = np.gradient(s, axis=0)
    M  = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    C  = (M.T @ M) / M.shape[0]
    return float(np.trace(C))


# ── Build Sample List ─────────────────────────────────────────
def build_samples(csv_path, art_dir, gt_dir, noise_threshold):
    df = pd.read_csv(csv_path)
    df['plane'] = df['NII_FileName'].str.extract(
        r'(sagittal|coronal|transection)', expand=False
    )

    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building samples"):
        fname      = row['NII_FileName'].strip().strip("'").replace('.nii.gz', '.0.nii.gz')
        plane_str  = row['plane']

        if pd.isna(plane_str):
            continue

        plane_label = PLANE_MAP.get(plane_str, -1)
        if plane_label == -1:
            continue

        # ── Ground Truth → Quality = 0 (Good) ────────────────
        gt_path = os.path.join(gt_dir, fname)
        if os.path.exists(gt_path):
            samples.append({
                'path':    gt_path,
                'quality': 0,
                'plane':   plane_label,
            })

        # ── Artifact → Quality = 1 or 2 ──────────────────────
        art_path = os.path.join(art_dir, fname)
        if os.path.exists(art_path):
            try:
                img   = nib.load(art_path).get_fdata()
                mid   = img.shape[2] // 2
                score = gradient_noise_score(img[:, :, mid])
                quality = 1 if score <= noise_threshold else 2
            except:
                quality = 2  # assume bad if load fails

            samples.append({
                'path':    art_path,
                'quality': quality,
                'plane':   plane_label,
            })

    return samples


# ── Dataset ───────────────────────────────────────────────────
class KMARDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        img  = nib.load(s['path']).get_fdata()
        mid  = img.shape[2] // 2
        sl   = img[:, :, mid].astype(np.float32)

        # Normalize
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)

        # Grayscale → 3 channel (224×224)
        sl = np.stack([sl, sl, sl], axis=0)          # (3, H, W)
        sl = torch.tensor(sl)

        if self.transform:
            sl = self.transform(sl)

        quality = torch.tensor(s['quality'], dtype=torch.long)
        plane   = torch.tensor(s['plane'],   dtype=torch.long)

        return sl, quality, plane


# ── Transforms ────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building sample list...")
    samples = build_samples(TRAIN_CSV, ART_DIR, GT_DIR, NOISE_THRESHOLD)
    print(f"Total samples: {len(samples)}")

    # Quality distribution
    q_counts = {0: 0, 1: 0, 2: 0}
    p_counts = {0: 0, 1: 0, 2: 0}
    for s in samples:
        q_counts[s['quality']] += 1
        p_counts[s['plane']]   += 1

    print(f"\nQuality: Good={q_counts[0]} | Moderate={q_counts[1]} | Bad={q_counts[2]}")
    print(f"Plane  : Sagittal={p_counts[0]} | Coronal={p_counts[1]} | Transection={p_counts[2]}")

    # Train / Val split
    train_s, val_s = train_test_split(samples, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(train_s)} | Val: {len(val_s)}")

    # Datasets
    train_ds = KMARDataset(train_s, transform=train_transform)
    val_ds   = KMARDataset(val_s,   transform=val_transform)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Verify one batch
    print("\nLoading one batch...")
    imgs, quality, plane = next(iter(train_dl))
    print(f"Image shape  : {imgs.shape}")
    print(f"Quality batch: {quality}")
    print(f"Plane batch  : {plane}")
    print("\nPhase 1 DONE — Dataloader verified OK")