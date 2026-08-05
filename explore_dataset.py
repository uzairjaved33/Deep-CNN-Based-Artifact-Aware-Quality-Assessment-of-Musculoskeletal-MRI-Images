"""
KMAR-50K Dataset Explorer
Run this script to fully understand your dataset before modeling.
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = r'D:\KMAR-50K\KMAR-50K'

FOLDERS = {
    'artifact':    os.path.join(BASE, 'ArtifactData_part1'),
    'groundtruth': os.path.join(BASE, 'GroundTruthData_part1'),
    'test_gt':     os.path.join(BASE, 'Testing _GroundTruthData'),
}

# ── 1. CSV Summary ────────────────────────────────────────────────────────────
print("=" * 60)
print("TRAINING CSV")
print("=" * 60)
train_df = pd.read_csv(os.path.join(BASE, 'TrainingCohort.csv'))
print(f"Shape       : {train_df.shape}")
print(f"Columns     : {train_df.columns.tolist()}")

# Extract plane from filename
train_df['plane'] = train_df['NII_FileName'].str.extract(r'(sagittal|coronal|transection)', expand=False)
print(f"\nPlane counts:\n{train_df['plane'].value_counts()}")
print(f"\nMagneticFieldStrength:\n{train_df['MagneticFieldStrength'].value_counts()}")
print(f"\nManufacturerModel:\n{train_df['ManufacturerModel'].value_counts()}")

print("\n" + "=" * 60)
print("TESTING CSV")
print("=" * 60)
test_df = pd.read_csv(os.path.join(BASE, 'TestingCohort.csv'))
print(f"Shape       : {test_df.shape}")
first_col = test_df.columns[0]
test_df['plane'] = test_df[first_col].str.extract(r'(sagittal|coronal|transection)', expand=False)
print(f"\nPlane counts:\n{test_df['plane'].value_counts()}")


# ── 2. Folder File Counts ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FOLDER CONTENTS")
print("=" * 60)
for name, path in FOLDERS.items():
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith('.gz')]
        print(f"{name:15s}: {len(files)} .gz files")
        if files:
            print(f"  Sample    : {files[0]}")
    else:
        print(f"{name:15s}: FOLDER NOT FOUND")


# ── 3. Check Pairing (artifact <-> groundtruth) ───────────────────────────────
print("\n" + "=" * 60)
print("PAIRING CHECK")
print("=" * 60)
art_files = set(os.listdir(FOLDERS['artifact'])) if os.path.exists(FOLDERS['artifact']) else set()
gt_files  = set(os.listdir(FOLDERS['groundtruth'])) if os.path.exists(FOLDERS['groundtruth']) else set()

matched   = art_files & gt_files
only_art  = art_files - gt_files
only_gt   = gt_files  - art_files

print(f"Artifact files   : {len(art_files)}")
print(f"GroundTruth files: {len(gt_files)}")
print(f"Matched pairs    : {len(matched)}")
print(f"Only in Artifact : {len(only_art)}")
print(f"Only in GT       : {len(only_gt)}")


# ── 4. Inspect One NIfTI File ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE NIfTI INSPECTION")
print("=" * 60)

def load_first(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.gz')]
    if not files:
        return None, None
    path = os.path.join(folder, files[0])
    img  = nib.load(path).get_fdata()
    return files[0], img

for name, folder in FOLDERS.items():
    if not os.path.exists(folder):
        continue
    fname, img = load_first(folder)
    if img is not None:
        print(f"{name:15s}: {fname}")
        print(f"  Shape  : {img.shape}")
        print(f"  Dtype  : {img.dtype}")
        print(f"  Min/Max: {img.min():.2f} / {img.max():.2f}")
        print(f"  Mean   : {img.mean():.4f}")


# ── 5. Visual — Artifact vs GroundTruth side by side ──────────────────────────
print("\n" + "=" * 60)
print("GENERATING VISUAL COMPARISON")
print("=" * 60)

art_name, art_img = load_first(FOLDERS['artifact'])
gt_name,  gt_img  = load_first(FOLDERS['groundtruth'])

if art_img is not None and gt_img is not None:
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2)

    for idx, (title, img, fname) in enumerate([
        ('ARTIFACT',    art_img, art_name),
        ('GROUND TRUTH', gt_img, gt_name),
    ]):
        mid = img.shape[2] // 2
        ax  = fig.add_subplot(gs[idx])
        ax.imshow(img[:, :, mid], cmap='gray')
        ax.set_title(f'{title}\n{fname}\nSlice {mid}/{img.shape[2]}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(BASE, 'dataset_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.show()


# ── 6. Noise Score Comparison ─────────────────────────────────────────────────
def noise_score(img):
    s  = img[:, :, img.shape[2] // 2].astype(np.float32)
    s  = (s - s.min()) / (s.max() - s.min() + 1e-8)
    Gx = np.gradient(s, axis=1)
    Gy = np.gradient(s, axis=0)
    M  = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    C  = (M.T @ M) / M.shape[0]
    return float(np.trace(C))

if art_img is not None and gt_img is not None:
    print("\n" + "=" * 60)
    print("NOISE SCORE (Gradient Covariance)")
    print("=" * 60)
    print(f"Artifact score    : {noise_score(art_img):.6f}")
    print(f"Ground truth score: {noise_score(gt_img):.6f}")
    print("(Higher = more texture/noise/artifact)")

print("\nDone. Share this output to proceed with modeling.")