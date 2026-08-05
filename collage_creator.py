"""
MRI Collage Viewer
Shows 4 rows x 2 columns: Artifact | Ground Truth pairs
Saves as high quality PNG for inspection
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

BASE    = r'D:\KMAR-50K\KMAR-50K'
ART_DIR = os.path.join(BASE, 'ArtifactData_part1')
GT_DIR  = os.path.join(BASE, 'GroundTruthData_part1')
OUT_DIR = os.path.join(BASE, 'collages')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

ROWS    = 4   # 4 pairs
SLICES  = 3   # show 3 slices per volume (25%, 50%, 75%)
NUM_IMAGES = 5 # How many collage images to generate

# ── Pick random paired files ───────────────────────────────────
all_files = [f for f in os.listdir(ART_DIR) if f.endswith('.gz')]

for i in range(NUM_IMAGES):
    print(f"─── Creating Collage {i+1}/{NUM_IMAGES} ────────────────────────")
    selected  = random.sample(all_files, ROWS)

    fig = plt.figure(figsize=(20, ROWS * 5))
    fig.patch.set_facecolor('black')

    outer = gridspec.GridSpec(ROWS, 1, hspace=0.05)

    for row_idx, fname in enumerate(selected):
        art_path = os.path.join(ART_DIR, fname)
        gt_path  = os.path.join(GT_DIR,  fname)

        if not os.path.exists(gt_path):
            print(f"No GT pair for {fname}, skipping")
            continue

        art_vol = nib.load(art_path).get_fdata()
        gt_vol  = nib.load(gt_path).get_fdata()
        n       = art_vol.shape[2]

        # Pick 3 representative slices
        slice_indices = [n//4, n//2, 3*n//4]

        inner = gridspec.GridSpecFromSubplotSpec(
            1, SLICES * 2, subplot_spec=outer[row_idx], wspace=0.02
        )

        for col_idx, sl_idx in enumerate(slice_indices):
            # Artifact
            ax_art = fig.add_subplot(inner[col_idx * 2])
            ax_art.imshow(art_vol[:, :, sl_idx], cmap='gray', aspect='auto')
            ax_art.axis('off')
            if col_idx == 0:
                ax_art.set_title(f'ARTIFACT\n{fname[:25]}', color='red',
                                fontsize=7, pad=3)
            else:
                ax_art.set_title(f'Slice {sl_idx}', color='red',
                                fontsize=7, pad=3)

            # Ground Truth
            ax_gt = fig.add_subplot(inner[col_idx * 2 + 1])
            ax_gt.imshow(gt_vol[:, :, sl_idx], cmap='gray', aspect='auto')
            ax_gt.axis('off')
            if col_idx == 0:
                ax_gt.set_title(f'GROUND TRUTH\n{fname[:25]}', color='lime',
                               fontsize=7, pad=3)
            else:
                ax_gt.set_title(f'Slice {sl_idx}', color='lime',
                               fontsize=7, pad=3)

    plt.suptitle('KMAR-50K: Artifact vs Ground Truth — Visual Inspection',
                 color='white', fontsize=14, y=0.95)
    
    out_path = os.path.join(OUT_DIR, f'collage_inspection_{i+1}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"Saved → {out_path}")
    plt.close(fig)

print("\nDone.")