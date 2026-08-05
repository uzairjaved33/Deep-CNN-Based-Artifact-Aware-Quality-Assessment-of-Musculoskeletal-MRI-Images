import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom
import os
import json

BASE = r'D:\KMAR-50K\KMAR-50K'
ART_DIR = os.path.join(BASE, 'ArtifactData_part1')
GT_DIR = os.path.join(BASE, 'GroundTruthData_part1')

print(" Computing SSIM (Safe Mode - Auto-Resizing)...")
print("="*60)

all_slice_scores = []
volume_stats = []

# Get all artifact files
art_files = [f for f in os.listdir(ART_DIR) if f.endswith('.nii.gz')]
print(f"Found {len(art_files)} pairs. Processing...\n")

for fname in art_files:
    art_path = os.path.join(ART_DIR, fname)
    gt_path = os.path.join(GT_DIR, fname)
    
    if os.path.exists(gt_path):
        try:
            art_vol = nib.load(art_path).get_fdata()
            gt_vol  = nib.load(gt_path).get_fdata()
            
            # Handle 4D data (take first volume)
            if art_vol.ndim == 4: art_vol = art_vol[:,:,:,0]
            if gt_vol.ndim == 4:  gt_vol = gt_vol[:,:,:,0]
            
            # Safe slice iteration
            min_slices = min(art_vol.shape[2], gt_vol.shape[2])
            slice_scores = []
            
            for i in range(min_slices):
                art_sl = art_vol[:,:,i]
                gt_sl  = gt_vol[:,:,i]
                
                #  Auto-Resize if X/Y dimensions differ
                if art_sl.shape != gt_sl.shape:
                    art_sl = zoom(art_sl, (
                        gt_sl.shape[0]/art_sl.shape[0],
                        gt_sl.shape[1]/art_sl.shape[1]
                    ))
                
                score = ssim(art_sl, gt_sl, data_range=gt_sl.max())
                slice_scores.append(score)
                all_slice_scores.append(score)
            
            mean_s = np.mean(slice_scores)
            volume_stats.append({'file': fname, 'mean_ssim': mean_s})
            print(f"  ✅ {fname:<35} | Mean SSIM: {mean_s:.3f}")
            
        except Exception as e:
            print(f"  ❌ {fname:<35} | Error: {e}")

# ── Summary Statistics ───────────────────────────────────────
if all_slice_scores:
    scores = np.array(all_slice_scores)
    print("\n" + "="*60)
    print("📊 GLOBAL SSIM STATISTICS (Slice-Level)")
    print("="*60)
    print(f"  Total Slices Analyzed: {len(scores)}")
    print(f"  Mean SSIM:  {scores.mean():.4f}")
    print(f"  Std Dev:    {scores.std():.4f}")
    print(f"  Median:     {np.median(scores):.4f}")
    print(f"  Min:        {scores.min():.4f}")
    print(f"  Max:        {scores.max():.4f}")
    
    # Suggested Thresholds
    m, s = scores.mean(), scores.std()
    print("\n🎯 RECOMMENDED LABEL THRESHOLDS:")
    print(f"  Good:      SSIM >= {m + 0.5*s:.3f}")
    print(f"  Moderate:  {m - 0.5*s:.3f} <= SSIM < {m + 0.5*s:.3f}")
    print(f"  Bad:       SSIM < {m - 0.5*s:.3f}")
    
    # Save to JSON
    with open('ssim_results.json', 'w') as f:
        json.dump({'stats': scores.tolist(), 'volumes': volume_stats}, f)
    print("\n Saved full results → ssim_results.json")
else:
    print("\n❌ No data processed. Check paths.")