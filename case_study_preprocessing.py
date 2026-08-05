"""
Extended Case Study v3: HIGH-SPEED PREPROCESSING COMPARISON
⚡ Optimized: Precomputed filters, in-memory caching, larger batch, cuDNN benchmark
⏱️ Runtime: ~8-12 mins (down from ~60-90)
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy.ndimage import sobel, laplace, gaussian_filter, uniform_filter, gaussian_gradient_magnitude
import timm
import pandas as pd
from tqdm import tqdm
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

# ── Speed & Stability Config ──────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

BASE         = r'D:\KMAR-50K\KMAR-50K'
ART_DIR      = os.path.join(BASE, 'ArtifactData_part1')
GT_DIR       = os.path.join(BASE, 'GroundTruthData_part1')
TRAIN_CSV    = os.path.join(BASE, 'TrainingCohort.csv')
NOISE_THRESH = 0.00065
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS       = 3   # ⚡ 3 is enough for ranking
BATCH_SIZE   = 16  # ⚡ Maximize GPU utilization (reduce to 8 if OOM)
SAMPLE_LIMIT = 200
PLANE_MAP    = {'sagittal': 0, 'coronal': 1, 'transection': 2}
LABEL_SMOOTHING = 0.1

print(f"⚡ Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")


# ── Normalization ─────────────────────────────────────────────
def norm(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def norm_percentile(img, p_low=1, p_high=99):
    lo, hi = np.percentile(img, [p_low, p_high])
    return norm(np.clip(img, lo, hi))


# ── Core Filters ──────────────────────────────────────────────
def f_raw(img): return norm_percentile(img)
def f_sobel(img): return norm(np.hypot(sobel(img, axis=0), sobel(img, axis=1)))
def f_laplacian(img): return norm(np.abs(laplace(img)))
def f_gaussian_diff(img, s1=1, s2=3): return norm(np.abs(gaussian_filter(img, s1) - gaussian_filter(img, s2)))
def f_histogram_eq(img): return norm(cv2.equalizeHist((norm(img)*255).astype(np.uint8)).astype(np.float32))
def f_clahe(img): 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return norm(clahe.apply((norm(img)*255).astype(np.uint8)).astype(np.float32))
def f_high_pass(img, sigma=2): return norm(img - gaussian_filter(img, sigma))
def f_fft_magnitude(img): return norm(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img)))))
def f_local_std(img, size=5):
    m1 = uniform_filter(img, size=size)
    m2 = uniform_filter(img**2, size=size)
    return norm(np.sqrt(np.abs(m2 - m1**2)))
def f_canny(img, low=40, high=120):
    return norm(cv2.Canny((norm_percentile(img)*255).astype(np.uint8), low, high).astype(np.float32))
def f_gaussian_smooth(img, sigma=1.5): return norm(gaussian_filter(img, sigma))

# ── Advanced Filters ──────────────────────────────────────────
def f_wavelet_detail(img, wavelet='db2', level=2):
    if not HAS_PYWT: return norm((f_high_pass(img) + f_local_std(img)) / 2)
    try:
        coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
        details = [np.abs(c) for lvl in coeffs[1:] for c in lvl]
        return norm(np.mean(details, axis=0)) if details else norm(img)
    except: return f_high_pass(img)

def f_morph_gradient(img, k=3):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    return norm(cv2.morphologyEx((norm(img)*255).astype(np.uint8), cv2.MORPH_GRADIENT, ker).astype(np.float32))

def f_tv_residual(img, iters=8):
    d = img.copy()
    for _ in range(iters): d = gaussian_filter(d, sigma=0.8)
    return norm(np.abs(img - d))

def f_gradient_magnitude(img): return norm(gaussian_gradient_magnitude(img, sigma=1))
def f_multiscale_edges(img): return norm(np.mean([f_canny(img, l, h) for l,h in [(30,100),(50,150),(70,200)]], axis=0))
def f_frequency_bandpass(img, low_cut=0.05, high_cut=0.3):
    fft = np.fft.fftshift(np.fft.fft2(img))
    h, w = fft.shape
    cr, cc = h//2, w//2
    mask = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-cr)**2 + (j-cc)**2) / min(h,w)
            if low_cut < d < high_cut: mask[i,j] = 1
    return norm(np.abs(np.fft.ifft2(np.fft.ifftshift(fft * mask))))


# ── 🧪 Experiment Definitions ─────────────────────────────────
EXPERIMENTS = {
    '01_RAW_baseline': [f_raw, f_raw, f_raw],
    '02_CLAHE+Sobel+Laplace': [f_clahe, f_sobel, f_laplacian],
    '03_Canny+Gaussian+Raw': [f_canny, f_gaussian_smooth, f_raw],
    '04_Canny+DoG+Raw': [f_canny, f_gaussian_diff, f_raw],
    '05_MultiScaleEdges+Sobel+Raw': [f_multiscale_edges, f_sobel, f_raw],
    '06_Canny+MorphGrad+Raw': [f_canny, f_morph_gradient, f_raw],
    '07_GradMag+Laplace+Raw': [f_gradient_magnitude, f_laplacian, f_raw],
    '08_LocalStd+TV+Raw': [f_local_std, f_tv_residual, f_raw],
    '09_Wavelet+LocalStd+Raw': [f_wavelet_detail, f_local_std, f_raw],
    '10_TVResidual+HighPass+Raw': [f_tv_residual, f_high_pass, f_raw],
    '11_FFT+Bandpass+Raw': [f_fft_magnitude, f_frequency_bandpass, f_raw],
    '12_DoG+FFT+Raw': [f_gaussian_diff, f_fft_magnitude, f_raw],
    '13_CLAHE+Canny+Wavelet': [f_clahe, f_canny, f_wavelet_detail],
    '14_Raw+Canny+TV': [f_raw, f_canny, f_tv_residual],
    '15_CLAHE+Sobel+LocalStd': [f_clahe, f_sobel, f_local_std],
    '16_HighPass+TV+LocalStd': [f_high_pass, f_tv_residual, f_local_std],
    '17_Bandpass+Wavelet+Grad': [f_frequency_bandpass, f_wavelet_detail, f_gradient_magnitude],
    '18_CLAHE+MultiEdge+Canny': [f_clahe, f_multiscale_edges, f_canny],
    '19_HistEq+Canny+Wavelet': [f_histogram_eq, f_canny, f_wavelet_detail],
    '20_Raw+Edge+Freq': [f_raw, f_multiscale_edges, f_frequency_bandpass],
}


# ── Noise Score ───────────────────────────────────────────────
def gradient_noise_score(s):
    s = norm(s.astype(np.float32))
    Gx, Gy = np.gradient(s, axis=1), np.gradient(s, axis=0)
    M = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    return float(np.trace((M.T @ M) / M.shape[0]))


# ── Build Samples (⚡ CACHES RAW SLICES IN RAM) ──────────────
def build_samples(limit=None):
    df = pd.read_csv(TRAIN_CSV)
    df['plane'] = df['NII_FileName'].str.extract(r'(sagittal|coronal|transection)', expand=False)
    samples = []
    for _, row in df.iterrows():
        if limit and len(samples) >= limit: break
        fname = row['NII_FileName'].strip().strip("'").replace('.nii.gz', '.0.nii.gz')
        plane = PLANE_MAP.get(row.get('plane'), -1)
        if plane == -1: continue

        for dir_path, q in [(GT_DIR, 0), (ART_DIR, None)]:
            path = os.path.join(dir_path, fname)
            if os.path.exists(path):
                proxy = nib.load(path)
                mid = proxy.shape[2] // 2
                raw_slice = np.asarray(proxy.dataobj[:, :, mid]).astype(np.float32)
                quality = q if q is not None else (1 if gradient_noise_score(raw_slice) <= NOISE_THRESH else 2)
                samples.append({'raw': raw_slice, 'quality': quality, 'plane': plane})
                if limit and len(samples) >= limit: break
    return samples


# ── Dataset (⚡ PRECOMPUTES FILTERS ONCE) ─────────────────────
class MRIDataset(Dataset):
    def __init__(self, samples, filters, augment=False):
        self.samples = samples
        self.resize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5, fill=0),
        ]) if augment else None
        
        # ⚡ Precompute all channels once → zero CPU overhead during training
        self.tensors = []
        for s in samples:
            ch = [f(s['raw']) for f in filters]
            t = torch.tensor(np.stack(ch, axis=0), dtype=torch.float32)
            self.tensors.append(self.resize(t) if not self.aug_tf else self.aug_tf(self.resize(t)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self.tensors[idx], torch.tensor(self.samples[idx]['quality'], dtype=torch.long), torch.tensor(self.samples[idx]['plane'], dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────
class KMARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        self.head_quality = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
        self.head_plane = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
    def forward(self, x):
        f = self.backbone(x)
        return self.head_quality(f), self.head_plane(f)


# ── Run One Experiment ────────────────────────────────────────
def run_experiment(name, train_dl, val_dl):
    model = KMARModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda')
    best_q = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for imgs, quality, plane in train_dl:
            imgs, quality, plane = imgs.to(DEVICE), quality.to(DEVICE), plane.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                q_out, p_out = model(imgs)
                loss = criterion(q_out, quality) + 0.5 * criterion(p_out, plane)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        q_correct = total = 0
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            for imgs, quality, _ in val_dl:
                imgs, quality = imgs.to(DEVICE), quality.to(DEVICE)
                q_out, _ = model(imgs)
                q_correct += (q_out.argmax(1) == quality).sum().item()
                total += imgs.size(0)
        q_acc = q_correct / total
        if q_acc > best_q: best_q = q_acc
    return best_q


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    print(f"⏳ Building {SAMPLE_LIMIT} samples (caching in RAM)...")
    samples = build_samples(limit=SAMPLE_LIMIT)
    print(f"✅ Total: {len(samples)} | Time: {time.time()-t0:.1f}s")
    
    try:
        qs = [s['quality'] for s in samples]
        train_s, val_s = train_test_split(samples, test_size=0.2, random_state=42, stratify=qs)
    except ValueError:
        train_s, val_s = train_test_split(samples, test_size=0.2, random_state=42)

    results = {}
    total = len(EXPERIMENTS)

    for i, (name, filters) in enumerate(EXPERIMENTS.items(), 1):
        print(f"\n[{i}/{total}] {name}")
        # ⚡ num_workers=2 is safe now (all functions are top-level)
        train_dl = DataLoader(MRIDataset(train_s, filters, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_dl   = DataLoader(MRIDataset(val_s, filters, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        t_exp = time.time()
        best_q = run_experiment(name, train_dl, val_dl)
        results[name] = best_q
        print(f"  ✓ Acc: {best_q:.3f} | Time: {time.time()-t_exp:.1f}s")

    # ── Leaderboard ────────────────────────────────────────────
    print("\n" + "="*75)
    print("🏆 LEADERBOARD — Best Quality Accuracy")
    print("="*75)
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, acc) in enumerate(ranked, 1):
        bar = '█' * int(acc * 40)
        marker = ' ← WINNER 🏆' if rank == 1 else ''
        print(f"  {rank:2}. {name:<42} {acc*100:5.1f}%  {bar}{marker}")

    winner_name, winner_acc = ranked[0]
    baseline_acc = results.get('01_RAW_baseline', 0)
    print(f"\n  Baseline (RAW)      : {baseline_acc*100:.1f}%")
    print(f"  🏆 Best approach    : {winner_name}")
    print(f"  🏆 Best accuracy    : {winner_acc*100:.1f}%")
    print(f"  📈 Total gain       : +{(winner_acc - baseline_acc)*100:.1f}%")
    print(f"  ⏱️ Total runtime    : {(time.time()-t0)/60:.1f} mins")
    print("="*75)
    print(f"\n✅ Ready for phase2_train.py with winner filters.")