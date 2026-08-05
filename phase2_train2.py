"""
KMAR-50K Phase 2b — Multi-Task Model + Training (Fixed)
Tasks: Quality (3-class) + Plane (3-class)
Backbone: EfficientNet-B0
Fixes: Early stopping, class weights, stronger regularization, frozen plane head option,
       proper resume from best checkpoint, learning rate scheduling
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
import json
from collections import Counter

# ── Config ────────────────────────────────────────────────────
BASE          = r'D:\KMAR-50K\KMAR-50K'
ART_DIR       = os.path.join(BASE, 'ArtifactData_part1')
GT_DIR        = os.path.join(BASE, 'GroundTruthData_part1')
TRAIN_CSV     = os.path.join(BASE, 'TrainingCohort.csv')
CKPT_DIR      = os.path.join(BASE, 'checkpoints')
HISTORY_FILE  = os.path.join(CKPT_DIR, 'history.json')

IMG_SIZE      = 224
BATCH_SIZE    = 8          # Increased (was 4) — regularization helps stability
NUM_WORKERS   = 0
EPOCHS        = 30
LR            = 5e-5       # Lower LR (was 1e-4) — gentler fine-tuning from best ckpt
WEIGHT_DECAY  = 1e-3       # Added — was missing entirely
NOISE_THRESH  = 0.00065
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Early stopping config
PATIENCE      = 5          # Stop if no val improvement for 5 epochs
MIN_DELTA     = 0.001

# Class weighting for quality (will be computed from data)
QUALITY_WEIGHTS = None     # Auto-computed below

os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ── Noise Score ───────────────────────────────────────────────
def gradient_noise_score(slice_2d):
    s  = slice_2d.astype(np.float32)
    s  = (s - s.min()) / (s.max() - s.min() + 1e-8)
    Gx = np.gradient(s, axis=1)
    Gy = np.gradient(s, axis=0)
    M  = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    C  = (M.T @ M) / M.shape[0]
    return float(np.trace(C))


# ── Build Samples ─────────────────────────────────────────────
PLANE_MAP = {'sagittal': 0, 'coronal': 1, 'transection': 2}

def build_samples(csv_path, art_dir, gt_dir):
    df = pd.read_csv(csv_path)
    df['plane'] = df['NII_FileName'].str.extract(
        r'(sagittal|coronal|transection)', expand=False
    )
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building samples"):
        fname = row['NII_FileName'].strip().strip("'").replace('.nii.gz', '.0.nii.gz')
        plane_str = row['plane']
        if pd.isna(plane_str):
            continue
        plane_label = PLANE_MAP.get(plane_str, -1)
        if plane_label == -1:
            continue

        gt_path = os.path.join(gt_dir, fname)
        if os.path.exists(gt_path):
            try:
                n_slices = nib.load(gt_path).shape[2]
            except:
                n_slices = 1
            for sl_idx in range(n_slices):
                samples.append({'path': gt_path, 'slice': sl_idx,
                                'quality': 0, 'plane': plane_label})

        art_path = os.path.join(art_dir, fname)
        if os.path.exists(art_path):
            try:
                proxy    = nib.load(art_path)
                n_slices = proxy.shape[2]
                mid      = n_slices // 2
                sl_mid   = np.asarray(proxy.dataobj[:, :, mid])
                score    = gradient_noise_score(sl_mid)
                quality  = 1 if score <= NOISE_THRESH else 2
                del proxy, sl_mid
            except:
                n_slices = 1
                quality  = 2
            for sl_idx in range(n_slices):
                samples.append({'path': art_path, 'slice': sl_idx,
                                'quality': quality, 'plane': plane_label})

    return samples


# ── Dataset ───────────────────────────────────────────────────
class KMARDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        proxy  = nib.load(s['path'])
        img    = np.asarray(proxy.dataobj[:, :, s['slice']])
        sl     = img.astype(np.float32)
        sl     = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        sl     = np.stack([sl, sl, sl], axis=0)
        sl     = torch.tensor(sl)
        if self.transform:
            sl = self.transform(sl)
        return (
            sl,
            torch.tensor(s['quality'], dtype=torch.long),
            torch.tensor(s['plane'],   dtype=torch.long),
        )


# ── Transforms ────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),           # Increased from 10
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Added jitter
    transforms.ColorJitter(brightness=0.1, contrast=0.1),        # Added
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


# ── Model ─────────────────────────────────────────────────────
class KMARMultiTask(nn.Module):
    def __init__(self, freeze_backbone=False, freeze_plane_head=False):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=True, num_classes=0
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        feat = self.backbone.num_features  # 1280

        self.head_quality = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Dropout(0.5),               # Increased from 0.3
            nn.Linear(256, 3),             # Good / Moderate / Bad
        )
        self.head_plane = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),             # Sagittal / Coronal / Transection
        )

        if freeze_plane_head:
            for param in self.head_plane.parameters():
                param.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_quality(feat), self.head_plane(feat)

    def unfreeze_backbone(self):
        """Call this after a few epochs to fine-tune backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ── Compute Class Weights ─────────────────────────────────────
def compute_quality_weights(samples):
    qualities = [s['quality'] for s in samples]
    counts = Counter(qualities)
    print(f"Quality class distribution: {dict(sorted(counts.items()))}")
    total = len(qualities)
    # Inverse frequency weighting
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    weight_tensor = torch.tensor([weights[i] for i in range(3)], dtype=torch.float32)
    print(f"Computed class weights: {weight_tensor}")
    return weight_tensor.to(DEVICE)


# ── Train One Epoch ───────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion_q, criterion_p, scheduler=None):
    model.train()
    total_loss = q_correct = p_correct = total = 0

    for imgs, quality, plane in tqdm(loader, desc="  Train", leave=False):
        imgs    = imgs.to(DEVICE)
        quality = quality.to(DEVICE)
        plane   = plane.to(DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            q_out, p_out = model(imgs)
            loss_q = criterion_q(q_out, quality)
            loss_p = criterion_p(p_out, plane)
            loss = loss_q + loss_p

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += loss.item()
        q_correct  += (q_out.argmax(1) == quality).sum().item()
        p_correct  += (p_out.argmax(1) == plane).sum().item()
        total      += imgs.size(0)

    return total_loss / len(loader), q_correct / total, p_correct / total


# ── Validate ──────────────────────────────────────────────────
def val_epoch(model, loader, criterion_q, criterion_p):
    model.eval()
    total_loss = q_correct = p_correct = total = 0
    all_q_preds, all_q_true = [], []
    all_p_preds, all_p_true = [], []

    with torch.no_grad():
        for imgs, quality, plane in tqdm(loader, desc="  Val  ", leave=False):
            imgs    = imgs.to(DEVICE)
            quality = quality.to(DEVICE)
            plane   = plane.to(DEVICE)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                q_out, p_out = model(imgs)
                loss_q = criterion_q(q_out, quality)
                loss_p = criterion_p(p_out, plane)
                loss = loss_q + loss_p

            total_loss += loss.item()
            q_correct  += (q_out.argmax(1) == quality).sum().item()
            p_correct  += (p_out.argmax(1) == plane).sum().item()
            total      += imgs.size(0)

            all_q_preds.extend(q_out.argmax(1).cpu().numpy())
            all_q_true.extend(quality.cpu().numpy())
            all_p_preds.extend(p_out.argmax(1).cpu().numpy())
            all_p_true.extend(plane.cpu().numpy())

    # Per-class accuracy for quality
    q_cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(all_q_true, all_q_preds):
        q_cm[t, p] += 1
    q_per_class = q_cm.diagonal() / (q_cm.sum(axis=1) + 1e-8)

    return (total_loss / len(loader), q_correct / total, p_correct / total,
            q_per_class, q_cm)


# ── Save / Load Checkpoint ────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, history, path):
    ckpt = {
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'best_loss':  best_loss,
    }
    if scheduler is not None:
        ckpt['scheduler'] = scheduler.state_dict()
    torch.save(ckpt, path)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'], ckpt['best_loss']


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':

    # Build samples
    print("Building samples...")
    samples = build_samples(TRAIN_CSV, ART_DIR, GT_DIR)
    print(f"Total: {len(samples)}")

    # 70 / 20 / 10 split
    train_s, temp_s  = train_test_split(samples, test_size=0.30, random_state=42)
    val_s,   test_s  = train_test_split(temp_s,  test_size=0.33, random_state=42)
    print(f"Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    # Save test set
    test_paths = [s['path'] for s in test_s]
    with open(os.path.join(CKPT_DIR, 'test_samples.json'), 'w') as f:
        json.dump(test_s, f, indent=2)
    print("Test set saved → checkpoints/test_samples.json (locked)")

    # Compute class weights for quality
    quality_weights = compute_quality_weights(train_s)

    # Dataloaders
    train_dl = DataLoader(KMARDataset(train_s, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(KMARDataset(val_s, val_tf),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # Model — start with plane head frozen for first N epochs
    FREEZE_PLANE_EPOCHS = 3
    model = KMARMultiTask(freeze_backbone=False, freeze_plane_head=True).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler('cuda')  # Fixed deprecation warning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Separate losses: weighted for quality, unweighted for plane
    criterion_q = nn.CrossEntropyLoss(weight=quality_weights)
    criterion_p = nn.CrossEntropyLoss()

    # Checkpoint paths
    best_ckpt   = os.path.join(CKPT_DIR, 'best_model.pt')
    last_ckpt   = os.path.join(CKPT_DIR, 'last_model.pt')
    resume_ckpt = best_ckpt  # <<< RESUME FROM BEST, NOT LAST

    start_epoch = 0
    best_loss   = float('inf')
    history     = {'train_loss': [], 'val_loss': [],
                   'train_q_acc': [], 'val_q_acc': [],
                   'train_p_acc': [], 'val_p_acc': [],
                   'val_q_per_class': [], 'lr': []}

    # ── Resume logic ──────────────────────────────────────────
    if os.path.exists(resume_ckpt):
        print(f"\nResuming from BEST checkpoint: {resume_ckpt}")
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, resume_ckpt)
        start_epoch += 1
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                history = json.load(f)
        print(f"Resuming from epoch {start_epoch}, best val loss so far: {best_loss:.4f}")
    else:
        print("\nNo best checkpoint found. Starting fresh training...")

    # Early stopping state
    epochs_no_improve = 0

    # Training Loop
    print(f"\nTraining for {EPOCHS} epochs on {DEVICE}\n")
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")

        # Unfreeze plane head after warmup
        if epoch == FREEZE_PLANE_EPOCHS:
            print("  >>> Unfreezing plane head <<<")
            for param in model.head_plane.parameters():
                param.requires_grad = True

        t_loss, t_qacc, t_pacc = train_epoch(
            model, train_dl, optimizer, scaler, criterion_q, criterion_p, scheduler
        )
        v_loss, v_qacc, v_pacc, v_q_per, v_cm = val_epoch(
            model, val_dl, criterion_q, criterion_p
        )

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        print(f"  Train → Loss: {t_loss:.4f} | Quality Acc: {t_qacc:.3f} | Plane Acc: {t_pacc:.3f}")
        print(f"  Val   → Loss: {v_loss:.4f} | Quality Acc: {v_qacc:.3f} | Plane Acc: {v_pacc:.3f}")
        print(f"  Val Quality per-class: Good={v_q_per[0]:.3f} Mod={v_q_per[1]:.3f} Bad={v_q_per[2]:.3f}")
        print(f"  LR: {current_lr:.2e}")

        # Save history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_q_acc'].append(t_qacc)
        history['val_q_acc'].append(v_qacc)
        history['train_p_acc'].append(t_pacc)
        history['val_p_acc'].append(v_pacc)
        history['val_q_per_class'].append(v_q_per.tolist())
        history['lr'].append(current_lr)

        # Save last checkpoint (always)
        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, history, last_ckpt)

        # Early stopping & best model logic
        improved = v_loss < (best_loss - MIN_DELTA)
        if improved:
            best_loss = v_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_loss, history, best_ckpt)
            print(f"  *** Best model saved (val_loss={best_loss:.4f}) ***")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{PATIENCE} epochs")

        # Save per-epoch checkpoint (optional, comment out to save disk)
        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, history,
                        os.path.join(CKPT_DIR, f'epoch_{epoch+1:02d}.pt'))

        # Early stop check
        if epochs_no_improve >= PATIENCE:
            print(f"\n!!! Early stopping triggered after {epoch+1} epochs !!!")
            break

    print("\nTraining complete!")
    print(f"Best val loss : {best_loss:.4f}")
    print(f"Checkpoints   : {CKPT_DIR}")