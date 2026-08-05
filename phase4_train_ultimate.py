"""
Phase 5 — STABLE FINE-TUNE (Fixes Oscillation)
- Loads Best Checkpoint (Epoch 17)
- Lowers Learning Rate to 1e-4 (Precision Mode)
- Uses Smooth Cosine Scheduler (No spikes)
- Resumes at Epoch 18
"""
import os, json, time, warnings, gc, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm, nibabel as nib
from collections import Counter
import cv2

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ── Config ──────────────────────────────────────────────────
BASE          = r'D:\KMAR-50K\KMAR-50K'
LABELED_JSON  = os.path.join(BASE, 'samples_ssim_labeled.json')
CKPT_DIR      = os.path.join(BASE, 'checkpoints')
CACHE_DIR     = os.path.join(BASE, 'preprocessed_cache')
VAL_CACHE_DIR = os.path.join(BASE, 'val_cache')
HISTORY_FILE  = os.path.join(CKPT_DIR, 'history_phase5.json')
IMG_SIZE      = 224
BATCH_SIZE    = 8
NUM_WORKERS   = 0
EPOCHS        = 30
LR            = 1e-4        # ⬇️ Lower LR for stability
WEIGHT_DECAY  = 5e-4
PATIENCE      = 8
MIN_DELTA     = 0.001
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_BATCHES     = 15

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VAL_CACHE_DIR, exist_ok=True)
print(f"⚡ Using Device: {DEVICE} | Cache: {CACHE_DIR}")


# ── Focal Loss ───────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Model ────────────────────────────────────────────────────
class KMARMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        self.head_quality = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 3))
        self.head_plane = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
    def forward(self, x):
        f = self.backbone(x)
        return self.head_quality(f), self.head_plane(f)


# ── Disk-Based Dataset ──────────────────────────────────────
class DiskDataset(Dataset):
    def __init__(self, cache_dir, transform=None):
        self.cache_dir = cache_dir
        self.transform = transform
        self.samples = []
        with open(os.path.join(cache_dir, 'index.json')) as f:
            self.samples = json.load(f)
        print(f"📂 Loaded {len(self.samples):,} samples from {cache_dir}")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        data = np.load(os.path.join(self.cache_dir, s['file']), allow_pickle=True).item()
        slc = torch.tensor(data['img'], dtype=torch.float32)
        if self.transform: slc = self.transform(slc)
        return slc, torch.tensor(data['quality'], dtype=torch.long), torch.tensor(data['plane'], dtype=torch.long)


# ── Preprocessing Functions (Kept for safety) ───────────────
def preprocess_sample_slice(s, img_size=IMG_SIZE):
    proxy = nib.load(s['path'])
    vol_data = proxy.dataobj
    n_slices = proxy.shape[2]
    idx_sl = s['slice']
    prev_sl = np.asarray(vol_data[:, :, max(0, idx_sl-1)], dtype=np.float32)
    curr_sl = np.asarray(vol_data[:, :, idx_sl], dtype=np.float32)
    next_sl = np.asarray(vol_data[:, :, min(n_slices-1, idx_sl+1)], dtype=np.float32)
    slc = np.stack([prev_sl, curr_sl, next_sl], axis=0)
    for c in range(3):
        mn, mx = slc[c].min(), slc[c].max()
        slc[c] = (slc[c] - mn) / (mx - mn + 1e-8)
    slc_resized = np.zeros((3, img_size, img_size), dtype=np.float32)
    for c in range(3):
        slc_resized[c] = cv2.resize(slc[c], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    slc_resized = (slc_resized - 0.5) / 0.5
    return slc_resized, s['quality'], s['plane']

def preprocess_and_save_batch(batch_samples, batch_idx, cache_dir):
    print(f"\n🔄 Batch {batch_idx+1} ({len(batch_samples):,} samples)")
    batch_index = []
    for i, s in enumerate(tqdm(batch_samples, desc="Slices", leave=False)):
        try:
            slc_resized, quality, plane = preprocess_sample_slice(s)
            filename = f"sample_{batch_idx:02d}_{i:05d}.npy"
            filepath = os.path.join(cache_dir, filename)
            np.save(filepath, {'img': slc_resized, 'quality': int(quality), 'plane': int(plane)}, allow_pickle=True)
            batch_index.append({'file': filename, 'quality': int(quality), 'plane': int(plane)})
            if i % 100 == 0: gc.collect()
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue
    with open(os.path.join(cache_dir, f'index_batch_{batch_idx}.json'), 'w') as f:
        json.dump(batch_index, f)
    print(f"   ✅ Batch {batch_idx+1} saved")
    gc.collect()
    return batch_index


# ── Training Functions ───────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion_q, criterion_p, scheduler):
    model.train()
    total_loss = q_correct = p_correct = total = 0
    for imgs, quality, plane in tqdm(loader, desc="  Train", leave=False):
        imgs, quality, plane = imgs.to(DEVICE, non_blocking=True), quality.to(DEVICE, non_blocking=True), plane.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            q_out, p_out = model(imgs)
            loss = criterion_q(q_out, quality) + criterion_p(p_out, plane)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        q_correct  += (q_out.argmax(1) == quality).sum().item()
        p_correct  += (p_out.argmax(1) == plane).sum().item()
        total      += imgs.size(0)
    return total_loss / len(loader), q_correct / total, p_correct / total

def val_epoch(model, loader, criterion_q, criterion_p):
    model.eval()
    total_loss = q_correct = p_correct = total = 0
    with torch.no_grad():
        for imgs, quality, plane in loader:
            imgs, quality, plane = imgs.to(DEVICE, non_blocking=True), quality.to(DEVICE, non_blocking=True), plane.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                q_out, p_out = model(imgs)
                loss = criterion_q(q_out, quality) + criterion_p(p_out, plane)
            total_loss += loss.item()
            q_correct  += (q_out.argmax(1) == quality).sum().item()
            p_correct  += (p_out.argmax(1) == plane).sum().item()
            total      += imgs.size(0)
    return total_loss / len(loader), q_correct / total, p_correct / total


# ── Main ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("📂 Loading SSIM-labeled dataset...")
    with open(LABELED_JSON) as f:
        all_samples = json.load(f)
    print(f"   Loaded {len(all_samples):,} samples")

    train_s, temp_s = train_test_split(all_samples, test_size=0.30, random_state=42)
    val_s, test_s   = train_test_split(temp_s, test_size=0.33, random_state=42)
    with open(os.path.join(CKPT_DIR, 'test_samples_ssim.json'), 'w') as f:
        json.dump(test_s, f, indent=2)
    print(f"   Split → Train: {len(train_s):,} | Val: {len(val_s):,} | Test: {len(test_s):,}")

    # ── PHASE 1: PREPROCESSING (Skip if exists) ─────────────
    print("\n" + "="*60)
    print("📦 PHASE 1: PREPROCESSING")
    print("="*60)
    train_index_file = os.path.join(CACHE_DIR, 'index.json')
    if os.path.exists(train_index_file):
        print(f"✅ Train cache found — skipping preprocessing")
    else:
        print(f"🔄 Preprocessing training set...")
        batch_size = len(train_s) // N_BATCHES + 1
        batches = [train_s[i:i+batch_size] for i in range(0, len(train_s), batch_size)]
        all_index = []
        for batch_idx, batch_samples in enumerate(batches):
            batch_index = preprocess_and_save_batch(batch_samples, batch_idx, CACHE_DIR)
            all_index.extend(batch_index)
        with open(train_index_file, 'w') as f: json.dump(all_index, f)
        print(f"✅ Train preprocessing complete")
    
    val_index_file = os.path.join(VAL_CACHE_DIR, 'index.json')
    if os.path.exists(val_index_file):
        print(f"✅ Val cache found — skipping")
    else:
        print(f"🔄 Preprocessing validation set...")
        val_batch_size = len(val_s) // 10 + 1
        val_batches = [val_s[i:i+val_batch_size] for i in range(0, len(val_s), val_batch_size)]
        all_val_index = []
        for i, batch in enumerate(val_batches):
            batch_idx = preprocess_and_save_batch(batch, i, VAL_CACHE_DIR)
            all_val_index.extend(batch_idx)
        with open(val_index_file, 'w') as f: json.dump(all_val_index, f)
        print(f"✅ Val preprocessing complete")
    
    # ── PHASE 2: TRAINING ───────────────────────────────────
    print("\n" + "="*60)
    print("🚀 PHASE 2: STABLE FINE-TUNE")
    print("="*60)
    
    train_dataset = DiskDataset(CACHE_DIR, transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)]))
    val_dataset = DiskDataset(VAL_CACHE_DIR)
    
    qualities = [s['quality'] for s in train_s]
    counts = Counter(qualities)
    total = len(qualities)
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    weight_tensor = torch.tensor([weights[i] for i in range(3)], dtype=torch.float32).to(DEVICE)
    
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = KMARMultiTask().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')
    
    # ✅ STABLE SCHEDULER: Smooth decay, no spikes
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion_q = FocalLoss(gamma=2.0, weight=weight_tensor)
    criterion_p = nn.CrossEntropyLoss()
    
    # ── RESUME LOGIC ────────────────────────────────────────
    START_EPOCH = 18  # Resume from here
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_q_acc': [], 'val_q_acc': [], 'lr': []}
    epochs_no_improve = 0
    
    BEST_MODEL_PATH = os.path.join(CKPT_DIR, 'best_model_phase4.pt')
    
    if os.path.exists(BEST_MODEL_PATH):
        print(f"🔄 Found best model weights (Epoch 17). Loading for stable fine-tune...")
        try:
            ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model'])
            print(f"   ✅ Loaded weights. Starting Epoch {START_EPOCH}/{EPOCHS}")
            print(f"   ⚠️ Optimizer reset to LR={LR} for stability.")
        except Exception as e:
            print(f"   ⚠️ Weights load failed: {e}")
    else:
        print(f"🆕 No checkpoints found. Starting fresh training.")
    # ────────────────────────────────────────────────────────
    
    t0 = time.time()
    for epoch in range(START_EPOCH, EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        t_loss, t_qacc, t_pacc = train_epoch(model, train_dl, optimizer, scaler, criterion_q, criterion_p, scheduler)
        v_loss, v_qacc, v_pacc = val_epoch(model, val_dl, criterion_q, criterion_p)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"  Train → Loss: {t_loss:.4f} | Quality: {t_qacc:.3f} | Plane: {t_pacc:.3f}")
        print(f"  Val   → Loss: {v_loss:.4f} | Quality: {v_qacc:.3f} | Plane: {v_pacc:.3f}")
        print(f"  LR    → {current_lr:.2e}")
        
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_q_acc'].append(t_qacc); history['val_q_acc'].append(v_qacc)
        history['lr'].append(current_lr)
        with open(HISTORY_FILE, 'w') as f: json.dump(history, f, indent=2)
        
        improved = v_loss < (best_loss - MIN_DELTA)
        if improved:
            best_loss = v_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # ── SAVE CHECKPOINT ─────────────────────────────────
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_loss': best_loss,
            'epochs_no_improve': epochs_no_improve
        }, os.path.join(CKPT_DIR, 'resume_checkpoint.pt'))
        
        if improved:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_loss': best_loss
            }, os.path.join(CKPT_DIR, 'best_model_phase5.pt'))
            print(f"  ✅ Checkpoint + Best Model Saved (val_loss={best_loss:.4f})")
        else:
            print(f"  ✅ Checkpoint Saved | ⏳ No improvement ({epochs_no_improve}/{PATIENCE})")
        # ───────────────────────────────────────────────────
        
        if epochs_no_improve >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🎉 Phase 5 complete! Time: {(time.time()-t0)/60:.1f}m")
    print(f"   Best val loss: {best_loss:.4f}")