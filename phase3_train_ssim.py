"""
Phase 3: Fine-tune with SSIM-Relabeled Dataset (GPU-OPTIMIZED)
Fixes: Data loading bottleneck, GPU utilization
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm, nibabel as nib
from collections import Counter
import torch.backends.cudnn as cudnn
import time

# ── CRITICAL: Speed Optimizations ────────────────────────────
cudnn.benchmark = True
cudnn.deterministic = False  # Faster, non-deterministic
torch.set_float32_matmul_precision('high')

# ── Config ────────────────────────────────────────────────────
BASE = r'D:\KMAR-50K\KMAR-50K'
LABELED_JSON = os.path.join(BASE, 'samples_ssim_labeled.json')
CKPT_DIR = os.path.join(BASE, 'checkpoints')
BEST_CKPT = os.path.join(CKPT_DIR, 'best_model.pt')
HISTORY_FILE = os.path.join(CKPT_DIR, 'history_ssim.json')

IMG_SIZE      = 224
BATCH_SIZE    = 16
NUM_WORKERS   = 4          # ⬆️ INCREASED from 2 → 4 (you have RAM!)
EPOCHS        = 20
LR            = 1e-5
WEIGHT_DECAY  = 1e-3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE      = 6
MIN_DELTA     = 0.001

# Verify GPU is active
if DEVICE.type == 'cuda':
    print(f"✅ CUDA Active: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
else:
    print("⚠️ WARNING: Running on CPU! Check CUDA installation.")

os.makedirs(CKPT_DIR, exist_ok=True)


# ── Optimized Dataset (Pre-allocate) ────────────────────────
class SSIMDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    # def __getitem__(self, idx):
    #     s = self.samples[idx]
    #     # Use memory-mapped loading (faster)
    #     with nib.load(s['path']) as proxy:
    #         img = np.asarray(proxy.dataobj[:, :, s['slice']])
    #     sl = img.astype(np.float32)
    #     sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    #     sl = np.stack([sl, sl, sl], axis=0)
    #     sl = torch.tensor(sl, dtype=torch.float32)
    #     if self.transform: sl = self.transform(sl)
    #     return sl, torch.tensor(s['quality'], dtype=torch.long), torch.tensor(s['plane'], dtype=torch.long)
    def __getitem__(self, idx):
        s = self.samples[idx]
        proxy = nib.load(s['path'])  # No 'with' statement
        img = np.asarray(proxy.dataobj[:, :, s['slice']])
        sl = img.astype(np.float32)
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        sl = np.stack([sl, sl, sl], axis=0)
        sl = torch.tensor(sl, dtype=torch.float32)
        if self.transform:
            sl = self.transform(sl)
        return sl, torch.tensor(s['quality'], dtype=torch.long), torch.tensor(s['plane'], dtype=torch.long)

# ── Model ────────────────────────────────────────────────────
class KMARMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        self.head_quality = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 3))
        self.head_plane = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
    def forward(self, x):
        f = self.backbone(x)
        return self.head_quality(f), self.head_plane(f)


# ── Transforms ────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.Normalize([0.5]*3, [0.5]*3)])


# ── Training Functions (with timing) ────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion_q, criterion_p, device):
    model.train()
    total_loss = q_correct = p_correct = total = 0
    start_time = time.time()
    
    for i, (imgs, quality, plane) in enumerate(loader):
        # ⚡ CRITICAL: Non-blocking transfer to GPU
        imgs = imgs.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True)
        plane = plane.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            q_out, p_out = model(imgs)
            loss = criterion_q(q_out, quality) + criterion_p(p_out, plane)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        q_correct  += (q_out.argmax(1) == quality).sum().item()
        p_correct  += (p_out.argmax(1) == plane).sum().item()
        total      += imgs.size(0)
        
        # Progress update every 100 batches
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    Batch {i}/{len(loader)} | Loss: {loss.item():.4f} | Speed: {elapsed/(i+1)*1000:.0f}ms/batch", end='\r')
    
    return total_loss / len(loader), q_correct / total, p_correct / total

def val_epoch(model, loader, criterion_q, criterion_p, device):
    model.eval()
    total_loss = q_correct = p_correct = total = 0
    with torch.no_grad():
        for imgs, quality, plane in loader:
            imgs = imgs.to(device, non_blocking=True)
            quality = quality.to(device, non_blocking=True)
            plane = plane.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                q_out, p_out = model(imgs)
                loss = criterion_q(q_out, quality) + criterion_p(p_out, plane)
            total_loss += loss.item()
            q_correct  += (q_out.argmax(1) == quality).sum().item()
            p_correct  += (p_out.argmax(1) == plane).sum().item()
            total      += imgs.size(0)
    return total_loss / len(loader), q_correct / total, p_correct / total


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("📂 Loading SSIM-relabeled dataset...")
    with open(LABELED_JSON) as f:
        all_samples = json.load(f)
    print(f"   Loaded {len(all_samples):,} samples")

    train_s, temp_s = train_test_split(all_samples, test_size=0.30, random_state=42)
    val_s, test_s   = train_test_split(temp_s, test_size=0.33, random_state=42)
    
    with open(os.path.join(CKPT_DIR, 'test_samples_ssim.json'), 'w') as f:
        json.dump(test_s, f, indent=2)
    print(f"   Split → Train: {len(train_s):,} | Val: {len(val_s):,} | Test: {len(test_s):,}")

    qualities = [s['quality'] for s in train_s]
    counts = Counter(qualities)
    total = len(qualities)
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    weight_tensor = torch.tensor([weights[i] for i in range(3)], dtype=torch.float32).to(DEVICE)
    print(f"   Class Weights: {weight_tensor.tolist()}")

    # ⚡ OPTIMIZED DATALOADERS
    train_dl = DataLoader(SSIMDataset(train_s, train_tf), batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                          prefetch_factor=2, drop_last=True)  # ← NEW: prefetch
    val_dl   = DataLoader(SSIMDataset(val_s, val_tf), batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
                          prefetch_factor=2)

    model = KMARMultiTask().to(DEVICE)
    if os.path.exists(BEST_CKPT):
        print(f"\n🔄 Loading epoch 9 weights...")
        ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
    else:
        print("\n⚠️ No checkpoint found.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion_q = nn.CrossEntropyLoss(weight=weight_tensor)
    criterion_p = nn.CrossEntropyLoss()

    start_epoch = 0
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_q_acc': [], 'val_q_acc': [], 'lr': []}
    epochs_no_improve = 0

    print(f"\n🚀 Fine-tuning (LR={LR}, Workers={NUM_WORKERS})\n")
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        epoch_start = time.time()
        
        t_loss, t_qacc, t_pacc = train_epoch(model, train_dl, optimizer, scaler, criterion_q, criterion_p, DEVICE)
        v_loss, v_qacc, v_pacc = val_epoch(model, val_dl, criterion_q, criterion_p, DEVICE)
        
        epoch_time = time.time() - epoch_start
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train → Loss: {t_loss:.4f} | Quality: {t_qacc:.3f} | Plane: {t_pacc:.3f} | Time: {epoch_time/60:.1f}m")
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
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_loss': best_loss}, 
                       os.path.join(CKPT_DIR, 'best_model_ssim.pt'))
            print(f"  ✅ Best model saved (val_loss={best_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break

    print("\n🎉 Phase 3 complete!")
    print(f"   Best val loss: {best_loss:.4f}")