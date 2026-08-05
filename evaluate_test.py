"""
Evaluate final model on locked test set
Run AFTER training completes
"""
import os, json, torch, numpy as np, pandas as pd, nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import timm, torch.nn as nn
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────
BASE = r'D:\KMAR-50K\KMAR-50K'
CKPT_DIR = os.path.join(BASE, 'checkpoints')
TEST_SAMPLES_FILE = os.path.join(CKPT_DIR, 'test_samples.json')
BEST_CKPT = os.path.join(CKPT_DIR, 'best_model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
IMG_SIZE = 224

# ── Dataset (same as training) ────────────────────────────────
class KMARDataset:  # Simplified for eval
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        proxy = nib.load(s['path'])
        img = np.asarray(proxy.dataobj[:, :, s['slice']]).astype(np.float32)
        sl = (img - img.min()) / (img.max() - img.min() + 1e-8)
        sl = np.stack([sl, sl, sl], axis=0)
        sl = torch.tensor(sl, dtype=torch.float32)
        sl = self.transform(sl)
        return sl, torch.tensor(s['quality']), torch.tensor(s['plane'])

# ── Model (must match training architecture) ──────────────────
class KMARMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.head_quality = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 3))
        self.head_plane = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3))
    def forward(self, x):
        f = self.backbone(x)
        return self.head_quality(f), self.head_plane(f)

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load test samples
    with open(TEST_SAMPLES_FILE) as f:
        test_samples = json.load(f)
    print(f"Loaded {len(test_samples)} test samples")
    
    # Load model
    model = KMARMultiTask().to(DEVICE)
    ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Evaluate
    dl = DataLoader(KMARDataset(test_samples), batch_size=BATCH_SIZE, shuffle=False)
    all_q_true, all_q_pred = [], []
    
    with torch.no_grad():
        for imgs, quality, _ in tqdm(dl, desc="Evaluating"):
            imgs, quality = imgs.to(DEVICE), quality.to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                q_out, _ = model(imgs)
            all_q_true.extend(quality.cpu().numpy())
            all_q_pred.extend(q_out.argmax(1).cpu().numpy())
    
    # Report
    print("\n" + "="*60)
    print("TEST SET RESULTS — Quality Classification")
    print("="*60)
    print(classification_report(all_q_true, all_q_pred, 
                              target_names=['Good', 'Moderate', 'Bad'], digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(all_q_true, all_q_pred))
    
    # Overall accuracy
    acc = np.mean(np.array(all_q_true) == np.array(all_q_pred))
    print(f"\n🎯 Overall Test Accuracy: {acc*100:.1f}%")
    print("="*60)
    
    # Decision guidance
    if acc >= 0.75:
        print("✅ Pipeline is strong. Consider deployment or minor refinements.")
    elif acc >= 0.65:
        print("⚠️ Moderate performance. Label noise or model capacity may be limiting.")
    else:
        print("❌ Low performance. Priority: fix label generation or artifact metric.")