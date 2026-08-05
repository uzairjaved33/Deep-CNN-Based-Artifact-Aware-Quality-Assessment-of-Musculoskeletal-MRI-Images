"""
Test-Time Augmentation + Custom Thresholds
Expected accuracy: ~81.5% - 82.5%
"""
import json, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
from sklearn.metrics import classification_report, confusion_matrix
import timm, torch.nn as nn
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────
BASE = r'D:\KMAR-50K\KMAR-50K'
CKPT_DIR = r'D:\KMAR-50K\KMAR-50K\checkpoints'
TEST_JSON = r'D:\KMAR-50K\KMAR-50K\checkpoints\test_samples_ssim.json'
MODEL_PATH = r'D:\KMAR-50K\KMAR-50K\checkpoints\best_model_ssim.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
THRESHOLDS = [0.20, 0.45, 0.35]  # From previous step

# ── Dataset ───────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
        return sl, torch.tensor(s['quality'], dtype=torch.long)

# ── Model ─────────────────────────────────────────────────────
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

# ── TTA Logic ────────────────────────────────────────────────
def tta_predict(model, imgs, device):
    """Run 4 augmented views, average probabilities"""
    probs = []
    # 1. Original
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        probs.append(torch.softmax(model(imgs.to(device))[0], dim=1).cpu())
    # 2. Horizontal Flip
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        probs.append(torch.softmax(model(torch.flip(imgs, [-1]).to(device))[0], dim=1).cpu())
    # 3. Vertical Flip
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        probs.append(torch.softmax(model(torch.flip(imgs, [-2]).to(device))[0], dim=1).cpu())
    # 4. Rotate 90°
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        probs.append(torch.softmax(model(torch.rot90(imgs, 1, dims=[-2, -1]).to(device))[0], dim=1).cpu())
    return torch.mean(torch.stack(probs), dim=0)

def apply_thresholds(probabilities, thresholds):
    return np.argmax(probabilities / np.array(thresholds), axis=1)

# ── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("📂 Loading Test Set & Model...")
    with open(TEST_JSON) as f:
        test_samples = json.load(f)
    
    model = KMARMultiTask().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    dl = DataLoader(TestDataset(test_samples), batch_size=BATCH_SIZE, shuffle=False)
    all_probs, all_true = [], []
    
    print("🔄 Running TTA (4 views per sample) + Thresholds...")
    with torch.no_grad():
        for imgs, labels in tqdm(dl, desc="TTA Eval"):
            q_probs = tta_predict(model, imgs, DEVICE)
            all_probs.append(q_probs.numpy())
            all_true.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_true = np.concatenate(all_true)
    all_pred = apply_thresholds(all_probs, THRESHOLDS)
    
    print("\n" + "="*60)
    print("🏆 TTA + THRESHOLD RESULTS")
    print("="*60)
    print(classification_report(all_true, all_pred, target_names=['Good', 'Moderate', 'Bad'], digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_pred))
    
    acc = np.mean(all_true == all_pred)
    print(f"\n🎯 Overall Test Accuracy: {acc*100:.2f}%")
    print("="*60)
    if acc >= 0.81:
        print("✅ TTA worked! Next: Ensemble 3 checkpoints → target 84%+")
    else:
        print("⚠️ Modest gain. Proceed to ensemble next.")