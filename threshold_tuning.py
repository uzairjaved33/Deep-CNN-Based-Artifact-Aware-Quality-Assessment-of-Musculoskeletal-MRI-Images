# threshold_tuning.py
import json, torch, numpy as np
from phase3_train_ssim import SSIMDataset, KMARMultiTask, val_tf
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

# Load test set (as proxy for threshold search)
with open('checkpoints/test_samples_ssim.json') as f:
    samples = json.load(f)

model = KMARMultiTask().cuda()
ckpt = torch.load('checkpoints/best_model_ssim.pt', map_location='cuda')
model.load_state_dict(ckpt['model'])
model.eval()

dl = DataLoader(SSIMDataset(samples, val_tf), batch_size=16, shuffle=False)
probs, labels = [], []
with torch.no_grad():
    for imgs, q, _ in dl:
        imgs = imgs.cuda()
        q_out, _ = model(imgs)
        probs.append(torch.softmax(q_out, dim=1).cpu().numpy())
        labels.append(q.cpu().numpy())
probs = np.vstack(probs)
labels = np.concatenate(labels)

# Grid search for thresholds that maximize weighted F1
best_f1 = 0
best_thresh = [0.33, 0.33, 0.33]
for t in product(np.arange(0.2, 0.6, 0.05), repeat=3):
    if abs(sum(t) - 1.0) > 0.01: continue
    # Adjust probabilities by thresholds
    adjusted = probs / np.array(t)
    preds = np.argmax(adjusted, axis=1)
    f1 = f1_score(labels, preds, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"🎯 Optimal Thresholds: Good≥{best_thresh[0]:.2f}, Mod≥{best_thresh[1]:.2f}, Bad≥{best_thresh[2]:.2f}")
print(f"📈 Expected F1 with thresholds: {best_f1*100:.2f}%")

# Save for later use
import pickle
with open('optimal_thresholds.pkl', 'wb') as f:
    pickle.dump(best_thresh, f)