"""
System Readiness Check for KMAR-50K Training
Run this on your laptop before starting any training.
"""

import os
import sys
import platform
import shutil

print("=" * 60)
print("SYSTEM READINESS CHECK")
print("=" * 60)

# ── 1. OS & Python ────────────────────────────────────────────
print("\n[1] SYSTEM")
print(f"  OS      : {platform.system()} {platform.release()}")
print(f"  Python  : {sys.version.split()[0]}")

# ── 2. RAM ────────────────────────────────────────────────────
print("\n[2] RAM")
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"  Total   : {ram.total / 1e9:.1f} GB")
    print(f"  Available: {ram.available / 1e9:.1f} GB")
    print(f"  Used    : {ram.percent}%")
    if ram.available / 1e9 < 8:
        print("  WARNING: Less than 8GB available — close other apps before training")
    else:
        print("  STATUS  : OK")
except ImportError:
    print("  psutil not installed — run: pip install psutil")

# ── 3. Disk Space ─────────────────────────────────────────────
print("\n[3] DISK SPACE")
BASE = r'D:\KMAR-50K\KMAR-50K'
drive = 'D:\\'
total, used, free = shutil.disk_usage(drive)
print(f"  Drive   : D:\\")
print(f"  Total   : {total / 1e9:.1f} GB")
print(f"  Used    : {used / 1e9:.1f} GB")
print(f"  Free    : {free / 1e9:.1f} GB")
if free / 1e9 < 20:
    print("  WARNING: Less than 20GB free — checkpoints + model need space")
else:
    print("  STATUS  : OK")

# ── 4. GPU Check ──────────────────────────────────────────────
print("\n[4] GPU")
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        vram = gpu.total_memory / 1e9
        print(f"  Name    : {gpu.name}")
        print(f"  VRAM    : {vram:.1f} GB")
        print(f"  CUDA    : {torch.version.cuda}")
        if vram < 4:
            print("  WARNING : Less than 4GB VRAM — use batch_size=2")
        elif vram < 6:
            print("  WARNING : 4GB VRAM — use batch_size=4, fp16")
        else:
            print("  STATUS  : OK")
    else:
        print("  No CUDA GPU detected — will train on CPU (very slow)")
except ImportError:
    print("  PyTorch not installed — run: pip install torch torchvision")

# ── 5. Required Libraries ─────────────────────────────────────
print("\n[5] REQUIRED LIBRARIES")
libs = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'nibabel': 'nibabel',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'sklearn': 'scikit-learn',
    'tqdm': 'tqdm',
    'timm': 'timm',
    'psutil': 'psutil',
}
missing = []
for name, pkg in libs.items():
    try:
        __import__(name)
        print(f"  {name:15s}: installed")
    except ImportError:
        print(f"  {name:15s}: MISSING")
        missing.append(pkg)

if missing:
    print(f"\n  Install missing: pip install {' '.join(missing)}")
else:
    print("\n  All libraries OK")

# ── 6. Dataset Check ──────────────────────────────────────────
print("\n[6] DATASET")
folders = {
    'ArtifactData_part1':     os.path.join(BASE, 'ArtifactData_part1'),
    'GroundTruthData_part1':  os.path.join(BASE, 'GroundTruthData_part1'),
    'TrainingCohort.csv':     os.path.join(BASE, 'TrainingCohort.csv'),
    'TestingCohort.csv':      os.path.join(BASE, 'TestingCohort.csv'),
}
for name, path in folders.items():
    exists = os.path.exists(path)
    print(f"  {name:30s}: {'OK' if exists else 'NOT FOUND'}")

# ── 7. Checkpoint Dir ─────────────────────────────────────────
print("\n[7] CHECKPOINT DIRECTORY")
ckpt_dir = os.path.join(BASE, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
print(f"  Path    : {ckpt_dir}")
print(f"  STATUS  : Ready")

# ── 8. Final Verdict ──────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)
try:
    import torch
    import psutil
    ram_ok  = psutil.virtual_memory().available / 1e9 >= 8
    gpu_ok  = torch.cuda.is_available()
    disk_ok = shutil.disk_usage('D:\\').free / 1e9 >= 20
    libs_ok = len(missing) == 0

    issues = []
    if not ram_ok:  issues.append("Low RAM")
    if not gpu_ok:  issues.append("No GPU")
    if not disk_ok: issues.append("Low Disk")
    if not libs_ok: issues.append("Missing Libraries")

    if not issues:
        print("  READY TO TRAIN")
    else:
        print(f"  ISSUES FOUND: {', '.join(issues)}")
except:
    print("  Run pip installs first then re-check")

print("=" * 60)