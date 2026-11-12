# CDCo-MambaNet V4 — Google Colab Training Workflow Guide

**Target Hardware:** Google Colab Free / Pro Tier (NVIDIA Tesla T4 GPU, 15–16 GB VRAM)  
**Execution Objective:** Seamless, reproducible execution of Phase 2 controlled experiments in Google Colab  
**Status:** Prepared and verified; standing by for authorization  

---

## 1. Overview & Advantages of Running on Colab

Google Colab provides an isolated cloud environment with an **NVIDIA Tesla T4 GPU (16 GB VRAM)**. This offers distinct advantages:
1. **Zero OS Display Overhead:** Headless Linux environment has ~0 MB display manager VRAM overhead (unlike Windows DWM).
2. **Generous VRAM Headroom:** Peak VRAM for CDCo-MambaNet V4 is only **180 MB** (tensor allocation) and **~1.2 GB** total system footprint, easily fitting into the T4's 16 GB capacity.
3. **Google Drive Checkpoint Persistence:** Checkpoints and evaluation logs can be written directly to a mounted Google Drive to survive session disconnects.

---

## 2. Step-by-Step Colab Execution Workflow

### Cell 1: Mount Google Drive & Clone Repository
```python
from google.colab import drive
import os

# Mount Google Drive to persist checkpoints and logs live
drive.mount('/content/drive')

# Create research folder in Google Drive
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/CDCo_Mamba_Research_Results"
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)

# Clone repository from GitHub (handles runtime restarts safely)
REPO_DIR = "/content/CDCO-Mamba-Research"
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/poorna-19/Main-project-.git {REPO_DIR}
%cd {REPO_DIR}
!git checkout improved-fixes
```

---

### Cell 2: Install Core Dependencies
```bash
# Install PyTorch and required neuroimaging / DL libraries
!pip install -q hydra-core>=1.3.2 omegaconf>=2.3.0 timm>=1.0.0 einops>=0.8.0 scikit-learn>=1.2.0 SimpleITK nilearn nibabel wandb
```

---

### Cell 3: GPU & Environment Sanity Verification
```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
```

---

### Cell 4: Execute Zero-Leakage Synthetic Verification
```bash
# Run the 100% synthetic verification script (zero patient data accessed)
!python tools/safe_fresh_system_check.py
```
*Verifies architecture parameters ($3,226,317$), autograd, gradient clipping, and frozen split JSON integrity in $< 10$ seconds.*

---

### Cell 5: Provide ABIDE Dataset Files
The ABIDE dataset can be loaded in one of two ways:

#### Option A: Copy Preprocessed Dataset from Google Drive
```python
# Assuming you uploaded abide_284.npy and abide_smri96.npy to your Google Drive
!mkdir -p /content/data/abide
!cp /content/drive/MyDrive/abide_data/abide_284.npy /content/data/abide/
!cp /content/drive/MyDrive/abide_data/abide_smri96.npy /content/data/abide/
!cp /content/drive/MyDrive/abide_data/cohort.csv /content/data/abide/
```

#### Option B: Build Directly from Public S3 Sources (Takes ~40 mins)
```bash
!python tools/prepare_abide.py --root /content/data/abide --workers 4
```

---

### Cell 6: Generate 4x sMRI Augmentation Memmap Cache
```bash
!python tools/generate_smri_cache.py \
    --seed 0 \
    --dataset_path /content/data/abide/abide_284.npy \
    --out_cache /content/data/abide/train_aug_smri.npy \
    --split_path splits/exact_splits_seed0.json
```

---

### Cell 7: Launch Phase 2 Experiment E1 (Upon Explicit Authorization)
```bash
%env WANDB_MODE=disabled

# Execute E1: Scheduler Cadence Benchmark (30 epochs, 171 warmup steps, 1710 total steps, 57 updates/epoch, accum 8, val_loss selection)
python __main__.py \
    model=cdcomambanet_v4_unified \
    dataset=ABIDE_lowmem \
    optimizer=adamw_e1_cos \
    training=e1_training \
    dataset.path="/content/data/abide/abide_284.npy" \
    dataset.aug_cache="/content/data/abide/train_aug_smri.npy" \
    dataset.exact_splits_path="splits/exact_splits_seed0.json" \
    repeat_time=1 seed=0 \
    log_path="/content/drive/MyDrive/CDCo_Mamba_Research_Results/phase2_experiments/E1_lr_cosine"
```

---

### Cell 8: Resume Training from Checkpoint (In Case of Colab Free Disconnect)
```bash
%env WANDB_MODE=disabled

python __main__.py \
    model=cdcomambanet_v4_unified \
    dataset=ABIDE_lowmem \
    optimizer=adamw_e1_cos \
    training=e1_training \
    dataset.path="/content/data/abide/abide_284.npy" \
    dataset.aug_cache="/content/data/abide/train_aug_smri.npy" \
    dataset.exact_splits_path="splits/exact_splits_seed0.json" \
    training.resume_checkpoint="/content/drive/MyDrive/CDCo_Mamba_Research_Results/phase2_experiments/E1_lr_cosine/seed_0/last_checkpoint.pt" \
    repeat_time=1 seed=0 \
    log_path="/content/drive/MyDrive/CDCo_Mamba_Research_Results/phase2_experiments/E1_lr_cosine"
```

---

### Cell 9: Verify Google Drive Deliverables
```bash
!ls -lh /content/drive/MyDrive/CDCo_Mamba_Research_Results/phase2_experiments/E1_lr_cosine/seed_0/
```
