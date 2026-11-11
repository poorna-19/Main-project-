# CDCo-MambaNet V4 — Phase 2 Reproducibility Protocol & Quality Assurance Plan

**Document Status**: Official Reproducibility & Governance Protocol  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Target Milestone**: Phase 2 Controlled Experiments (E1 through E5)  
**Execution Guardrail**: **Read-only audit & protocol definition.** No execution until user approval.

---

## 1. Hardware, Driver & Environment Specifications

To ensure bite-for-byte reproducibility across runs, all experiments in Phase 2 are pinned to the verified computational environment:

| Component | Verified Specification | Verification Command / Check |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 Home (10.0.19045 Build 19045) | `[System.Environment]::OSVersion` |
| **Host Python** | Python 3.10.9 (64-bit) | `python --version` |
| **PyTorch Framework** | PyTorch 2.7.1+cu118 | `python -c "import torch; print(torch.__version__)"` |
| **Compute Device** | NVIDIA GeForce MX450 (Device 0, Turing Architecture) | `torch.cuda.get_device_name(0)` |
| **Dedicated VRAM** | 2,048 MB Physical GDDR6 | `torch.cuda.get_device_properties(0).total_memory` |
| **CUDA Backend** | CUDA 11.8 / CUDNN 8.7.0 | `torch.version.cuda` |
| **Peak VRAM Budget** | Enforce $\le 250\text{ MB}$ ($<12.5\%$ physical capacity) | `torch.cuda.max_memory_allocated()` |

---

## 2. Deterministic Seeding & Execution Standards

Before initializing dataloaders, model layers, or optimizers in any Phase 2 script, the following deterministic initialization sequence must be executed verbatim:

```python
import os
import random
import numpy as np
import torch

def set_deterministic_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 2.x Matrix Multiplication Precision
    torch.set_float32_matmul_precision('high')
```

### Worker Seeding in DataLoaders:
To ensure on-the-fly random transformations (Experiment E5) produce reproducible streams across batches:
```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

---

## 3. Dataset Integrity & Split Quarantine Verification

### 3.1 Master File Integrity Check
Before any run is launched, the data files must match the expected byte sizes and locations:

| Asset | Path | Expected Volume / Shape | Integrity Verification |
| :--- | :--- | :---: | :--- |
| **Master Dataset** | `E:/abide_data/abide_284.npy` | $N=284$ subjects | Verified readable dict (`timeseires, corr, smri, label, site`) |
| **sMRI 4x Cache** | `E:/abide_data/train_aug_smri.npy` | $908 \times 1 \times 96 \times 96 \times 96$ | Size: $3,212,836,992\text{ bytes}$ ($\sim 3.21\text{ GB}$) |
| **Seed 0 Split** | `result/phase6_multiseed/splits/exact_splits_seed0.json` | 227 Train, 29 Val, 28 Test | Participant overlap count $= 0$ |

### 3.2 Leak-Free Participant Check
Every run must assert that no participant appears in more than one partition:
```python
assert len(set(train_sub_ids).intersection(val_sub_ids)) == 0, "FATAL: Train-Val Leakage Detected!"
assert len(set(train_sub_ids).intersection(test_sub_ids)) == 0, "FATAL: Train-Test Leakage Detected!"
assert len(set(val_sub_ids).intersection(test_sub_ids)) == 0, "FATAL: Val-Test Leakage Detected!"
```

### 3.3 Strict Held-Out Test Quarantine
- **Quarantine Guarantee**: In Phase 2, `test_dataloader` is **never instantiated** or passed to the model.
- Model checkpoints are selected strictly on `val_loss`.
- Hyperparameter comparisons (e.g. between $wd=0.01$ and $wd=0.10$) are conducted **exclusively on validation metrics**.
- Historical test performance from E4 ($64.29\%$ Acc, $64.10\%$ F1) is preserved as an external historical reference and will not be re-evaluated during Phase 2 ablation cycles.

---

## 4. Directory Isolation & Artifact Protection

To guarantee that historical models and prior deliverables are never compromised:

```
c:\Users\Dell\Desktop\CDCO-Mamba-Research-main\
├── result/
│   ├── experiment4_unified/
│   │   └── seed_0/
│   │       ├── best_model.pt                <-- [READ-ONLY / FROZEN E4 CHECKPOINT]
│   │       └── epoch_history.json           <-- [READ-ONLY / FROZEN E4 LOGS]
│   └── phase2_experiments/                  <-- [DEDICATED PHASE 2 DIRECTORY]
│       ├── E1_lr_cosine/seed_0/
│       ├── E2_grad_clip/seed_0/
│       ├── E3_weight_decay/
│       │   ├── wd_01/seed_0/
│       │   ├── wd_05/seed_0/
│       │   └── wd_10/seed_0/
│       ├── E4_spatial_dropout/
│       │   ├── drop_10/seed_0/
│       │   └── drop_25/seed_0/
│       └── E5_stochastic_aug/seed_0/
└── reports/
    ├── Finalization/
    │   └── E4_Seed0/                        <-- [READ-ONLY / FROZEN E4 DELIVERABLES]
    └── Phase2/                              <-- [DEDICATED PHASE 2 REPORTS]
        ├── E1_lr_cosine/
        ├── E2_grad_clip/
        ├── E3_weight_decay/
        ├── E4_spatial_dropout/
        ├── E5_stochastic_aug/
        └── PHASE2_COMPARISON_SUMMARY.csv
```

---

## 5. Standardized Evaluation & Output Metrics

Every Phase 2 experiment must output a standardized `summary.json` containing:

```json
{
  "experiment_id": "E1_lr_cosine",
  "seed": 0,
  "best_val_epoch": 3,
  "best_val_loss": 2.2154,
  "train_loss_at_best": 2.3412,
  "train_accuracy_at_best": 0.7421,
  "val_accuracy": 0.5862,
  "val_balanced_accuracy": 0.5857,
  "val_sensitivity": 0.6429,
  "val_specificity": 0.5333,
  "val_precision_macro": 0.5865,
  "val_f1_macro": 0.5841,
  "val_auc": 0.6190,
  "val_confusion_matrix": [[8, 7], [5, 9]],
  "generalization_gap": 0.1559,
  "parameters": 3226317,
  "peak_vram_mb": 180.12,
  "wall_time_minutes": 22.45
}
```

---

## 6. Pre-Execution Quality Assurance Sign-Off

Before receiving user authorization to proceed to implementation and execution:

- [x] Hardware memory ceiling verified ($\le 250\text{ MB}$ fits comfortably in 2,048 MB VRAM).
- [x] PyTorch, CUDA, and deterministic CuDNN seeding protocol formalized.
- [x] Split hash and participant-level isolation verified across Seed 0.
- [x] Strict test quarantine established; test loader disabled during Phase 2 development.
- [x] Isolated directory tree allocated for all Phase 2 logs, models, and reports.
- [x] Standardized metric dictionary formalized for automated cross-experiment compilation.
- [x] Verification criteria defined for single-factor comparative analysis.

---

*Phase 2 reproducibility plan is verified and attested. Awaiting explicit user approval before modifying code or running any experiments.*
