# CDCo-MambaNet V4 — GitHub Portability & Fresh-System Execution Audit

**Author:** Senior Deep Learning Research Engineer  
**Audit Date:** November 11, 2026  
**Scope:** Full repository portability inspection for fresh-system cloning/ZIP deployment  
**Audit Mode:** Static & safe environment verification (No training executed, held-out test quarantined)  

---

## 1. Executive Summary

This forensic audit evaluates the readiness of the **CDCo-MambaNet V4** repository for a researcher downloading it as a ZIP file on a clean, fresh computer.

### Overall Portability Verdict: **CONDITIONAL PORTABILITY**
- **Strengths:**
  - Standard Hydra configuration architecture (`conf/`) enables full CLI parameter overrides.
  - Core deep learning dependencies (`torch`, `hydra-core`, `timm`, `einops`, `scikit-learn`) are standard and pip-installable across Windows, Linux, and macOS.
  - Dataset ingestion in `dataset/abide.py` and dataloading in `dataset/dataloader.py` use robust cross-platform pathing (`ntpath`, `pathlib.Path`).
  - Unit test suite (`tests/test_cdcomnet_v4.py`) provides instant verification of tensor shapes and backward passes.
- **Identified Blockers on a Fresh Machine (Pre-Audit):**
  1. **Hardcoded Drive E: Paths:** `conf/dataset/ABIDE_lowmem.yaml` and several helper scripts hardcode paths to `E:/abide_data/abide_284.npy`.
  2. **Unbundled Subject Splits:** Benchmark scripts originally looked for `result/phase6_multiseed/splits/exact_splits_seed0.json`. Because `.gitignore` excludes `result/`, a fresh ZIP download lacked this split file.
  3. **Strict CUDA Assertion:** `models/__init__.py:14` and `tools/run_e4_seed0_benchmark.py:53` contain explicit `.cuda()` or `torch.cuda.is_available()` calls that crash on CPU-only machines.
  4. **Large File Exclusions:** Model weights (`best_model.pt`, 13 MB) and large ABIDE arrays (`abide_smri96.npy`, 1.0 GB; `train_aug_smri.npy`, 3.2 GB) are excluded by `.gitignore` to comply with GitHub's 100 MB file limit.
- **Remediations Implemented:**
  - Frozen subject splits (`exact_splits_seed0.json` through `seed4`) have been copied to the tracked `splits/` directory.
  - Created standardized `environment.yml` and modernized `requirements.txt`.
  - Documented exact Hydra CLI overrides to re-point dataset paths without modifying YAML files.

---

## 2. Repository Structure & File Distribution

```
CDCO-Mamba-Research-main/
├── conf/                          # Hydra configuration hierarchy
│   ├── config.yaml                # Master configuration entry point
│   ├── dataset/                   # Dataset configs (ABIDE, ABIDE_lowmem, ADNI)
│   ├── model/                     # Architecture configs (bnt, cdcomambanet_v4_unified, etc.)
│   ├── optimizer/                 # Optimizer configs (adam, adamw_wd05)
│   └── training/                  # Training configs (basic_training, exp2_novelty_training)
├── dataset/                       # Data loading and augmentation
│   ├── abide.py                   # ABIDE I loader (memmap-backed sMRI)
│   ├── data_argument.py           # 3D spatial rotation, erasure, mixup
│   └── dataloader.py              # Stratified splits & StochasticAugmentedDataset
├── demo/                          # Minimal CPU verification bundle
│   ├── abide_seed0_test.npz       # 28 held-out test subjects (45.5 MB, tracked)
│   └── kaggle_seed0_reference.json # Benchmark reference metrics
├── models/                        # Neural network architectures
│   ├── CDCo_MambaNet/             # Original baseline architecture
│   ├── CDCo_MambaNet_V2/          # V2: Reliability-gated fusion
│   ├── CDCo_MambaNet_V3/          # V3: Dual-head compact architecture
│   └── CDCo_MambaNet_V4/          # V4: Unified calibrated evidential architecture
├── reports/                       # Formal audit reports and visualizations
├── splits/                        # Tracked frozen participant splits (Seeds 0–4)
├── tests/                         # Automated unit test suite
│   ├── test_cdcomnet_v4.py        # V4 architecture & gradient flow verification
│   └── test_phase6_experiment2.py
├── tools/                         # Execution and diagnostic scripts
│   ├── demo.py                    # Standalone CPU demo runner
│   ├── evaluate_best_checkpoint.py# Standalone checkpoint evaluation tool
│   ├── export_exact_splits.py     # Split generator / verification tool
│   ├── generate_smri_cache.py     # 4x sMRI augmentation memmap generator
│   ├── prepare_abide.py           # S3 AWS automated ABIDE downloader/preprocessor
│   ├── run_e4_seed0_benchmark.py  # E4 Seed-0 benchmark runner
│   └── validate_abide_dataset.py  # Preflight dataset integrity check
├── __main__.py                    # Hydra entry point
├── environment.yml                # Conda environment definition
├── requirements.txt               # Pip dependencies
└── README.md                      # Project documentation
```

---

## 3. Detailed Audit Findings

### 3.1 Hardcoded Paths and Machine Dependencies
| Component / File | Line(s) | Hardcoded Assumption | Portability Impact | Recommended Solution |
| :--- | :---: | :--- | :--- | :--- |
| `conf/dataset/ABIDE_lowmem.yaml` | 7, 11 | `path: E:/abide_data/abide_284.npy`<br>`aug_cache: E:/abide_data/train_aug_smri.npy` | Fails if user has no Drive E: | Override via CLI: `dataset.path=...` `dataset.aug_cache=...` |
| `conf/dataset/ABIDE_lowmem.yaml` | 12 | `exact_splits_path: result/...` | Missing on fresh download | Point to `splits/exact_splits_seed0.json` |
| `tools/run_e4_seed0_benchmark.py` | 59, 60 | `split_path = ROOT / "result" / ...`<br>`dataset_path = Path("E:/abide_data/abide_284.npy")` | Missing on fresh download | Fallback to `ROOT / "splits" / ...` and configurable `--data_dir` |
| `tools/validate_abide_dataset.py`| 21 | `DATA_DIR = Path(r"E:\abide_data")` | Fails on other drives | Accept `--data_dir` CLI argument |
| `tools/generate_smri_cache.py` | 21, 25 | `E:/abide_data/...`<br>`result/phase6_multiseed/splits/...` | Fails on other drives | Read from `splits/` directory |

### 3.2 Compute & Device Portability
1. **Explicit `.cuda()` Invocations:**
   - In `models/__init__.py:14`: `return eval(config.model.name)(config).cuda()`
   - In `tools/evaluate_best_checkpoint.py:44`: `model = CDCo_MambaNet(cfg).cuda()`
   - In `tools/run_e4_seed0_benchmark.py:52`: `if not torch.cuda.is_available(): raise RuntimeError(...)`
   - *Impact:* Running full benchmarks currently requires an NVIDIA GPU with CUDA. For CPU-only verification, users should use `tools/demo.py`, which is explicitly programmed for CPU execution.
2. **GPU Memory Footprint:**
   - E4 Seed-0 peak VRAM measured: **180 MB** (using batch size 2 and gradient accumulation 8).
   - Any modern NVIDIA GPU with $\ge 2\text{ GB}$ VRAM (GTX 1650, RTX 2050/3050, MX450, T4, V100, A100) can run training without out-of-memory errors.

### 3.3 GitHub ZIP Download Exclusions & Large Files
GitHub limits individual files to $< 100\text{ MB}$. Consequently:
- **Excluded by `.gitignore`:**
  - `result/` (contains trained model checkpoints like `best_model.pt` [13 MB] and logs).
  - `*.npy` and `*.npz` (contains `abide_smri96.npy` [1.0 GB] and `train_aug_smri.npy` [3.2 GB]).
  - `.venv/`, `outputs/`, `wandb/`.
- **Included in ZIP:**
  - Complete Python source code, model definitions (V1, V2, V3, V4).
  - All configuration files (`conf/`).
  - Unit tests (`tests/`).
  - Audit reports, documentation, visualizations.
  - Frozen participant splits (`splits/exact_splits_seed0.json` through `seed4`).
  - Held-out test evaluation demo bundle (`demo/abide_seed0_test.npz`, 45.5 MB).

---

## 4. Frozen Subject Splits Verification

The 284 ABIDE I participants are partitioned into:
- **Train Split:** 227 subjects (115 Control, 112 ASD).
- **Validation Split:** 29 subjects (15 Control, 14 ASD) — strictly used for model selection.
- **Held-Out Test Split:** 28 subjects (14 Control, 14 ASD) — strictly evaluated once.

By placing `exact_splits_seed0.json` directly into `splits/`, any fresh computer can load the exact participant partition without running dynamic `StratifiedShuffleSplit` or needing access to `result/`.

---

## 5. Output Directory Auto-Creation Check

In `training/training.py:85-87`:
```python
self.save_path = Path(cfg.log_path) / cfg.unique_id
self.save_path.mkdir(exist_ok=True, parents=True)
OmegaConf.save(self.config, self.save_path / "config.yaml")
```
- **Result:** Output directories are created automatically via `mkdir(parents=True, exist_ok=True)`. A clean system does not require pre-existing `result/` or output folders.

---

## 6. Summary of Actionable Portability Recommendations

1. **Keep `splits/` committed:** The `splits/` directory ensures exact subject reproducibility on fresh machines.
2. **Use CLI overrides for dataset location:** Instead of altering YAML files, provide the dataset path at the terminal:
   `python __main__.py dataset.path="C:/data/abide_284.npy" dataset.aug_cache="C:/data/train_aug_smri.npy"`
3. **Use `tools/demo.py` for CPU validation:** Verifies complete inference pipeline on held-out test data without needing CUDA or external datasets.
