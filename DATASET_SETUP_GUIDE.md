# CDCo-MambaNet V4 — Dataset Setup Guide

**Cohort:** ABIDE I Preprocessed Connectomes & 3D sMRI ($N=284$ participants)  
**Task:** Binary Classification — Autism Spectrum Disorder (ASD: 140) vs. Neurotypical Control (TC: 144)  

---

## 1. Required Files and Directory Structure

When running full training on a new computer, create a dataset folder on an internal SSD (e.g., `D:\abide_data` or `C:\data\abide_data`):

```
abide_data/
├── abide_284.npy            # Master dictionary containing fMRI, connectomes, labels (30.45 MB)
├── abide_smri96.npy         # Raw 3D structural MRI array (1,005.06 MB / ~1.0 GB)
├── cohort.csv               # Subject demographics, sites, age, sex (130 KB)
└── train_aug_smri.npy       # Disk-backed 4x training sMRI memmap cache (3,213.36 MB / ~3.2 GB)
```

> [!IMPORTANT]
> Both `abide_284.npy` and `abide_smri96.npy` MUST be located in the **same directory**. The loader (`dataset/abide.py`) dynamically reads `abide_smri96.npy` relative to the path of `abide_284.npy`.

---

## 2. Expected Tensor Specifications & Dictionary Keys

`abide_284.npy` is a Python dictionary containing the following verified keys:

| Key Name | Array Shape | Data Type | Description |
| :--- | :--- | :---: | :--- |
| `timeseires` | `(284, 116, 115)` | `float32` | AAL-116 ROI BOLD timeseries truncated to 115 timepoints |
| `corr` | `(284, 116, 116)` | `float32` | Pearson correlation connectome adjacency matrices |
| `label` | `(284,)` | `int64` | Binary diagnosis: `0` = Control (144), `1` = ASD (140) |
| `smri` | `"abide_smri96.npy"` | `str` | Relative filename pointing to the 3D volume memmap |
| `site` | `(284,)` | `str` / `object`| Scanner acquisition site identifier (17 total sites) |
| `file_id` | `(284,)` | `str` / `object`| Standard ABIDE pipeline subject file identifier |

When loaded, `abide_smri96.npy` provides 3D brain volumes of shape:
$$\text{sMRI Volume Shape} = (284, 96, 96, 96) \quad (\text{float32})$$

---

## 3. How to Configure Dataset Paths on a Fresh Computer

You do **not** need to modify repository source code or default configuration files. You can specify the dataset location on your new computer using Hydra command-line overrides.

### Example: Dataset located at `D:\abide_data`

```powershell
python __main__.py dataset=ABIDE_lowmem dataset.path="D:/abide_data/abide_284.npy" dataset.aug_cache="D:/abide_data/train_aug_smri.npy" dataset.exact_splits_path="splits/exact_splits_seed0.json"
```

---

## 4. Validating Your Dataset

The repository provides a fast, read-only diagnostic script to verify file integrity, checksums, tensor shapes, and NaN-free data:

```powershell
python tools/validate_abide_dataset.py
```
*(If your dataset is not in `E:\abide_data`, edit line 21 of `tools/validate_abide_dataset.py` to point to your folder).*

**Expected Output:**
```
=== Validating ABIDE Dataset in D:\abide_data ===
[OK] Found abide_284.npy (30.45 MB)
[OK] Found abide_smri96.npy (1005.06 MB)
  Key 'timeseires': present
  Key 'corr': present
  Key 'label': present
  Key 'site': present
  Key 'file_id': present
  Key 'smri': present
[OK] All required keys present.
[OK] Subject count: 284
[OK] Class balance: 144 Control, 140 ASD
[OK] Zero NaN or Inf values detected.
```

---

## 5. Generating the 4x sMRI Augmentation Cache

To prevent loading 3.2 GB of augmented 3D brain volumes into RAM during training, the pipeline uses a memory-mapped disk cache (`train_aug_smri.npy`).

If you have downloaded `abide_284.npy` and `abide_smri96.npy` but do not have `train_aug_smri.npy`, generate it in ~1 minute using:

```powershell
python tools/generate_smri_cache.py
```
*(Optionally pass `--dataset_path` and `--out_cache` for custom directories).*
