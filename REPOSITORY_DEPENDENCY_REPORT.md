# CDCo-MambaNet V4 — Repository Dependency Report

**Author:** Senior Deep Learning Research Engineer  
**Date:** November 11, 2026  
**Environment Baseline:** Python 3.10.9 (64-bit AMD64) on Windows 11  

---

## 1. Core Dependency Matrix

| Package Name | Tested Version | Minimum Required | Role in Project | Verification Status |
| :--- | :---: | :---: | :--- | :---: |
| **Python** | `3.10.9` | `3.10.0` | Execution Runtime | **VERIFIED** |
| **torch** | `2.7.1+cu118` | `2.0.0` | Deep learning framework, autograd, AMP | **VERIFIED** |
| **torchvision** | `0.22.1+cu118` | `0.15.0` | Vision utilities | **VERIFIED** |
| **torchaudio** | `2.7.1+cu118` | `2.0.0` | PyTorch audio subsystem | **VERIFIED** |
| **numpy** | `2.2.6` | `1.24.0` | Array manipulation, memmaps, metrics | **VERIFIED** |
| **scipy** | `1.15.3` | `1.10.0` | Scientific computing, statistical tests | **VERIFIED** |
| **pandas** | `2.3.3` | `2.0.0` | Phenotype cohorts, metadata, CSV reports | **VERIFIED** |
| **scikit-learn**| `1.7.2` | `1.2.0` | Stratified splits, ROC-AUC, classification metrics | **VERIFIED** |
| **hydra-core** | `1.3.8` | `1.3.2` | Hierarchical configuration management | **VERIFIED** |
| **omegaconf** | `2.3.1` | `2.3.0` | Dynamic configuration dictionary handling | **VERIFIED** |
| **einops** | `0.8.2` | `0.7.0` | Tensor restructuring in Mamba fusion blocks | **VERIFIED** |
| **timm** | `1.0.30` | `0.9.0` | Deep learning models and layer blocks | **VERIFIED** |
| **matplotlib** | `3.10.8` | `3.7.0` | Visualizations, training curves, ROC plots | **VERIFIED** |
| **networkx** | `3.4.2` | `3.0` | Graph representation for connectome matrices | **VERIFIED** |
| **SimpleITK** | `2.5.6` | `2.2.0` | 3D NIfTI preprocessing (N4 bias, registration) | **VERIFIED** |
| **nilearn** | `0.14.1` | `0.10.0` | Neuroimaging fMRI ROI extraction (AAL-116) | **VERIFIED** |
| **nibabel** | `5.3.3` | `5.0.0` | Neuroimaging NIfTI I/O | **VERIFIED** |
| **wandb** | `0.19.x` (Opt) | `0.15.0` | Cloud logging (gracefully disabled if absent) | **VERIFIED** |

---

## 2. PyTorch & CUDA Compatibility Matrix

| Environment Configuration | Supported? | Performance Characteristics | Setup Command |
| :--- | :---: | :--- | :--- |
| **CUDA 11.8 (Default)** | **YES (Primary)** | Optimal performance on RTX 20/30/40 & GTX series | `pip install torch==2.x.x+cu118 ...` |
| **CUDA 12.1 / 12.4** | **YES** | Recommended for newer RTX 4000 series & Ada GPUs | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **CPU-Only** | **PARTIAL** | Inference and `tools/demo.py` work cleanly; training requires removing `.cuda()` | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| **Apple Silicon (MPS)** | **PARTIAL** | Inference runs; requires map_location='mps' | Default torch pip wheel |

---

## 3. Mamba / State-Space Model Implementation

A common concern in State Space Model (SSM) research is whether the compiled `mamba-ssm` CUDA C++ extension is strictly required.

- **CDCo-MambaNet V4 Architecture:**
  The Mamba fusion block in `models/CDCo_MambaNet_V4/encoder/mamba_fuse_v4.py` utilizes a **pure-PyTorch fallback implementation** of the selective scan mechanism.
- **Portability Benefit:**
  Users do **not** need to compile CUDA C++ kernels, install Microsoft Visual C++ Build Tools, or install `mamba-ssm`. The model runs out of the box on standard PyTorch across Windows and Linux.

---

## 4. Standard Library Dependencies

The repository leverages standard, stable Python libraries with zero external system-level C bindings:
- `os`, `sys`, `pathlib.Path`: Cross-platform filesystem navigation.
- `ntpath`: Robust filename handling across forward/backward slashes.
- `json`: Checkpoint metadata and exact splits serialization.
- `shutil`: Checkpoint copying and disk space verification.
- `time`, `datetime`: Timestamp generation for unique run IDs.
- `urllib.request`: S3 dataset fetching and demo weight downloads.
