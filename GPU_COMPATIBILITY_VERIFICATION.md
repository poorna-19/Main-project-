# CDCo-MambaNet V4 — GPU & Hardware Memory Compatibility Verification

**Author:** Senior Deep Learning Research Engineer  
**Date:** November 11, 2026  
**Hardware Audit Objective:** Rigorous deconstruction of VRAM, RAM, Storage, and CUDA compatibility  

---

## 1. Forensic Deconstruction of the "180 MB VRAM" Measurement

In Experiment E4 documentation, peak VRAM is recorded as **180.0 MB**. It is essential to specify the exact conditions under which this metric was obtained to avoid misleading hardware claims.

### Exact Measurement Conditions
- **Model Architecture:** `CDCo_MambaNet_V4` ($3,226,317$ parameters / 12.9 MB model weights).
- **Precision:** Single precision (`float32`). AMP disabled.
- **Batch Size:** Mini-batch size $B = 2$.
- **Gradient Accumulation:** $A = 8$ (effective batch size $16$).
- **Input Dimensions per Mini-Batch:**
  - fMRI BOLD Timeseries: `(2, 116, 115)` ($372\text{ KB}$)
  - Pearson Correlation Matrix: `(2, 116, 116)` ($107\text{ KB}$)
  - 3D sMRI Volume: `(2, 1, 96, 96, 96)` ($7.08\text{ MB}$)
- **Measurement Metric:** `torch.cuda.max_memory_allocated(0) / (1024**2)`.
- **Execution Stage:** Peak activation and intermediate state memory during the forward-backward pass of a single micro-batch.

---

## 2. Why "Any 2 GB GPU" Cannot Be Guaranteed Without Qualification

The **180 MB** metric reflects **pure PyTorch tensor allocation**. It does **not** represent total device VRAM consumption. The true VRAM demand on a host machine is:

$$\text{Total VRAM Consumed} = V_{\text{OS/DWM}} + V_{\text{CUDA Driver}} + V_{\text{PyTorch Runtime}} + V_{\text{Tensor Alloc}} + V_{\text{Reserved Cache}}$$

### Non-Tensor VRAM Overhead Breakdown
1. **Operating System Display Overhead ($V_{\text{OS/DWM}}$):**
   On Windows 10/11, the Desktop Window Manager (DWM) and display driver consume **200 MB to 600 MB** of VRAM depending on screen resolution (1080p vs 4K) and external monitor connections.
2. **CUDA Context & Runtime Initialization ($V_{\text{CUDA Driver}}$):**
   Initializing the PyTorch CUDA context (`torch.cuda.init()`) automatically allocates **400 MB to 750 MB** of VRAM on modern CUDA 11.8/12.1 runtimes before a single tensor is created.
3. **PyTorch Caching Allocator ($V_{\text{Reserved Cache}}$):**
   To prevent frequent system calls, PyTorch reserves memory blocks. Peak reserved memory typically reaches **300 MB to 500 MB**.

### Total Real-World GPU Footprint
$$\text{Total Realistic Footprint} \approx 400\text{ MB (OS)} + 500\text{ MB (CUDA Context)} + 180\text{ MB (Tensors)} + 200\text{ MB (Cache)} \approx \mathbf{1,280\text{ MB}}$$

### Technical Verdict:
- On a clean 2,048 MB GPU (such as the development NVIDIA GeForce MX450), **training succeeds** because $1,280\text{ MB} < 2,048\text{ MB}$ (leaving ~760 MB buffer).
- However, if a user's 2 GB GPU is driving a 4K display, running GPU-accelerated browsers, or handling background video decoding, available VRAM may drop below 1.2 GB, causing a `CUDA out of memory` runtime exception.
- **Rule:** A 2 GB GPU is viable **only under low display load**. The recommended minimum for stable, uninterrupted training is **4 GB VRAM**.

---

## 3. Comprehensive System Resource Requirements

| Resource Type | Minimum Requirement | Recommended Specification | Function & Impact |
| :--- | :--- | :--- | :--- |
| **GPU Dedicated VRAM** | 2,048 MB (Clean OS state) | 4,096 MB – 8,192 MB | Model weights, micro-batch activations, Mamba scan state |
| **CUDA Version** | CUDA 11.8 | CUDA 11.8 or 12.1 | PyTorch backend compatibility |
| **System RAM** | 8 GB | 16 GB – 32 GB | OS overhead, dataloader multi-processing, dataset loading |
| **Storage Technology** | SATA SSD | NVMe M.2 SSD | High-throughput memory-mapped disk I/O for 3D sMRI volumes |
| **Storage Free Space** | 6 GB (Single seed) | 20 GB – 25 GB (Multi-seed) | Dataset ($1.1\text{ GB}$) + 4x sMRI Caches ($3.2\text{ GB}$ per seed) |

---

## 4. Disk-Backed sMRI Augmentation Cache Requirements

The training pipeline relies on a 4x spatial augmentation expansion ($227 \times 4 = 908$ training samples).
- **Single-Seed Cache Size:**
  $$\text{Shape} = (908, 1, 96, 96, 96) \quad (\text{float32}) \implies 908 \times 96^3 \times 4\text{ bytes} \approx \mathbf{3,213,361,152\text{ bytes (3.21 GB)}}$$
- **Multi-Seed (5 Seeds) Cache Storage:**
  $$5 \times 3.21\text{ GB} = \mathbf{16.05\text{ GB}}$$
- **Critical Requirement:**
  The augmentation cache (`train_aug_smri.npy`) must be stored on an **SSD** (internal or fast external USB 3.2). Generating or reading memory maps on slow spinning HDDs causes severe I/O bottlenecks and data starvation.

---

## 5. Hardware Compatibility Classification

### Verified Hardware (Tested & Validated):
- **NVIDIA GeForce MX450:** 2,048 MB VRAM, CUDA 11.8, Windows 11 (Verified 100% stable at batch size 2, accum 8).
- **NVIDIA Tesla T4:** 16,384 MB VRAM, CUDA 12.x, Ubuntu 20.04 / Kaggle (Verified 100% stable at batch sizes 2, 8, 16).
- **CPU-Only (x86_64 Intel/AMD):** Verified for `tools/demo.py` and `tools/safe_fresh_system_check.py`.

### Unverified Hardware (Marked as UNVERIFIED):
- **Apple Silicon (M1/M2/M3 via MPS):** *Unverified for full training.* PyTorch MPS lacks native support for several custom scatter/gather ops used in GNN/Mamba layers.
- **AMD Radeon GPUs (ROCm):** *Unverified.* PyTorch ROCm builds on Windows are unsupported; Linux ROCm untested.
- **Intel Arc GPUs (oneAPI / IPEX):** *Unverified.*
- **CPU-Only Full Training:** *Currently blocked by explicit `.cuda()` statements in training loops.* (Inference works; training unverified).
