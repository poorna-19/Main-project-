# CDCo-MambaNet V4 — Fresh System Setup Guide

**Target Audience:** Researchers and developers setting up the project on a clean computer  
**Supported Platforms:** Windows 10/11 (Primary), Ubuntu Linux 20.04/22.04, macOS (Inference/Demo)  
**Python Compatibility:** Python 3.10 (Recommended: 3.10.9–3.10.13) or Python 3.11  

---

## 1. Hardware Requirements & Recommendations

| Specification | Minimum (Demo / Testing) | Recommended (Full Training) |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 or Ubuntu 20.04+ | Windows 10/11 64-bit |
| **CPU** | 4-Core Intel Core i5 or AMD Ryzen 5 | 8-Core Intel Core i7 / AMD Ryzen 7 |
| **RAM** | 8 GB System RAM | 16 GB+ System RAM (for 4x augmentation) |
| **GPU** | None (CPU mode for `tools/demo.py`) | NVIDIA GPU with $\ge 2\text{ GB}$ VRAM (CUDA 11.8 or 12.1) |
| **Measured VRAM**| — | **180 MB** (Batch size 2, gradient accumulation 8) |
| **Free Storage** | 500 MB (Code & demo test set) | 15 GB (ABIDE dataset + 4x sMRI augmentation caches) |

---

## 2. Windows Installation Walkthrough (Step-by-Step)

### Step 1: Install Python 3.10
1. Download Python 3.10 from [python.org](https://www.python.org/downloads/release/python-31013/).
2. During installation, **check the box: "Add python.exe to PATH"**.
3. Verify in PowerShell:
   ```powershell
   python --version
   # Expected: Python 3.10.x
   ```

### Step 2: Unzip the Repository
1. Download the ZIP file from GitHub and extract it to your preferred location (e.g., `C:\Research\CDCO-Mamba-Research-main`).
2. Open PowerShell and navigate into the folder:
   ```powershell
   cd C:\Research\CDCO-Mamba-Research-main
   ```

### Step 3: Create an Isolated Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy error, run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

### Step 4: Install PyTorch with CUDA (or CPU)
- **For NVIDIA GPU with CUDA 11.8 (Tested Baseline):**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **For NVIDIA GPU with CUDA 12.1:**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **For CPU-Only Computers:**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

### Step 5: Install Project Dependencies
```powershell
pip install -r requirements.txt
```

---

## 3. Alternative Conda Installation

If you prefer Anaconda or Miniconda:
```powershell
conda env create -f environment.yml
conda activate cdco-mambanet
```

---

## 4. Hardware & Environment Verification Script

Run the following one-line command to verify your Python, PyTorch, and GPU setup:

```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'); print('VRAM (MB):', torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 'N/A')"
```

**Expected Healthy GPU Output:**
```
PyTorch Version: 2.x.x+cu118
CUDA Available: True
Device Name: NVIDIA GeForce ...
VRAM (MB): 2048.0 (or higher)
```

---

## 5. Instant Smoke Test (No Dataset Needed)

The repository includes a bundled test cohort (`demo/abide_seed0_test.npz`) and unit tests that run immediately without downloading the full ABIDE dataset.

### Test 1: Run Architecture Unit Test
```powershell
python tests/test_cdcomnet_v4.py
```
*Verifies model forward pass, unified dual-head evidential outputs, Dirichlet loss computation, and backpropagation gradient flow.*

### Test 2: Run CPU Test Evaluation Demo
```powershell
python tools/demo.py
```
*Runs inference across the 28 held-out test subjects on CPU under clean and perturbed scanner conditions.*
