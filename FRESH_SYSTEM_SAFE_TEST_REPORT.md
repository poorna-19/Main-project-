# CDCo-MambaNet V4 — Fresh-System Safe Test Report

**Execution Date:** November 11, 2026  
**Execution Command:** `python tools/safe_fresh_system_check.py`  
**Execution Status:** 100% PASSED (All 4 verification stages successful)  
**Safety Protocol:** Zero real patient data accessed; zero training epochs; zero checkpoints modified  

---

## 1. Execution Log Summary

```
************************************************************
CDCo-MambaNet V4 — Fresh-System Safe Verification
************************************************************

============================================================
STEP 1: Checking Python & Core Dependencies
============================================================
[OK] Python Runtime: 3.10.9
  [PASS] torch        (v2.7.1+cu118) — PyTorch deep learning framework
  [PASS] hydra        (v1.3.8) — Hydra configuration manager
  [PASS] omegaconf    (v2.3.1) — OmegaConf dictionary tools
  [PASS] timm         (v1.0.30) — PyTorch Image Models
  [PASS] einops       (v0.8.2) — Tensor transformation operations
  [PASS] sklearn      (v1.7.2) — Scikit-learn metrics & splits
  [PASS] scipy        (v1.15.3) — Scientific computing
  [PASS] pandas       (v2.3.3) — Data manipulation tools
  [PASS] matplotlib   (v3.10.8) — Plotting utilities
  [PASS] networkx     (v3.4.2) — Graph algorithms

============================================================
STEP 2: Compute Device & PyTorch Detection
============================================================
[OK] PyTorch Version: 2.7.1+cu118
[INFO] CUDA Available: True
  [PASS] GPU Detected: NVIDIA GeForce MX450
  [PASS] Total GPU VRAM: 2047.8 MB (2.00 GB)
  [PASS] CUDA Capability: (7, 5)

============================================================
STEP 3: Synthetic-Data Architecture Verification
============================================================
NOTE: Using 100% synthetic random tensors. Real patient data is QUARANTINED.
[1/4] Instantiating CDCo_MambaNet_V4 architecture...
  [PASS] Total Parameters: 3,226,317 (Trainable: 3,226,317)
[2/4] Executing synthetic forward pass...
  [PASS] Forward output received: Dual evidential tuple
         Head 1 Logits shape: torch.Size([2, 2]) (expected: [2, 2])
         Head 2 Alpha/Evidence shape: torch.Size([2, 2]) (expected: [2, 2])
[3/4] Testing backward pass gradient computation...
  [PASS] Backward pass complete. Loss: 0.8038
  [PASS] Active gradient tensors: 217/251
[4/4] Testing gradient clipping safety (max_norm=1.0)...
  [PASS] Gradient clipping verified. Raw norm: 230.9694 -> Clipped to <= 1.0

============================================================
STEP 4: Verifying Frozen Split Files
============================================================
  [PASS] exact_splits_seed0.json   -> Train: 227, Val: 29, Held-Out Test: 28
  [PASS] exact_splits_seed1.json   -> Train: 227, Val: 29, Held-Out Test: 28
  [PASS] exact_splits_seed2.json   -> Train: 227, Val: 29, Held-Out Test: 28
  [PASS] exact_splits_seed3.json   -> Train: 227, Val: 29, Held-Out Test: 28
  [PASS] exact_splits_seed4.json   -> Train: 227, Val: 29, Held-Out Test: 28

============================================================
VERIFICATION SUMMARY
============================================================
  1. Dependencies:      PASSED
  2. Compute Device:    PASSED (CUDA)
  3. Synthetic Model:   PASSED
  4. Split Integrity:   PASSED
============================================================
[SUCCESS] All fresh-system safety checks passed without errors.
[SAFETY] Held-out test data remained 100% quarantined. Zero patient data accessed.
```

---

## 2. Quantitative Verification Analysis

### Stage 1: Dependency Integrity
All 10 fundamental project libraries loaded and reported versions satisfying the defined compatibility boundaries.

### Stage 2: Hardware Interfacing
The PyTorch CUDA runtime successfully interfaced with the host GPU (`NVIDIA GeForce MX450`, compute capability 7.5, 2,048 MB VRAM).

### Stage 3: Synthetic Deep Learning Pipeline
Using synthetic inputs ($B=2$), the model:
1. Validated the unified architecture parameter count: exactly **$3,226,317$ parameters**.
2. Successfully produced dual evidential outputs matching target tensor shapes (`torch.Size([2, 2])`).
3. Computed valid cross-entropy Dirichlet loss ($0.8038$) and backpropagated without numerical instability (217 active parameter tensors receiving non-zero gradients).
4. Confirmed that Phase 2 gradient clipping accurately constrained gradients from raw norm $230.97$ down to $\le 1.0$.

### Stage 4: Participant Split Immutability
All 5 seed files in `splits/` (`seed0` through `seed4`) were verified to be valid, well-formed JSON structures with exact subject counts:
$$\text{Train} = 227 \quad | \quad \text{Validation} = 29 \quad | \quad \text{Held-Out Test} = 28$$

---

## 3. Strict Safety Compliance Statement

- **Quarantine Maintained:** The file `demo/abide_seed0_test.npz` was **never opened or referenced** during this verification.
- **Model Isolation:** No historical checkpoints in `result/` were modified or overwritten.
- **No Training Executed:** Zero training epochs were run on real or cached ABIDE data.
