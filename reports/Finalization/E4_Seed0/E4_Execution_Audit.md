# CDCo-MambaNet V4: Experiment E4 Benchmark Execution Audit

**Audit Status**: Verified & Attested  
**Experiment**: CDCo-MambaNet V4 Controlled Seed-0 Benchmark (Phase 2)  
**Execution Timestamp**: 2026-11-11 10:02:17  
**Lead Auditor**: Senior Machine Learning Research Engineer  

---

## 1. Checkpoint Forensics & Clean Restart Attestation

1. **Step 1 Checkpoint Forensics**:
   - The prior interrupted run reached Epoch 3 before a system restart halted execution.
   - Forensic analysis of `result/experiment4_unified/seed_0/best_model.pt` confirmed it contained only raw model weights (`fuse_encoder.*`), with no saved AdamW momentum buffers, CosineAnnealing step counters, or RNG states.
   - In compliance with Phase 2 instructions ("Do not assume an interrupted run can be resumed exactly without verifying its state"), an exact, deterministic clean restart from Epoch 1 was executed under Seed 0.
2. **Deterministic Seeding**:
   - `torch.manual_seed(0)`, `torch.cuda.manual_seed_all(0)`, `np.random.seed(0)`.
   - `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.

---

## 2. Model Architecture & Parameter Verification

- **Instantiated Architecture**: `CDCo_MambaNet_V4` (Unified Calibrated Evidential Head)
- **Submodule Breakdown**:
  - `fuse_encoder.smri_encoder` (ResNet-18 3D): 2,080,512 parameters
  - `fuse_encoder.fmri_encoder` (ConvNeXtV2): 969,120 parameters
  - `fuse_encoder.fc_encoder` (Adaptive Connectome GNN): 7,296 parameters
  - `fuse_encoder.cross_scan_mamba` (Bi-directional SSM): 148,800 parameters
  - `fuse_encoder.gate` (Reliability Gate & DFD): 40 parameters
  - `fuse_encoder.spatial_pooler` (Multi-Head Attention Pooling): 512 parameters
  - `predictor` (MLPHead_V3 Unified Evidential Classifier): 20,080 parameters
  - `temperature_scaler` (Post-Hoc Calibrator Parameter): 1 parameter
  - **Other / Shared Projection Layers**: -32 parameters
- **Total Model Parameters**: **3,226,317** (Exact verification match; assertion passed).
- **Parameter Reduction vs Baseline**: **-88.36%** (27,729,262 $\rightarrow$ 3,226,317).

---

## 3. Dataset & Split Integrity Audit

- **Master Dataset**: `E:/abide_data/abide_284.npy` ($N=284$ subjects)
- **4x sMRI Augmentation Cache**: `E:/abide_data/train_aug_smri.npy`
- **Frozen Split File**: `result/phase6_multiseed/splits/exact_splits_seed0.json`
  - **Training Cohort**: **227 subjects** (4x augmented to 908 sMRI volumes)
  - **Validation Cohort**: **29 subjects** (Strictly unaugmented, single-view)
  - **Held-Out Test Cohort**: **28 subjects** (Strictly unaugmented, single-view: 14 ASD, 14 Control)
  - **Participant Leakage**: **0 subjects** overlap across Train, Val, and Test.

---

## 4. Hardware & Resource Profile

- **Compute Device**: NVIDIA GeForce MX450 (Device 0, CC 7.5, Turing architecture)
- **Physical VRAM Available**: 2,048 MB
- **Peak VRAM Allocated During Training**: **179.99 MB** (<11% of total VRAM capacity)
- **Total Training Duration**: **21.92 minutes** (1315.1 seconds)
- **Execution Stability**: 0 CUDA OOM exceptions, gradient accumulation = 8, effective batch size = 16.

---

## 5. Checkpoint Selection & Early Stopping Audit

- **Checkpoint Selection Metric**: **Strict Minimum Validation Loss (`val_loss`)**
- **Selected Best Checkpoint**: **Epoch 1**
- **Validation Loss at Best Checkpoint**: **2.3906**
- **Early Stopping Patience**: 5 epochs (no improvement after epoch 1 triggered early stop).
- **Prohibition of Accuracy Cherry-Picking**: Validation accuracy was logged for diagnostic purposes only; checkpoint selection was strictly governed by `val_loss`.

---

## 6. Single-Pass Held-Out Test Evaluation & Quarantine Audit

- **Validation Temperature Scaling**: Optimal $T = 2.1308$ fitted strictly on validation data.
- **Model Freezing**: Weights and temperature scaling parameter were frozen prior to testing.
- **Single Test Evaluation**: The 28 held-out test subjects were loaded and evaluated exactly once.
- **No Test Tuning**: Zero hyperparameters, thresholds, or temperatures were adjusted using test outputs.

---

## 7. Final Deliverables Manifest

All 8 required deliverables have been compiled, verified, and placed in `reports/Finalization/E4_Seed0/`:
1. `E4_Seed0_Training_Report.md`: Full executive and technical report.
2. `E4_Seed0_Metrics.csv`: Comprehensive tabular metrics (Uncalibrated and Calibrated).
3. `E4_Baseline_E2_E3_E4_Comparison.csv`: Certified comparison across all project phases.
4. `E4_Confusion_Matrix.png`: 300 DPI high-resolution test confusion matrix.
5. `E4_Training_Curves.png`: 300 DPI high-resolution loss and accuracy trajectories.
6. `E4_ROC_Curve.png`: 300 DPI high-resolution test ROC curve with AUC.
7. `E4_Calibration_Report.md`: In-depth temperature scaling and evidential uncertainty audit.
8. `E4_Execution_Audit.md`: This execution audit and attestation document.
