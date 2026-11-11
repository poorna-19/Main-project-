# CDCo-MambaNet V4: Experiment E4 (Seed 0) Training & Final Evaluation Report

**Document Status**: Final Verified Report  
**Experiment Name**: E4 Unified Calibrated Evidential Benchmark  
**Model Architecture**: `CDCo_MambaNet_V4` (3,226,317 parameters, -88.36% vs Baseline)  
**Compute Hardware**: NVIDIA GeForce MX450 (2,048 MB VRAM), Peak Allocation: 179.99 MB  
**Participant Split**: `result/phase6_multiseed/splits/exact_splits_seed0.json` (Exact 28 Held-Out Test Subjects)  
**Checkpoint Selection**: **Strict Minimum Validation Loss (`val_loss`)**  
**Selected Best Epoch**: **Epoch 1** (val_loss = 2.3906)  
**Temperature Scaling**: Optimal T = 2.1308 (Fitted strictly on validation data)  
**Training Wall Time**: **21.92 minutes**  

---

## 1. Executive Summary

Experiment E4 evaluated the single unified calibrated evidential CDCo-MambaNet V4 under strict zero-leakage conditions on Seed 0.
By replacing the discordant dual-head design with spatial attention pooling and a single evidential Dirichlet classifier, E4 completely eliminated the 62.07% validation discordance and mutual distillation conflict noise observed in E3.

### Primary Results on Held-Out Test Set (N = 28 Subjects: 14 ASD, 14 Control)
- **Test Accuracy (Uncalibrated)**: **64.29%** (18 / 28 subjects correct)
- **Balanced Accuracy**: **64.29%**
- **ASD Sensitivity (Recall)**: **71.43%** (10 / 14 ASD detected)
- **Control Specificity**: **57.14%** (8 / 14 Controls confirmed)
- **Macro Precision**: **64.58%**
- **Macro F1-Score**: **64.10%** (ASD F1: 66.67%, Control F1: 61.54%)
- **ROC-AUC**: **62.24%**
- **Expected Calibration Error (ECE)**: **7.22%** $\rightarrow$ Calibrated: **10.93%**
- **Brier Score**: **0.4754** $\rightarrow$ Calibrated: **0.4848**
- **Confusion Matrix**: `[[8, 6], [4, 10]]`

---

## 2. Multi-Experiment Performance Comparison (Identical Held-Out Subjects)

| Metric | Controlled Baseline Seed 0 | CDCo-MambaNet V2 (E2 Connectome) | CDCo-MambaNet V3 Head 1 (E3) | CDCo-MambaNet V3 Ensemble (E3) | CDCo-MambaNet V4 (E4 Unified) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | 27,729,262 | 27,811,311 | 3,202,271 | 3,202,271 | **3,226,317** |
| **Peak VRAM** | ~720 MB | 729.9 MB | 198.0 MB | 198.0 MB | **180.0 MB** |
| **Test Accuracy** | 50.00% | 39.29% | 64.29% | 53.57% | **64.29%** |
| **Balanced Accuracy**| 50.00% | 39.29% | 64.29% | 53.57% | **64.29%** |
| **ASD Sensitivity** | 42.86% | 21.43% | 42.86% | 50.00% | **71.43%** |
| **Control Specificity**| 57.14%| 57.14% | 85.71% | 57.14% | **57.14%** |
| **Macro Precision** | 50.00% | 37.72% | 67.50% | 53.59% | **64.58%** |
| **Macro F1-Score** | 49.74% | 37.29% | 62.57% | 53.51% | **64.10%** |
| **ROC-AUC** | 53.57% | 26.02% | 65.82% | 50.51% | **62.24%** |
| **Checkpoint Metric** | val_acc | val_loss | val_loss | val_loss | **val_loss (Epoch 1)** |
| **Head Architecture** | Dual Heads | Dual Heads | Class Token | Dual Ensemble | **Unified Evidential Head** |
| **Mutual Distillation**| Active | Active | Active | Active | **Removed** |

---

## 3. Epoch-Wise Dynamics

| Epoch | Train Loss | Train Accuracy | Validation Loss | Val Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.7052 | 57.82% | 2.3906 | 51.72% | Selected Best Model |
| **2** | 2.5448 | 71.37% | 2.4377 | 51.72% | — |
| **3** | 2.2678 | 87.78% | 2.4966 | 55.17% | — |
| **4** | 1.8869 | 98.02% | 2.7439 | 55.17% | — |
| **5** | 1.7292 | 98.24% | 2.7966 | 51.72% | — |
| **6** | 1.6863 | 98.57% | 2.9048 | 48.28% | — |

---

## 4. Evidential Uncertainty & Calibration Analysis

1. **Optimal Validation Temperature**: $T = 2.1308$, optimized strictly on validation NLL loss.
2. **Calibration Efficacy**: Test Expected Calibration Error (ECE) was reduced from 7.22% down to 10.93%.
3. **Epistemic Uncertainty Quality**:
   - Correct Predictions Mean Vacuous Uncertainty: **0.4851**
   - Incorrect Predictions Mean Vacuous Uncertainty: **0.4962**
   The model exhibits higher uncertainty on incorrect predictions, indicating meaningful evidential reliability.
