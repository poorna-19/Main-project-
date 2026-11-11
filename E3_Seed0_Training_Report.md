# CDCo-MambaNet V3: Experiment E3 (Seed 0) Training & Final Evaluation Report

**Document Status**: Final Verified Report  
**Experiment Name**: E3 Parameter-Efficient Right-Sized Benchmark  
**Model Architecture**: `CDCo_MambaNet_V3` (3,202,271 parameters, -88.49% vs E2)  
**Hardware Platform**: NVIDIA GeForce MX450 (2,048 MB VRAM), Peak Allocation: 198.01 MB  
**Participant Split**: `result/phase6_multiseed/splits/exact_splits_seed0.json` (Exact 28 Held-Out Test Subjects)  
**Checkpoint Selection**: **Strict Minimum Validation Loss (`val_loss`)**  
**Selected Best Epoch**: **Epoch 2** (val_loss = 2.6401)  
**Training Wall Time**: **52.23 minutes**  

---

## 1. Executive Summary

Experiment E3 evaluated the parameter-efficient right-sized CDCo-MambaNet V3 under strict zero-leakage conditions on Seed 0.
By reducing the parameter count from 27.8M to 3.20M, E3 directly attacked the 122,000:1 parameter-to-sample ratio that caused severe overfitting in E2.

### Primary Results on Held-Out Test Set (N = 28 Subjects: 14 ASD, 14 Control)
- **Test Accuracy**: **53.57%** (15 / 28 subjects correct)
- **Balanced Accuracy**: **53.57%**
- **ASD Sensitivity (Recall)**: **50.00%** (7 / 14 ASD detected)
- **Control Specificity**: **57.14%** (8 / 14 Controls confirmed)
- **Macro Precision**: **53.59%**
- **Macro F1-Score**: **53.51%** (ASD F1: 51.85%, Control F1: 55.17%)
- **ROC-AUC**: **50.51%**
- **Confusion Matrix**: `[[8, 6], [7, 7]]`

---

## 2. Multi-Experiment Performance Comparison (Identical Held-Out Subjects)

| Metric | Controlled Baseline Seed 0 | CDCo-MambaNet V2 (E2 Connectome) | CDCo-MambaNet V3 (E3 Right-Sized) | Delta (E3 vs Controlled Baseline) | Delta (E3 vs E2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | 27,729,262 | 27,811,311 | **3,202,271** | **-88.45%** | **-88.49%** |
| **Peak VRAM** | ~720 MB | 729.9 MB | **198.0 MB** | **-508 MB** | **-518 MB** |
| **Test Accuracy** | 50.00% | 39.29% | **53.57%** | **+3.57%** | **+14.28%** |
| **Balanced Accuracy**| 50.00% | 39.29% | **53.57%** | **+3.57%** | **+14.28%** |
| **ASD Sensitivity** | 42.86% | 21.43% | **50.00%** | **+7.14%** | **+28.57%** |
| **Control Specificity**| 57.14%| 57.14% | **57.14%** | **+0.00%** | **+0.00%** |
| **Macro Precision** | 50.00% | 37.72% | **53.59%** | **+3.59%** | **+15.87%** |
| **Macro F1-Score** | 49.74% | 37.29% | **53.51%** | **+3.77%** | **+16.22%** |
| **ROC-AUC** | 53.57% | 26.02% | **50.51%** | **-3.06%** | **+24.49%** |
| **Checkpoint Metric** | val_acc | val_loss (Epoch 4) | **val_loss (Epoch 2)** | Methodological Shift | Consistent |
| **fMRI Encoder** | ConvNeXtV2 (24.7M) | ConvNeXtV2 (24.7M) | **Compact ConvNeXtV2 (0.97M)** | Right-Sized | Right-Sized |
| **sMRI Encoder** | 3D ResNet-50 (2.9M) | 3D ResNet-50 (2.9M) | **3D ResNet-18 (2.08M)** | Right-Sized | Right-Sized |
| **FC GNN Modality** | Ignored (0%) | Active (82K params) | **Active (36K params)** | Integrated | Integrated |

---

## 3. Epoch-Wise Training and Validation Dynamics

| Epoch | Train Loss | Train Accuracy | Validation Loss | Val Head 1 Acc | Val Head 2 Acc | Val Ensemble Acc | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.5602 | 52.31% | 2.8476 | 44.83% | 51.72% | 44.83% | — |
| **2** | 2.9529 | 66.74% | 2.6401 | 41.38% | 62.07% | 58.62% | Selected Best Model |
| **3** | 2.6289 | 78.85% | 2.7325 | 41.38% | 55.17% | 65.52% | — |
| **4** | 2.4085 | 88.99% | 2.6886 | 41.38% | 51.72% | 58.62% | — |
| **5** | 2.2471 | 95.15% | 2.6953 | 51.72% | 55.17% | 62.07% | — |
| **6** | 2.1142 | 97.14% | 2.7543 | 55.17% | 58.62% | 65.52% | — |
| **7** | 2.0603 | 98.46% | 2.8845 | 51.72% | 51.72% | 55.17% | — |

---

## 4. Scientific Verification & Diagnostics

1. **Parameter Reduction Verification**: The model size decreased from 27.8M parameters to exactly **3,202,271 parameters** (an 88.49% decrease).
2. **Active Gradient Flow Across Tri-Modal Pipeline**: sMRI, fMRI, and FC branches all actively participated in backpropagation and forward inference.
3. **Test Cohort Integrity**: The held-out test evaluation evaluated strictly once on the reloaded `best_model.pt` from Epoch 2.
4. **All Deliverables Successfully Generated**:
   - Report: `E3_Seed0_Training_Report.md`
   - Tabular Metrics: `E3_Seed0_Metrics.csv`
   - Model Comparison: `Baseline_E2_E3_Comparison.csv`
   - Memory Report: `E3_Parameter_and_Memory_Report.md`
   - Confusion Matrix: `E3_Confusion_Matrix.png`
   - Training Curves: `E3_Training_Curves.png`
