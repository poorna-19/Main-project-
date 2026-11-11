# CDCo-MambaNet V3 (Experiment E3) — Metric Consistency & Forensic Data Audit

**Document ID**: `E3_Metric_Consistency_Audit.md`  
**Audit Scope**: Read-Only Forensic Audit of E3 Seed-0 Experiment Results, Confusion Matrices, Participant Splits, Baseline References, and Documentation Consistency.  
**Audited Artifacts**:
- [`result/experiment3_rightsized/seed_0/final.json`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/result/experiment3_rightsized/seed_0/final.json) (Programmatic Ground-Truth Test Output)
- [`E3_Seed0_Metrics.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/E3_Seed0_Metrics.csv) (Official E3 Metrics Table)
- [`Baseline_E2_E3_Comparison.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/Baseline_E2_E3_Comparison.csv) (3-Way Historical Benchmark Table)
- [`Baseline_vs_E2_Comparison.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/Baseline_vs_E2_Comparison.csv) (Baseline & E2 Benchmark Table)
- [`Experiment_Tracking.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/Experiment_Tracking.csv) (Experiment Registry)
- [`E3_Seed0_Training_Report.md`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/E3_Seed0_Training_Report.md) (E3 Training Benchmark Report)
- [`E3_Final_Review_For_Research.md`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/E3_Final_Review_For_Research.md) (Research Review Document)
- [`result/phase6_multiseed/splits/exact_splits_seed0.json`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/result/phase6_multiseed/splits/exact_splits_seed0.json) (Frozen Evaluation Cohort)

---

## 1. Verified Key Metric Summary

Below are the audited, mathematically verified values along with the exact source files supporting each:

| Metric Target | Verified Value | Primary Supporting File | Secondary Supporting File(s) |
| :--- | :---: | :--- | :--- |
| **Controlled Baseline ROC-AUC** | **53.57%** | `Baseline_E2_E3_Comparison.csv` (Row 13) | `Experiment_Tracking.csv` (Row 3), `Baseline_vs_E2_Comparison.csv` (Row 11), `phase6_exp2_audit_report.md` (Row 189) |
| **E3 Head 1 Macro F1** | **62.57%** (0.625668) | `result/experiment3_rightsized/seed_0/final.json` (`test_at_best_val.head1.f1`) | `E3_Seed0_Metrics.csv` (Row 12), `E3_Final_Review_For_Research.md` (Row 176) |
| **E3 Head 1 Specificity (Control)**| **85.71%** (12 / 14) | `result/experiment3_rightsized/seed_0/final.json` (`test_at_best_val.head1.specificity`) | `E3_Seed0_Metrics.csv` (Row 5), `E3_Final_Review_For_Research.md` (Lines 125, 172) |
| **E3 Ensemble Accuracy** | **53.57%** (15 / 28) | `result/experiment3_rightsized/seed_0/final.json` (`test_at_best_val.ensemble.accuracy`) | `E3_Seed0_Metrics.csv` (Row 2), `Baseline_E2_E3_Comparison.csv` (Row 7), `E3_Seed0_Training_Report.md` (Line 20) |
| **E3 Ensemble ROC-AUC** | **50.51%** (0.505102) | `result/experiment3_rightsized/seed_0/final.json` (`test_at_best_val.ensemble.auc`) | `E3_Seed0_Metrics.csv` (Row 13), `Baseline_E2_E3_Comparison.csv` (Row 13), `E3_Seed0_Training_Report.md` (Line 26) |

---

## 2. Mathematical Verification of Confusion Matrices

Each confusion matrix from [`result/experiment3_rightsized/seed_0/final.json`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/result/experiment3_rightsized/seed_0/final.json) and [`E3_Seed0_Metrics.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/E3_Seed0_Metrics.csv) was re-computed from first principles across all 28 held-out test subjects (Class 0: Control, $N=14$; Class 1: ASD, $N=14$).

### 2.1 E3 Primary Ensemble
$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=8 & \text{FP}=6 \\ \text{FN}=7 & \text{TP}=7 \end{pmatrix}$$

- **Total Samples**: $8 + 6 + 7 + 7 = 28$
- **Accuracy**: $\frac{8 + 7}{28} = \frac{15}{28} = \mathbf{53.5714\%} \rightarrow \mathbf{53.57\%}$
- **Balanced Accuracy**: $\frac{1}{2}\left(\frac{8}{14} + \frac{7}{14}\right) = \mathbf{53.5714\%} \rightarrow \mathbf{53.57\%}$
- **ASD Sensitivity (Recall)**: $\frac{7}{7 + 7} = \frac{7}{14} = \mathbf{50.0000\%} \rightarrow \mathbf{50.00\%}$
- **Control Specificity**: $\frac{8}{8 + 6} = \frac{8}{14} = \mathbf{57.1429\%} \rightarrow \mathbf{57.14\%}$
- **Precision (ASD)**: $\frac{7}{7 + 6} = \frac{7}{13} = \mathbf{53.8462\%} \rightarrow \mathbf{53.85\%}$
- **Precision (Control)**: $\frac{8}{8 + 7} = \frac{8}{15} = \mathbf{53.3333\%} \rightarrow \mathbf{53.33\%}$
- **Macro Precision**: $\frac{53.8462\% + 53.3333\%}{2} = \mathbf{53.5897\%} \rightarrow \mathbf{53.59\%}$
- **Macro Recall**: $\frac{50.0000\% + 57.1429\%}{2} = \mathbf{53.5714\%} \rightarrow \mathbf{53.57\%}$
- **ASD F1-Score**: $\frac{2 \times 7}{2 \times 7 + 6 + 7} = \frac{14}{27} = \mathbf{51.8519\%} \rightarrow \mathbf{51.85\%}$
- **Control F1-Score**: $\frac{2 \times 8}{2 \times 8 + 6 + 7} = \frac{16}{29} = \mathbf{55.1724\%} \rightarrow \mathbf{55.17\%}$
- **Macro F1-Score**: $\frac{51.8519\% + 55.1724\%}{2} = \mathbf{53.5121\%} \rightarrow \mathbf{53.51\%}$
- **ROC-AUC**: Evaluated on predicted ensemble probabilities $p_{\text{ens}} = 0.5(p_1 + p_2) \implies \mathbf{50.5102\%} \rightarrow \mathbf{50.51\%}$
- **Status**: **100% mathematically consistent with `E3_Seed0_Metrics.csv` and `final.json`**.

---

### 2.2 E3 Head 1 (Class Token Stream)
$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=12 & \text{FP}=2 \\ \text{FN}=8 & \text{TP}=6 \end{pmatrix}$$

- **Total Samples**: $12 + 2 + 8 + 6 = 28$
- **Accuracy**: $\frac{12 + 6}{28} = \frac{18}{28} = \mathbf{64.2857\%} \rightarrow \mathbf{64.29\%}$
- **Balanced Accuracy**: $\frac{1}{2}\left(\frac{12}{14} + \frac{6}{14}\right) = \mathbf{64.2857\%} \rightarrow \mathbf{64.29\%}$
- **ASD Sensitivity (Recall)**: $\frac{6}{6 + 8} = \frac{6}{14} = \mathbf{42.8571\%} \rightarrow \mathbf{42.86\%}$
- **Control Specificity**: $\frac{12}{12 + 2} = \frac{12}{14} = \mathbf{85.7143\%} \rightarrow \mathbf{85.71\%}$
- **Precision (ASD)**: $\frac{6}{6 + 2} = \frac{6}{8} = \mathbf{75.0000\%} \rightarrow \mathbf{75.00\%}$
- **Precision (Control)**: $\frac{12}{12 + 8} = \frac{12}{20} = \mathbf{60.0000\%} \rightarrow \mathbf{60.00\%}$
- **Macro Precision**: $\frac{75.00\% + 60.00\%}{2} = \mathbf{67.5000\%} \rightarrow \mathbf{67.50\%}$
- **Macro Recall**: $\frac{42.8571\% + 85.7143\%}{2} = \mathbf{64.2857\%} \rightarrow \mathbf{64.29\%}$
- **ASD F1-Score**: $\frac{2 \times 6}{2 \times 6 + 2 + 8} = \frac{12}{22} = \mathbf{54.5455\%} \rightarrow \mathbf{54.55\%}$
- **Control F1-Score**: $\frac{2 \times 12}{2 \times 12 + 2 + 8} = \frac{24}{34} = \mathbf{70.5882\%} \rightarrow \mathbf{70.59\%}$
- **Macro F1-Score**: $\frac{54.5455\% + 70.5882\%}{2} = \mathbf{62.5668\%} \rightarrow \mathbf{62.57\%}$
- **ROC-AUC**: Evaluated on $p_1 \implies \mathbf{65.8163\%} \rightarrow \mathbf{65.82\%}$
- **Status**: **100% mathematically consistent with `E3_Seed0_Metrics.csv` and `final.json`**.

---

### 2.3 E3 Head 2 (Spatial MLP Stream)
$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=4 & \text{FP}=10 \\ \text{FN}=6 & \text{TP}=8 \end{pmatrix}$$

- **Total Samples**: $4 + 10 + 6 + 8 = 28$
- **Accuracy**: $\frac{4 + 8}{28} = \frac{12}{28} = \mathbf{42.8571\%} \rightarrow \mathbf{42.86\%}$
- **Balanced Accuracy**: $\frac{1}{2}\left(\frac{4}{14} + \frac{8}{14}\right) = \mathbf{42.8571\%} \rightarrow \mathbf{42.86\%}$
- **ASD Sensitivity (Recall)**: $\frac{8}{8 + 6} = \frac{8}{14} = \mathbf{57.1429\%} \rightarrow \mathbf{57.14\%}$
- **Control Specificity**: $\frac{4}{4 + 10} = \frac{4}{14} = \mathbf{28.5714\%} \rightarrow \mathbf{28.57\%}$
- **Precision (ASD)**: $\frac{8}{8 + 10} = \frac{8}{18} = \mathbf{44.4444\%} \rightarrow \mathbf{44.44\%}$
- **Precision (Control)**: $\frac{4}{4 + 6} = \frac{4}{10} = \mathbf{40.0000\%} \rightarrow \mathbf{40.00\%}$
- **Macro Precision**: $\frac{44.4444\% + 40.00\%}{2} = \mathbf{42.2222\%} \rightarrow \mathbf{42.22\%}$
- **Macro Recall**: $\frac{57.1429\% + 28.5714\%}{2} = \mathbf{42.8571\%} \rightarrow \mathbf{42.86\%}$
- **ASD F1-Score**: $\frac{2 \times 8}{2 \times 8 + 10 + 6} = \frac{16}{32} = \mathbf{50.0000\%} \rightarrow \mathbf{50.00\%}$
- **Control F1-Score**: $\frac{2 \times 4}{2 \times 4 + 10 + 6} = \frac{8}{24} = \mathbf{33.3333\%} \rightarrow \mathbf{33.33\%}$
- **Macro F1-Score**: $\frac{50.00\% + 33.3333\%}{2} = \mathbf{41.6667\%} \rightarrow \mathbf{41.67\%}$
- **ROC-AUC**: Evaluated on $p_2 \implies \mathbf{39.7959\%} \rightarrow \mathbf{39.80\%}$
- **Status**: **100% mathematically consistent with `E3_Seed0_Metrics.csv` and `final.json`**.

---

### 2.4 Controlled Baseline (Seed 0)
$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN}=8 & \text{FP}=6 \\ \text{FN}=8 & \text{TP}=6 \end{pmatrix}$$

- **Total Samples**: $8 + 6 + 8 + 6 = 28$
- **Accuracy**: $\frac{8 + 6}{28} = \frac{14}{28} = \mathbf{50.00\%}$
- **Balanced Accuracy**: $\frac{1}{2}\left(\frac{8}{14} + \frac{6}{14}\right) = \mathbf{50.00\%}$
- **ASD Sensitivity**: $\frac{6}{14} = \mathbf{42.8571\%} \rightarrow \mathbf{42.86\%}$
- **Control Specificity**: $\frac{8}{14} = \mathbf{57.1429\%} \rightarrow \mathbf{57.14\%}$
- **Macro Precision**: $\frac{50.00\% + 50.00\%}{2} = \mathbf{50.00\%}$
- **Macro F1-Score**: $\mathbf{49.74\%}$
- **ROC-AUC**: $\mathbf{53.57\%}$
- **Status**: **100% mathematically consistent with `Baseline_E2_E3_Comparison.csv` and `Experiment_Tracking.csv`**.

---

## 3. Verification of Held-Out Test Participant Identity

To ensure valid comparative claims without data leakage or cohort shift, the exact test participants were checked across all experimental configurations:

- **Split Source File**: [`result/phase6_multiseed/splits/exact_splits_seed0.json`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/result/phase6_multiseed/splits/exact_splits_seed0.json)
- **Held-Out Test Cohort Size**: Exactly **28 subjects** (14 ASD, 14 Control).
- **Exact Subject IDs**:
  `'51189'`, `'50319'`, `'50205'`, `'51251'`, `'50731'`, `'50628'`, `'50295'`, `'50571'`, `'50011'`, `'50208'`, `'51249'`, `'50969'`, `'51164'`, `'50376'`, `'50414'`, `'50807'`, `'50602'`, `'50360'`, `'50724'`, `'51134'`, `'51267'`, `'51231'`, `'50314'`, `'50285'`, `'50742'`, `'51203'`, `'51183'`, `'50406'`
- **Exact File IDs**:
  `'Stanford_0051189'`, `'UM_1_0050319'`, `'SDSU_0050205'`, `'UCLA_1_0051251'`, `'Leuven_2_0050731'`, `'Yale_0050628'`, `'UM_1_0050295'`, `'Yale_0050571'`, `'Pitt_0050011'`, `'SDSU_0050208'`, `'UCLA_1_0051249'`, `'NYU_0050969'`, `'Stanford_0051164'`, `'UM_1_0050376'`, `'UM_2_0050414'`, `'KKI_0050807'`, `'Yale_0050602'`, `'UM_1_0050360'`, `'Leuven_2_0050724'`, `'Trinity_0051134'`, `'UCLA_1_0051267'`, `'UCLA_1_0051231'`, `'UM_1_0050314'`, `'UM_1_0050285'`, `'Leuven_2_0050742'`, `'UCLA_1_0051203'`, `'Stanford_0051183'`, `'UM_2_0050406'`
- **Zero Leakage Verification**:
  - $\text{Train Subjects} \cap \text{Test Subjects} = \emptyset$ (0 overlap)
  - $\text{Val Subjects} \cap \text{Test Subjects} = \emptyset$ (0 overlap)
  - $\text{Train Subjects} \cap \text{Val Subjects} = \emptyset$ (0 overlap)
  - Total unique participants: $227 + 29 + 28 = 284$.
- **Cross-Experiment Confirmation**:
  - Baseline: Used `exact_splits_seed0.json`
  - Experiment E2: Used `exact_splits_seed0.json`
  - Experiment E3: Used `exact_splits_seed0.json`
- **Conclusion**: **Identical 28 test participants were evaluated across all stages.**

---

## 4. Discrepancy Forensic Audit

This section documents every discrepancy identified between the disk files, `E3_Final_Review_For_Research.md`, and conversational chat summaries.

### Discrepancy 1: Baseline ROC-AUC Value
- **True Audited Value**: **53.57%**
- **Supporting Disk Source**: [`Baseline_E2_E3_Comparison.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/Baseline_E2_E3_Comparison.csv) (Row 13), [`Experiment_Tracking.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/Experiment_Tracking.csv) (Row 3).
- **Discrepancy Location**: In the conversational assistant response for Turn 10, Baseline ROC-AUC was misstated as `45.41%`.
- **In `E3_Final_Review_For_Research.md`**: Line 187 listed `Baseline: 50.00% Accuracy, 49.74% F1` but omitted the ROC-AUC value.
- **Resolution**: Certified baseline ROC-AUC is **53.57%**.

---

### Discrepancy 2: E3 Head 1 Macro F1-Score Value
- **True Audited Value**: **62.57%** (exact: `0.625668449197861`)
- **Supporting Disk Source**: [`result/experiment3_rightsized/seed_0/final.json`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/result/experiment3_rightsized/seed_0/final.json) (`head1.f1`), [`E3_Seed0_Metrics.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/E3_Seed0_Metrics.csv) (Row 12).
- **Discrepancy Location**: In the conversational assistant response for Turn 10, Head 1 Macro F1 was misstated as `63.50%`.
- **In `E3_Final_Review_For_Research.md`**: Line 176 correctly states **`62.57%`**.
- **Resolution**: The file `E3_Final_Review_For_Research.md` is correct at **62.57%**; the conversational chat table had a transcription error (`63.50%`). Certified value is **62.57%**.

---

### Discrepancy 3: Baseline Sensitivity, Specificity, and Macro F1
- **True Audited Values**:
  - Baseline ASD Sensitivity: **42.86%** (6 / 14)
  - Baseline Control Specificity: **57.14%** (8 / 14)
  - Baseline Macro F1: **49.74%**
- **Supporting Disk Source**: [`Baseline_E2_E3_Comparison.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/Baseline_E2_E3_Comparison.csv) (Rows 9–12), [`Experiment_Tracking.csv`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/Experiment_Tracking.csv) (Row 3).
- **Discrepancy Location**: In the conversational assistant response for Turn 10, Baseline Sensitivity was printed as `50.00%`, Specificity as `50.00%`, and Macro F1 as `49.94%`.
- **In `E3_Final_Review_For_Research.md`**: Correctly notes baseline accuracy 50.00% and F1 49.74%.
- **Resolution**: Certified values are Sensitivity **42.86%**, Specificity **57.14%**, and Macro F1 **49.74%**.

---

### Discrepancy 4: Confusion Matrix Mathematical Labeling in `E3_Final_Review_For_Research.md`
- **Location**: Line 180 of `E3_Final_Review_For_Research.md`.
- **Existing Text**:
  $$\begin{pmatrix} \text{True Negative (Control)} = 8 & \text{False Positive (ASD)} = 6 \\ \text{False Negative (Control)} = 7 & \text{True Positive (ASD)} = 7 \end{pmatrix}$$
- **Discrepancy Detail**:
  The bottom-left entry is labeled as `False Negative (Control) = 7`. In binary classification where ASD is the positive class ($y=1$), a False Negative represents an **actual ASD subject misclassified as Control** (not a Control).
- **Correct Formulation**:
  $$\begin{pmatrix} \text{TN (Actual Control, Pred Control)} = 8 & \text{FP (Actual Control, Pred ASD)} = 6 \\ \text{FN (Actual ASD, Pred Control)} = 7 & \text{TP (Actual ASD, Pred ASD)} = 7 \end{pmatrix}$$
- **Resolution**: The numerical matrix $\begin{pmatrix} 8 & 6 \\ 7 & 7 \end{pmatrix}$ and subsequent prose bullet points (Lines 181–184) are mathematically correct; only the LaTeX subscript in the matrix formula was mislabeled.

---

## 5. Master Audited Comparison Table

Direct comparison across all stages with verified data sources:

| Metric | Controlled Baseline (Seed 0) | CDCo-MambaNet V2 (E2) | CDCo-MambaNet V3 Head 1 (E3) | CDCo-MambaNet V3 Ensemble (E3) | E3 vs. Controlled Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | 27,729,262 | 27,811,311 | 3,202,271 | **3,202,271** | **-88.45%** |
| **Test Accuracy** | 50.00% (14/28) | 39.29% (11/28) | **64.29%** (18/28) | **53.57%** (15/28) | **+3.57%** |
| **Balanced Accuracy** | 50.00% | 39.29% | **64.29%** | **53.57%** | **+3.57%** |
| **ASD Sensitivity (Recall)**| 42.86% (6/14) | 21.43% (3/14) | 42.86% (6/14) | **50.00%** (7/14) | **+7.14%** |
| **Control Specificity**| 57.14% (8/14) | 57.14% (8/14) | **85.71%** (12/14) | **57.14%** (8/14) | **+0.00%** |
| **ASD Precision** | 50.00% (6/12) | 33.33% (3/9) | **75.00%** (6/8) | **53.85%** (7/13) | **+3.85%** |
| **Control Precision** | 50.00% (8/16) | 42.11% (8/19) | **60.00%** (12/20) | **53.33%** (8/15) | **+3.33%** |
| **Macro Precision** | 50.00% | 37.72% | **67.50%** | **53.59%** | **+3.59%** |
| **Macro F1-Score** | 49.74% | 37.29% | **62.57%** | **53.51%** | **+3.77%** |
| **ROC-AUC** | 53.57% | 26.02% | **65.82%** | **50.51%** | **-3.06%** |
| **Confusion Matrix** | `[[8, 6], [8, 6]]` | `[[8, 6], [11, 3]]` | `[[12, 2], [8, 6]]` | `[[8, 6], [7, 7]]` | FN: -1, TP: +1 |
| **Primary Source** | `Baseline_E2_E3_Comparison.csv` | `final.json` (E2) | `final.json` (E3) | `final.json` (E3) | Exact Ground Truth |

---

## 6. Audit Conclusion

1. **Test Cohort**: Exactly identical 28 subjects evaluated across Baseline, E2, and E3 with zero participant overlap.
2. **Metrics Consistency**: All values in `E3_Seed0_Metrics.csv`, `final.json`, and `Baseline_E2_E3_Comparison.csv` are 100% mathematically consistent and verified.
3. **Discrepancy Resolution**: All transcription discrepancies from conversational chat summaries have been pinpointed, documented, and traced to their verified disk sources.
4. **Research State**: Checkpoints, models, and code remain unmodified and intact.
