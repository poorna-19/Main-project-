# CDCo-MambaNet V4 — Experiment E5: Multi-Seed Statistical Validation Protocol

**Document Status**: Official Multi-Seed Protocol & Split Audit  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Cohort**: ABIDE I ($N=284$ subjects), 17 Scanner Sites  
**Splits Evaluated**: 5 Independent Partitions (`exact_splits_seed0.json` through `exact_splits_seed4.json`)  
**Execution Guardrail**: **Protocol specification only.** Zero training, zero test set access.

---

## 1. Multi-Seed Split Forensic Verification

All five split files located in `result/phase6_multiseed/splits/` were forensically inspected for participant counts, class distributions, scanner site distribution, and subject-level leakage:

| Split File | Seed | Training Partition | Validation Partition | Held-Out Test Partition | Leakage Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `exact_splits_seed0.json` | **0** | **227** (112 ASD / 115 Con) | **29** (14 ASD / 15 Con) | **28** (14 ASD / 14 Con) | **0 subjects** |
| `exact_splits_seed1.json` | **1** | **227** (112 ASD / 115 Con) | **29** (14 ASD / 15 Con) | **28** (14 ASD / 14 Con) | **0 subjects** |
| `exact_splits_seed2.json` | **2** | **227** (112 ASD / 115 Con) | **29** (14 ASD / 15 Con) | **28** (14 ASD / 14 Con) | **0 subjects** |
| `exact_splits_seed3.json` | **3** | **227** (112 ASD / 115 Con) | **29** (14 ASD / 15 Con) | **28** (14 ASD / 14 Con) | **0 subjects** |
| `exact_splits_seed4.json` | **4** | **227** (112 ASD / 115 Con) | **29** (14 ASD / 15 Con) | **28** (14 ASD / 14 Con) | **0 subjects** |

### Split Independence Audit:
- **Zero Within-Seed Leakage**: In every split file, $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, and $\text{Val} \cap \text{Test} = \emptyset$.
- **Distinct vs. Shared Evaluation**:
  - The five splits are **distinct, independently sampled stratified partitions**.
  - Across seeds, test sets overlap by only $1$ to $6$ subjects (average: $3.7 / 28$ subjects, $\approx 13\%$), consistent with the expected theoretical overlap of independent $10\%$ random draws from $N=284$:
    $$\mathbb{E}[\text{overlap}] = 28 \times \frac{28}{284} \approx 2.76\text{ subjects}$$
  - No split is an identical clone or trivial permutation of another.

---

## 2. Cross-Seed Scanner-Site Distribution Audit

The distribution of the 17 scanner sites was verified across all 5 seeds:

| Scanner Site | Total in ABIDE I | Seed 0 (Tr/Val/Te) | Seed 1 (Tr/Val/Te) | Seed 2 (Tr/Val/Te) | Seed 3 (Tr/Val/Te) | Seed 4 (Tr/Val/Te) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NYU** | 58 | 50 / 7 / 1 | 48 / 6 / 4 | 44 / 9 / 5 | 45 / 7 / 6 | 45 / 6 / 7 |
| **UM_1** | 40 | 30 / 4 / 6 | 32 / 4 / 4 | 33 / 4 / 3 | 31 / 5 / 4 | 32 / 5 / 3 |
| **UCLA_1** | 29 | 20 / 4 / 5 | 24 / 2 / 3 | 25 / 3 / 1 | 24 / 2 / 3 | 25 / 2 / 2 |
| **YALE** | 21 | 17 / 1 / 3 | 16 / 3 / 2 | 16 / 1 / 4 | 18 / 2 / 1 | 17 / 2 / 2 |
| **KKI** | 16 | 13 / 2 / 1 | 13 / 1 / 2 | 14 / 1 / 1 | 13 / 1 / 2 | 12 / 3 / 1 |
| **UM_2** | 15 | 9 / 4 / 2 | 12 / 2 / 1 | 12 / 2 / 1 | 12 / 1 / 2 | 11 / 2 / 2 |
| **LEUVEN_2**| 14 | 8 / 3 / 3 | 12 / 2 / 0 | 11 / 1 / 2 | 10 / 2 / 2 | 12 / 1 / 1 |
| **OLIN** | 14 | 14 / 0 / 0 | 12 / 1 / 1 | 12 / 1 / 1 | 10 / 2 / 2 | 11 / 2 / 1 |
| **STANFORD**| 13 | 10 / 0 / 3 | 11 / 2 / 0 | 11 / 1 / 1 | 11 / 2 / 0 | 10 / 2 / 1 |
| **PITT** | 13 | 11 / 1 / 1 | 10 / 2 / 1 | 9 / 2 / 2 | 10 / 1 / 2 | 11 / 1 / 1 |
| **TRINITY** | 13 | 12 / 0 / 1 | 9 / 2 / 2 | 10 / 1 / 2 | 12 / 1 / 0 | 10 / 1 / 2 |
| **UCLA_2** | 11 | 10 / 1 / 0 | 9 / 1 / 1 | 9 / 0 / 2 | 8 / 2 / 1 | 10 / 0 / 1 |
| **USM** | 11 | 11 / 0 / 0 | 7 / 0 / 4 | 9 / 1 / 1 | 9 / 1 / 1 | 8 / 1 / 2 |
| **SDSU** | 10 | 7 / 1 / 2 | 8 / 1 / 1 | 8 / 1 / 1 | 8 / 0 / 2 | 8 / 1 / 1 |
| **MAX_MUN** | 4 | 3 / 1 / 0 | 2 / 0 / 2 | 2 / 2 / 0 | 4 / 0 / 0 | 3 / 0 / 1 |
| **CALTECH** | 1 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |
| **LEUVEN_1**| 1 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |

---

## 3. Predeclared Final Model-Selection Procedure

To ensure scientific integrity and eliminate post-hoc selection bias, the decision protocol is formalized:

1. **Phase 2 Development (Seed 0 Only)**:
   - All ablation decisions (E5-A through E5-E) are decided **strictly on the Validation partition of Seed 0**.
   - Checkpoint selection is strictly minimum validation loss (`val_loss`):
     $$\theta^* = \arg\min_{\theta \in \{\theta_1, \dots, \theta_{30}\}} \mathcal{L}_{\text{val}}(\theta)$$
   - Early stopping patience is fixed at 5 epochs.
   - **Zero test data is evaluated during Phase 2**.
2. **Phase 5 Multi-Seed Evaluation (Seeds 0, 1, 2, 3, 4)**:
   - The single best configuration emerging from Phase 2 will be frozen.
   - It will be trained independently across Seeds 0, 1, 2, 3, and 4.
   - In every seed run, the checkpoint is selected strictly by minimum validation loss (`val_loss`) on that seed's validation set.
   - Once all 5 models are frozen, each seed's held-out test set will be evaluated in a **single forward pass**.
3. **Statistical Reporting**:
   - Report Mean, Standard Deviation, and 95% Confidence Intervals across the 5 independent seeds:
     $$\bar{X} \pm \sigma, \quad \text{CI}_{95\%} = \bar{X} \pm 2.776 \times \frac{\sigma}{\sqrt{5}}$$
   - Metrics reported: Accuracy, Balanced Accuracy, ASD Sensitivity, Specificity, Macro Precision, Macro F1, ROC-AUC, ECE, Brier Score, Parameters, Peak VRAM, and Wall-clock Training Time.

---

## 4. Test Quarantine Declaration
- **Strict Prohibition**: Held-out test performance will never be used for hyperparameter selection, scheduler tuning, or early stopping.
- Historical test performance from E4 Seed 0 ($64.29\%$ Acc, $64.10\%$ F1) remains permanently documented as the historical reference point.
