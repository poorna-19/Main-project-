# Multimodal Neuroimaging Model Comparison & Validation Protocol

**Author:** Senior Deep Learning Research Scientist  
**Focus:** Scientific Rigor, Zero-Leakage Governance, and Reproducibility in ABIDE I Benchmarks  
**Date:** November 1, 2026  
**Status:** Certified Validation Protocol

---

## 1. Principles of Fair & Rigorous Comparison

In computational neuroimaging and clinical machine learning, subtle forms of data leakage frequently lead to overly optimistic performance claims that collapse upon external replication. To ensure scientific integrity, any proposed model—including `CDCo-MambaNet`, `CDCo-CrossAttentionNet`, and classical baselines—must adhere to the standardized comparison protocol detailed below.

```
                           [DATASET: ABIDE I (N=284)]
                                        |
                   +--------------------+--------------------+
                   |                                         |
     [Locked 5-Seed Stratified Splits]             [Scanner-Site Generalization]
     - Seed 0, 1, 2, 3, 4 (SHA-256 Locked)         - Leave-One-Site-Out (LOSO)
     - 80% Train (N=227), 10% Val, 10% Test        - Train on 16 Sites, Test on Held-out Site
                   |                                         |
         +---------+---------+                     +---------+---------+
         |                   |                     |                   |
    [Train Fold]        [Val Fold]            [Train Sites]       [Held-Out Site]
    - Fit Preproc       - Hyperparam Tuning   - Fit Model         - Zero Tuning
    - Compute FC Cov    - Early Stopping      - Calibrate Gate    - Evaluate Generalization
    - Train Weights     - Model Selection           |
         |                   |                      v
         +---------+---------+               [Site Generalization Metric]
                   |
                   v
         [HELD-OUT TEST SET (N=28)]
         - Strictly Post-Training
         - Evaluated ONCE at Best-Val Checkpoint
         - Zero Hyperparameter Tuning
```

---

## 2. Experimental Data Governance & Split Protocol

### 2.1 Locked Five-Seed Splits
All comparative experiments must utilize the exact pre-generated and cryptographically certified splits:
- `result/phase6_multiseed/splits/exact_splits_seed0.json` (SHA-256: `cb327dd6...`)
- `result/phase6_multiseed/splits/exact_splits_seed1.json` (SHA-256: `f05951a1...`)
- `result/phase6_multiseed/splits/exact_splits_seed2.json` (SHA-256: `7bb75bdb...`)
- `result/phase6_multiseed/splits/exact_splits_seed3.json` (SHA-256: `ea8777ce...`)
- `result/phase6_multiseed/splits/exact_splits_seed4.json` (SHA-256: `15ce9eff...`)

Each split partitions the $N=284$ subjects into:
- **Train ($80\%$):** 227 subjects (112 ASD, 115 Control).
- **Validation ($10\%$):** 29 subjects (14 ASD, 15 Control).
- **Held-Out Test ($10\%$):** 28 subjects (14 ASD, 14 Control).

### 2.2 Strict Zero-Leakage Invariants
1. **Zero Overlap:** The intersection between train, validation, and test indices must be strictly empty:
   $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
2. **Preprocessing Isolation:** Any feature normalization (e.g. z-scoring, quantile transforms, PCA projection, graph thresholding $\tau$) must be computed **strictly on $\mathcal{S}_{\text{train}}$** and applied as a frozen transformation to $\mathcal{S}_{\text{val}}$ and $\mathcal{S}_{\text{test}}$.
3. **Data Augmentation Isolation:** Structural MRI augmentations (e.g., 4x expansion in `train_aug_smri_seed{s}.npy`) must be restricted entirely to the training set. Validation and test sets must remain unaugmented raw scans.

---

## 3. Model Training & Selection Rules

### 3.1 Validation-Only Model Selection
- Checkpoint selection must be driven exclusively by a validation metric (e.g. `val_acc_ensemble` or validation balanced accuracy).
- **Prohibition:** Under no circumstances may checkpoints be selected based on test set accuracy, test set loss, or post-hoc test peeking.
- Early stopping patience is fixed at **5 epochs** without validation improvement.

### 3.2 Evaluation Order
```
For each Seed s in [0, 1, 2, 3, 4]:
    1. Initialize model with random seed s.
    2. Train on Train split for maximum 30 epochs.
    3. At the end of each epoch, evaluate on Validation split.
    4. If Validation metric improves:
           Save model weights as 'best_model.pt'.
           Update best_val_epoch.
       Else:
           Increment no_improvement_count.
    5. If no_improvement_count >= patience:
           Trigger Early Stopping.
    6. POST-TRAINING ONLY:
           Load 'best_model.pt'.
           Run single evaluation pass on Test split.
           Record ACC, BACC, SEN, SPE, F1, AUC, and Confusion Matrix.
           Save artifacts strictly in final.json.
```

---

## 4. Benchmark Models & Baselines

To ensure a comprehensive and fair comparison, five distinct model paradigms will be benchmarked on identical splits:

| Paradigm # | Model Name | Input Modalities | Primary Mechanism | Parameter Count |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Classical Ridge / Logistic** | Static Pearson FC (6,670) | Regularized Linear Model ($\ell_2$) | 6,671 |
| **2** | **Classical LinearSVC** | Static Pearson FC (6,670) | Support Vector Classifier ($C=1.0$) | 6,671 |
| **3** | **Classical Graph GNN** | Static Pearson FC (Graph) | 2-layer Graph Attention Network (GAT) | ~165,000 |
| **4** | **CDCo-MambaNet** (Baseline) | sMRI + fMRI time series | ConvNeXtV2-Mamba + 3D ResNet-50 | 27,703,836 |
| **5** | **Hybrid FC + Deep** | 6,670 FC + 424 Deep Latents | $\ell_2$ Logistic Regression on Fused Latents | 7,095 |
| **6** | **CDCo-CrossAttentionNet** | sMRI + fMRI + FC Graph | Morpho-Temporal Cross-Attention + GRL | ~3,995,000 |

---

## 5. Statistical Aggregation & Significance Testing

For every metric $M$ (Accuracy, Balanced Accuracy, Sensitivity, Specificity, ROC-AUC), report:
1. **Individual Seed Values:** $\{M_0, M_1, M_2, M_3, M_4\}$.
2. **Mean:** $\bar{M} = \frac{1}{5} \sum_{s=0}^4 M_s$.
3. **Sample Standard Deviation ($ddof=1$):**
   $$s = \sqrt{\frac{1}{4} \sum_{s=0}^4 (M_s - \bar{M})^2}$$
4. **Population Standard Deviation ($ddof=0$):**
   $$\sigma = \sqrt{\frac{1}{5} \sum_{s=0}^4 (M_s - \bar{M})^2}$$

### Significance Testing:
- To test if Model A significantly outperforms Model B across the 5 seeds:
  - Conduct **Paired Two-Sample t-test** ($p < 0.05$).
  - Conduct **Wilcoxon Signed-Rank Test** for non-parametric validation.
  - Compute **DeLong Test** on ROC curves pooled across test splits.

---

## 6. Scanner-Site Generalization Protocol (Leave-One-Site-Out)

In addition to the 5-seed random splits, evaluate scanner-site generalization using Leave-One-Site-Out (LOSO) cross-validation on the top 4 major sites:
- Fold 1: Train on 16 sites, Test exclusively on **NYU** ($N=58$).
- Fold 2: Train on 16 sites, Test exclusively on **UM_1** ($N=40$).
- Fold 3: Train on 16 sites, Test exclusively on **UCLA_1** ($N=29$).
- Fold 4: Train on 16 sites, Test exclusively on **YALE** ($N=21$).

This isolates whether the model has learned true neurobiological markers of autism or site-specific scanner artifacts.
