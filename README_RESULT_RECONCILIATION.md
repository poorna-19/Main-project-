# CDCo-MambaNet V4 — README Result Reconciliation Report

**Author:** Senior Deep Learning Research Engineer  
**Date:** November 11, 2026  
**Objective:** Complete scientific reconciliation of historical benchmark metrics and public documentation  

---

## 1. Discrepancy Analysis

A surface inspection of project documentation reveals an apparent conflict:
- **Public `README.md` Summary:** Reports **$55.00 \pm 7.82\%$** test accuracy.
- **E4 Seed-0 Benchmark Report:** Records **$64.29\%$** test accuracy.

This report establishes the precise experimental origin, model architecture, training configuration, checkpoint selection rule, and evaluation cohort for each reported metric. Neither result contradicts the other; they reflect fundamentally different experimental conditions.

---

## 2. Definitive Benchmark Taxonomy

| Metric Dimension | Baseline Kaggle Reproduction (`README.md`) | E4 Seed-0 Controlled Benchmark (`reports/Finalization/`) | Phase 2 Controlled Experiments (`PHASE2_CONFIGS.yaml`) |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Original Repaired CDCo-MambaNet | **CDCo-MambaNet V4** (Unified Evidential Head) | CDCo-MambaNet V4 (Isolated Ablations) |
| **Model Parameters** | $27,729,262$ (~27.7M) | **$3,226,317$ (~3.2M, 88.36% reduction)** | $3,226,317$ |
| **Evaluation Scope** | 5-Seed Average (Seeds 0–4) | **Single Seed (Seed 0 Frozen Benchmark)** | Single Seed & Multi-Seed (Pending) |
| **Training Duration** | 100 Epochs | **6 Epochs** | 30 Epochs (Controlled Cadence) |
| **Checkpoint Evaluated**| **Last Epoch Model (Epoch 100)** | **Strict Minimum Validation Loss (Epoch 1)** | Strict Minimum Validation Loss |
| **Split Methodology** | Kaggle dynamic stratified 80/10/10 | **`splits/exact_splits_seed0.json` (Frozen)** | `splits/exact_splits_seed0.json` (Frozen) |
| **Test Accuracy** | **$55.00 \pm 7.82\%$** | **$64.29\%$ (18/28 correct)** | **UNMEASURED (0 runs executed)** |
| **Balanced Accuracy** | ~$55.00\%$ | **$64.29\%$** | **UNMEASURED** |
| **Macro F1-Score** | $53.00 \pm 7.60\%$ | **$64.10\%$** | **UNMEASURED** |
| **ASD Sensitivity** | ~$54.00\%$ | **$71.43\%$ (10/14 detected)** | **UNMEASURED** |
| **Control Specificity**| ~$56.00\%$ | **$57.14\%$ (8/14 detected)** | **UNMEASURED** |
| **ROC-AUC** | $55.41 \pm 13.85\%$ | **$62.24\%$** | **UNMEASURED** |
| **Primary Finding** | Overfitting over 100 epochs; near-chance predictions | Early stopping at Epoch 1 val_loss preserves generalization | Evaluates regularization to prevent late-epoch collapse |

---

## 3. Detailed Root-Cause Explanation

### Why Did the Kaggle Run Reach 55.0%?
The original Kaggle experiment evaluated the **final epoch model (Epoch 100)**. Because the original 27.7M-parameter model possessed massive excess capacity relative to the small ABIDE training cohort (227 subjects), the model severely overfit early in training. By Epoch 100, validation loss had degenerated and test predictions collapsed toward chance ($55.0 \pm 7.8\%$).

### Why Did E4 Seed-0 Reach 64.29%?
Experiment E4 implemented three vital scientific corrections:
1. **Model Rightsizing:** Cut parameter count from 27.7M down to 3.2M ($88.4\%$ parameter reduction), eliminating unneeded capacity.
2. **Unified Evidential Architecture:** Replaced dual conflicting distillation heads with a single unified calibrated evidential head.
3. **Validation Loss Model Selection:** Evaluated the checkpoint selected strictly by **minimum validation loss** (`val_loss = 2.3906` at Epoch 1). At Epoch 1, the model retained generalization capability before catastrophic memorization took place. (At Epoch 6, training accuracy reached $98.57\%$ while validation accuracy collapsed to $48.28\%$, yielding final-epoch test accuracy of only $53.57\%$).

---

## 4. Scientific Status of Phase 2 (E1–E5)

> [!IMPORTANT]
> **Phase 2 performance has NOT yet been measured.**
> All Phase 2 experiments (E1 Scheduler Correction, E2 Gradient Clipping, E3 Weight Decay, E4 Spatial Dropout, E5 Stochastic Augmentation) are currently in **code preparation and static verification state**. Zero training epochs have been launched. It is scientifically impermissible to claim any Phase 2 accuracy improvement until controlled training is authorized and executed.

---

## 5. Updates to `README.md`

`README.md` will be updated to display both metrics in their proper historical and architectural contexts:
1. Section 1 will display the historical 100-epoch Kaggle baseline ($55.00 \pm 7.82\%$).
2. Section 2 will present the rightsized CDCo-MambaNet V4 Seed-0 benchmark ($64.29\%$ at Epoch 1 val_loss checkpoint).
3. Section 3 will explicitly state that Phase 2 controlled generalization experiments are queued and awaiting execution authorization.
