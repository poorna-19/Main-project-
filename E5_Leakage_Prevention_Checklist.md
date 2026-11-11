# CDCo-MambaNet V4 — Experiment E5: Data Leakage Prevention & Audit Checklist

**Document Status**: Official Quality Assurance & Leakage Prevention Checklist  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Scope**: All Phase 2 and Phase 5 Controlled Experiments (E5-A through E5-E, Multi-Seed)  
**Verification Standard**: Strict Zero-Leakage & Participant Isolation  

---

## 1. Participant-Level Split Isolation

- [x] **Split File Integrity**:
  - `exact_splits_seed0.json` through `exact_splits_seed4.json` are frozen and read-only.
  - Participant counts verified: Exactly 227 Train, 29 Validation, 28 Test across all 5 seeds.
- [x] **Cross-Partition Overlap Verification**:
  - Training subjects $\cap$ Validation subjects $= \emptyset$ (Verified: $0$ subjects overlap).
  - Training subjects $\cap$ Test subjects $= \emptyset$ (Verified: $0$ subjects overlap).
  - Validation subjects $\cap$ Test subjects $= \emptyset$ (Verified: $0$ subjects overlap).
- [x] **Subject Identity Encoding**:
  - Partitions reference unique subject IDs (`sub_ids`) rather than volume indices, ensuring augmented multi-view volumes of the same subject never cross partition boundaries.

---

## 2. Preprocessing & Normalization Protocol

- [x] **fMRI Time-Series Standardization**:
  - Normalized strictly per-subject and per-ROI:
    $$x_{s, r}(t) \leftarrow \frac{x_{s, r}(t) - \mu_{s, r}}{\sigma_{s, r}}$$
  - Zero population-level mean or variance statistics are computed across subjects.
- [x] **Functional Connectivity Pearson Correlation**:
  - Correlation matrices $C \in \mathbb{R}^{116 \times 116}$ are calculated strictly within each individual subject's standardized timeseries.
  - No cross-subject graph pooling or graph-level standardization is applied.
- [x] **3D Structural MRI (sMRI)**:
  - MNI-152 nonlinear registration and affine alignment were performed independently per scan during dataset preprocessing.

---

## 3. Data Augmentation Isolation Protocol

- [x] **Training-Only Augmentation**:
  - In baseline cached mode, `data_aug` operates exclusively on `final_timeseires_train, smri_train`.
  - In on-the-fly stochastic mode (E5), randomized Euler rotations ($\pm 5^\circ$) and connectome edge dropout ($p=0.05$) are bound strictly inside `train_dataloader`.
- [x] **Validation & Test Quarantine**:
  - `val_dataloader` and `test_dataloader` are instantiated with `shuffle=False` and zero augmentation.
  - Validation metrics reflect genuine, unmodified clinical brain scans.

---

## 4. Scanner-Site Information Handling Protocol

- [x] **Zero Target Leakage**:
  - Scanner site metadata is used strictly for post-hoc disaggregated reporting (assessing performance equity across machines).
  - Scanner site labels are **never passed as input features** to the model.
- [x] **Instance Normalization**:
  - When evaluating feature-level normalization (Ablation E5-D), `nn.InstanceNorm1d` standardizes feature channels per instance dynamically without knowing or utilizing scanner labels.

---

## 5. Model Selection & Checkpoint Governance

- [x] **Predeclared Objective**:
  - Checkpoint selection is strictly governed by **minimum validation loss (`val_loss`)**:
    $$\theta^* = \arg\min_{\theta \in \{\theta_1, \dots, \theta_E\}} \mathcal{L}_{\text{val}}(\theta)$$
- [x] **Prohibition of Test Peeking**:
  - `test_dataloader` is disabled during all Phase 2 ablation cycles.
  - Early stopping decisions are triggered strictly by 5 non-improving validation epochs.

---

## 6. Post-Hoc Calibration & Temperature Scaling Protocol

- [x] **Validation-Only Fitting**:
  - The temperature parameter $T > 0$ is fitted strictly by minimizing Negative Log-Likelihood (NLL) on unnormalized validation logits from the frozen best checkpoint.
- [x] **Quarantine Attestation**:
  - Test set logits and labels are completely isolated from the calibration optimizer.
  - Model weights remain frozen during temperature optimization.

---

## 7. Artifact Safety & Non-Destructive Storage Protocol

- [x] **E4 Frozen Storage**:
  - `result/experiment4_unified/seed_0/best_model.pt` is locked and read-only.
  - `reports/Finalization/E4_Seed0/` reports and figures are preserved untouched.
- [x] **Dedicated E5 Tree**:
  - All E5 checkpoints, logs, and deliverables write exclusively to `result/experiment5_generalization/` and `reports/Finalization/E5_Generalization/`.

---

**Certification**: This checklist has been forensically verified and signed off. Zero leakage avenues exist across the experimental pipeline.
