# CDCo-MambaNet V4 — Experiment E5: Revised Experimental Design Specification

**Document Status**: Official Revised Experimental Specification  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Reference Baseline**: Experiment E4 Seed 0 (Epoch 1 Best Checkpoint: Acc 64.29%, F1 64.10%, Sens 71.43%, AUC 62.24%)  
**Dataset**: ABIDE I Cohort ($N=284$ subjects), 17 Scanner Sites  
**Evaluation Protocol**: Zero Test Leakage; Validation-Governed Model Selection  
**Execution Guardrail**: **Configuration preparation only.** No training started, no background jobs launched, no checkpoints overwritten.

---

## 1. Executive Summary & Historical Baseline Evidence

The forensic audit of Experiment E4 Seed 0 established two primary findings:
1. **Strong Epoch 1 Generalization**: The unified calibrated evidential head successfully eliminated dual-head discordance, achieving **$71.43\%$ ASD sensitivity**, **$64.10\%$ Macro F1**, and **$64.29\%$ accuracy** on the 28 held-out test participants.
2. **Post-Epoch 1 Overfitting & Collapse**: Training accuracy climbed from $57.82\%$ (Epoch 1) to $98.57\%$ (Epoch 6) while validation loss drifted upward ($2.3906 \rightarrow 2.9048$). Evaluating the overfit Epoch 6 model on held-out test data caused accuracy to collapse to **$42.86\%$** (sub-chance).

Experiment E5 investigates five targeted ablation pillars to systematically improve generalization without compromising scientific integrity.

---

## 2. Pillar E5-A: Regularization Optimization

### 2.1 Experimental Factors
All factors are evaluated in isolated, single-factor configurations:
- **A1: Gradient Norm Clipping**:
  - `clip_grad_norm: 1.0` vs. unclipped baseline (`null`).
  - Bounds gradient spikes across accumulated batches to prevent kernel distortion.
- **A2: Weight Decay Systematic Sweep**:
  - `weight_decay` $\in \{0.05, 0.10, 0.20\}$ (Baseline: $0.05$).
  - Evaluates $L_2$ regularization pressure on the 2.08M parameters of 3D ResNet-18.
- **A3: Spatial Attention Pooling Dropout**:
  - `spatial_attn_dropout` $\in \{0.10, 0.25\}$ (Baseline: $0.10$).
  - Prevents the query vector $\mathbf{q} \in \mathbb{R}^{48}$ from locking onto fixed high-contrast boundary tokens.
- **A4: Classification Head Dropout**:
  - `unified_head_dropout` $\in \{0.50, 0.60\}$ (Baseline: $0.50$).
  - Evaluates feature co-adaptation penalty in the final Dirichlet projection layer.

### 2.2 Verification of Gradient Accumulation, Optimizer Stepping & Clipping Placement
- **Accumulation Structure**: Micro-batch size is $2$, `accumulate_grad_batches = 8`, effective batch size is $16$.
- **Stepping Cadence**: With $454$ total mini-batches per epoch, $454 = 56 \times 8 + 6$.
  - Batches $1$–$448$: Accumulate over 8 batches, triggering 56 optimizer steps.
  - Batches $449$–$454$: Accumulate over final 6 batches, flushed by boundary check `(batch_idx + 1) == num_batches`.
  - Exactly **57 optimizer update steps occur per epoch**.
- **Clipping Placement**: `clip_grad_norm_` is placed **immediately prior to `optimizer.step()`**, acting on the accumulated gradients of the effective batch of 16. It is never called inside the micro-batch accumulation step.

---

## 3. Pillar E5-B: Learning-Rate Schedule & Warmup

### 3.1 Optimizer & Scheduler Inspection
- **Baseline Flaw**: In [`conf/optimizer/adamw_wd05.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/optimizer/adamw_wd05.yaml), `mode: cos_lr` computed:
  $$\text{period} = \max(1, \text{total\_steps} // 100) = 13,620 // 100 = 136\text{ steps}$$
  Because $454$ mini-batches occur per epoch, the learning rate cycled **$3.338$ complete times per epoch**.

### 3.2 Resolution of Gradient Accumulation & Step Discrepancy
- Mini-batch steps per epoch: $454$
- Optimizer update steps per epoch: $57$
- Total mini-batch steps (30 epochs): $13,620$
- Total optimizer update steps (30 epochs): $1,710$

### 3.3 Warmup Calculation Using Actual Optimizer Update Steps
- A 3-epoch warmup corresponds to:
  $$3\text{ epochs} \times 57\text{ optimizer steps/epoch} = \mathbf{171\text{ optimizer update steps}}$$
  $$(3\text{ epochs} \times 454\text{ mini-batches/epoch} = \mathbf{1,362\text{ mini-batch steps}})$$
- Warmup smoothly scales learning rate from $1.0\times 10^{-6}$ to $1.0\times 10^{-4}$.

### 3.4 Verification of Monotonic Cosine Schedule
- In `mode: cos`:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T_{\max} - t_{\text{warmup}}}\pi\right)\right)$$
- The schedule is strictly monotonic decreasing after the warmup period. It **never cycles within an epoch**.

---

## 4. Pillar E5-C: Multimodal Capacity & Unimodal Ablation

To isolate the individual contribution of each modality before contemplating any architectural alterations, four separate validation-only configurations are specified:

| Configuration ID | sMRI (3D ResNet-18) | fMRI (ConvNeXtV2) | FC (BrainConnectomeGNN) | Availability Mask (`avail`) |
| :--- | :---: | :---: | :---: | :---: |
| **E5-C1 (sMRI-Only)** | **Active** (2.08M params) | Masked (Stand-in token) | Masked (Stand-in token) | `[True, False, False]` |
| **E5-C2 (fMRI-Only)** | Masked (Stand-in token) | **Active** (969K params) | Masked (Stand-in token) | `[False, True, False]` |
| **E5-C3 (FC-Only)** | Masked (Stand-in token) | Masked (Stand-in token) | **Active** (7.3K params) | `[False, False, True]` |
| **E5-C4 (Tri-Modal)** | **Active** | **Active** | **Active** | `[True, True, True]` (Baseline) |

### Strict Capacity Guardrail:
In accordance with instructions, **Connectome GNN hidden dimensions ($32$) will not be expanded** until baseline unimodal contributions (E5-C1, E5-C2, E5-C3) are fully measured against validation data.

---

## 5. Pillar E5-D: Scanner-Site Robustness & Distribution Audit

### 5.1 Site-Wise Validation Distribution & Class Balance ($N=29$)

| Scanner Site | Scanner Model & Field Strength | Total in Val | ASD in Val | Control in Val | Class Imbalance | Small-Sample Category |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **NYU** | Siemens Allegra 3.0T | **7** | 5 | 2 | 71.4% ASD | High-Volume ($N \ge 4$) |
| **UM_1** | GE Signa 3.0T | **4** | 2 | 2 | Balanced (50/50) | High-Volume ($N \ge 4$) |
| **UCLA_1** | Siemens Trio 3.0T | **4** | 4 | 0 | **100% ASD** | High-Volume ($N \ge 4$) |
| **UM_2** | GE Signa 3.0T | **4** | 1 | 3 | 75.0% Control | High-Volume ($N \ge 4$) |
| **LEUVEN_2** | Philips Intera 3.0T | **3** | 0 | 3 | **100% Control** | Medium-Volume ($N = 3$) |
| **KKI** | Philips Achieva 3.0T | **2** | 0 | 2 | **100% Control** | Low-Volume ($N \le 2$) |
| **MAX_MUN** | Siemens Verio 3.0T | **1** | 0 | 1 | 100% Control | Very Small ($N = 1$) |
| **PITT** | Siemens Allegra 3.0T | **1** | 0 | 1 | 100% Control | Very Small ($N = 1$) |
| **SDSU** | GE Discovery 3.0T | **1** | 0 | 1 | 100% Control | Very Small ($N = 1$) |
| **UCLA_2** | Siemens Trio 3.0T | **1** | 1 | 0 | 100% ASD | Very Small ($N = 1$) |
| **YALE** | Siemens Trio 3.0T | **1** | 1 | 0 | 100% ASD | Very Small ($N = 1$) |
| **Total** | 11 Sites | **29** | **14** | **15** | Balanced (48/52) | Entire Validation Cohort |

### 5.2 Critical Small-Sample Methodological Guardrail
- 6 out of 11 sites contain **only 1 subject**. Reporting site-specific validation accuracy for these sites produces binary $0\%$ or $100\%$ noise.
- 3 sites with $N \ge 2$ (UCLA_1, LEUVEN_2, KKI) have **$100\%$ uniform class skew**.
- **Rule**: Robustness must be evaluated by aggregating into Scanner Manufacturer Groups:
  - **Siemens 3T** ($N=15$: NYU, UCLA_1, UCLA_2, PITT, MAX_MUN, YALE)
  - **GE 3T** ($N=9$: UM_1, UM_2, SDSU)
  - **Philips 3T** ($N=5$: LEUVEN_2, KKI)

### 5.3 Instance Normalization Ablation
- In feature projection adapters (`smri_change`, `fmri_change`), add 1D Instance Normalization (`nn.InstanceNorm1d(48, affine=False)`) prior to SSM mixing.
- Normalizes feature channel variance across each subject independently, suppressing scanner-specific intensity offsets without requiring scanner site labels (zero target leakage).

---

## 6. Pillar E5-E: Multimodal Fusion Optimization

Three independent single-factor comparisons:

| Comparison ID | Factor Under Investigation | Baseline Reference | Experimental Variant | Dimensional Compatibility Verification |
| :---: | :--- | :---: | :---: | :--- |
| **E5-E1** | Spatial Attention Heads | 4 Heads ($d_h=12$) | **8 Heads ($d_h=6$)** | $d_{\text{embed}} = 48$; $48 \pmod 8 = 0$. Fully divisible. |
| **E5-E2** | DFD Minimum Dimension | $min\_dim = 12$ | **$min\_dim = 24$** | Range: $k \in [24, 48]$. Preserves $\ge 50\%$ feature width. |
| **E5-E3** | Contrastive Loss Weight | $\lambda_{\text{ct}} = 1.0$ | **$\lambda_{\text{ct}} = 0.5$** | Scales auxiliary contrastive perturbation penalty. |

---

## 7. Multi-Seed Protocol & Split Verification

### 7.1 Cross-Seed Forensic Analysis (5 Independent Splits)

| Split File | Train Subjects | Val Subjects | Held-Out Test Subjects | Leakage Across Partitions | Pairwise Test Overlap vs Seed 0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `exact_splits_seed0.json` | 227 (112 ASD / 115 Con) | 29 (14 ASD / 15 Con) | 28 (14 ASD / 14 Con) | **0 subjects** | Baseline ($28/28$) |
| `exact_splits_seed1.json` | 227 (112 ASD / 115 Con) | 29 (14 ASD / 15 Con) | 28 (14 ASD / 14 Con) | **0 subjects** | 3 / 28 subjects |
| `exact_splits_seed2.json` | 227 (112 ASD / 115 Con) | 29 (14 ASD / 15 Con) | 28 (14 ASD / 14 Con) | **0 subjects** | 5 / 28 subjects |
| `exact_splits_seed3.json` | 227 (112 ASD / 115 Con) | 29 (14 ASD / 15 Con) | 28 (14 ASD / 14 Con) | **0 subjects** | 4 / 28 subjects |
| `exact_splits_seed4.json` | 227 (112 ASD / 115 Con) | 29 (14 ASD / 15 Con) | 28 (14 ASD / 14 Con) | **0 subjects** | 5 / 28 subjects |

- **Finding**: The five splits are **independently sampled stratified partitions**.
- Each test set contains exactly 14 ASD and 14 Control participants.
- Within each seed, subject leakage is strictly **0**.
- Across seeds, test sets overlap by only $1$–$5$ subjects ($\approx 10\%$), matching the theoretical expected overlap for random independent sampling from $N=284$.

### 7.2 Predeclared Model-Selection Procedure
- Checkpoint selection is strictly governed by **minimum validation loss (`val_loss`)**.
- Early stopping patience: 5 non-improving validation epochs.
- Zero held-out test data will be used for model selection, early stopping, or hyperparameter tuning.

---

## 8. File and Checkpoint Safety Attestation

- `models/CDCo_MambaNet_V4/`: Permanently read-only.
- `result/experiment4_unified/seed_0/best_model.pt`: Permanently preserved.
- `reports/Finalization/E4_Seed0/`: Permanently preserved.
- All E5 outputs will be written exclusively to `result/experiment5_generalization/` and `reports/Finalization/E5_Generalization/`.

---

*Revised design specification complete. No code changes or experiments will be executed until explicit user authorization is provided.*
