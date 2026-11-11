# CDCo-MambaNet V4 — Experiment E5: Generalization Improvement Research Plan

**Document Status**: Official Experimental Design Specification  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Target Architecture**: `CDCo_MambaNet_V5` (Generalization-Optimized Unified Architecture)  
**Historical Baseline**: Experiment E4 Seed 0 (Epoch 1 Best Checkpoint: Acc 64.29%, F1 64.10%, Sens 71.43%, AUC 62.24%)  
**Dataset**: ABIDE I Cohort ($N=284$ subjects), 17 Scanner Sites  
**Evaluation Protocol**: Zero Test Leakage; Strictly Validation-Governed Model Selection  

---

## 1. Executive Objective & Historical Baseline Context

In Experiment E4, CDCo-MambaNet V4 successfully unified global class tokens and spatial-topological tokens through spatial attention pooling, eliminating inter-head discordance and achieving **71.43% ASD sensitivity** on held-out test data.

However, forensic audit of E4 Seed 0 revealed severe post-Epoch 1 overfitting:
- **Training Accuracy**: Climbed rapidly from $57.82\%$ (Epoch 1) to $98.57\%$ (Epoch 6).
- **Validation Loss**: Minimum at Epoch 1 ($2.3906$), drifting upward monotonically to $2.9048$ by Epoch 6.
- **Empirical Generalization Collapse**: Evaluating the overfit Epoch 6 model on the test set collapsed accuracy from **$64.29\%$ down to $42.86\%$** (sub-chance).

### Objective of Experiment E5:
Systematically optimize regularization, optimization dynamics, modality capacity balance, and scanner-site robustness to **suppress training memorization, reduce the generalization gap, and improve balanced classification performance** across multiple seeds without compromising scientific integrity.

---

## 2. Strict Experimental Guardrails & Data Leakage Prevention

1. **E4 Integrity Freeze**:
   - `models/CDCo_MambaNet_V4/`, `result/experiment4_unified/seed_0/`, and `reports/Finalization/E4_Seed0/` remain permanently frozen and immutable.
   - All E5 experiments run in isolated directories (`result/experiment5_generalization/` and `reports/Finalization/E5_Generalization/`).
2. **Held-Out Test Set Quarantine**:
   - The 28 held-out test participants in `exact_splits_seed0.json` (and multi-seed splits) are strictly quarantined.
   - **Zero test data** will be used for hyperparameter tuning, learning rate selection, early stopping, or checkpoint selection.
   - Historical test metrics ($64.29\%$ accuracy, $64.10\%$ F1) serve solely as historical reference evidence.
3. **Predeclared Checkpoint Selection Metric**:
   - Checkpoint selection is governed strictly by **minimum validation loss (`val_loss`)**:
     $$\theta^* = \arg\min_{\theta_e} \mathcal{L}_{\text{val}}(e)$$
   - Accuracy-based cherry-picking is strictly prohibited.
4. **Leak-Free Preprocessing**:
   - All feature standardization, normalization statistics, and temperature calibration are fitted strictly on the respective training/validation partitions.

---

## 3. Controlled Ablation Matrix (E5-A through E5-E)

To isolate causative factors, each ablation factor is tested systematically against the Seed-0 frozen split:

```
[E4 Seed 0 Baseline]
   │
   ├── E5-A: Regularization Optimization (Gradient Clipping, Weight Decay, Dropout)
   │
   ├── E5-B: Learning-Rate Schedule Optimization (Monotonic Cosine Annealing vs Pulsing cos_lr)
   │
   ├── E5-C: Modality Capacity Rebalancing (Taming sMRI Overdominance vs Connectome GNN)
   │
   ├── E5-D: Scanner-Site Robustness & Domain-Invariance Analysis
   │
   └── E5-E: Multimodal Attention Fusion Optimization
```

---

### Ablation E5-A: Regularization Optimization
*Hypothesis*: Unbounded gradient updates and weak convolution weight penalties enable rapid memorization of small training cohorts ($N=227$).
- **A1: Gradient Norm Clipping**:
  - Enforce `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` across all gradient accumulation steps.
  - Prevents outlier scans from distorting conv/SSM kernels.
- **A2: Weight Decay Systematic Sweep**:
  - Evaluate $wd \in [0.05, 0.10, 0.20]$ (Baseline is $0.05$).
  - Imposes stronger $L_2$ shrinkage on the 2.08M parameters of 3D ResNet-18.
- **A3: Spatial Attention Pooling Dropout**:
  - Increase pooling attention dropout from $0.10$ to $0.25$.
  - Forces the learnable query $\mathbf{q} \in \mathbb{R}^{48}$ to attend across diverse spatial tokens rather than locking onto fixed anatomical landmarks.
- **A4: Classification Head Dropout**:
  - Evaluate head dropout $p \in [0.5, 0.6]$.

---

### Ablation E5-B: Learning-Rate Schedule Optimization
*Hypothesis*: The baseline `cos_lr` scheduler pulses 3.33 times per epoch (136-step cycle), repeatedly injecting gradient shocks that prevent settling into smooth, flat minima.
- **B1: Monotonic Full-Epoch Cosine Annealing (`mode: cos`)**:
  - Replace high-frequency cycling `cos_lr` with standard monotonic Cosine Annealing:
    $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T_{\max} - t_{\text{warmup}}}\pi\right)\right)$$
- **B2: Epoch-Aware Linear Warmup**:
  - Expand warmup from 5 individual mini-batches to 3 full epochs ($3 \times 454 = 1,362\text{ steps}$), warming up from $1.0\times 10^{-6}$ to $\eta_{\max}$.
  - Allows AdamW second-order moment buffers to stabilize before full-magnitude updates.
- **B3: Peak Learning Rate Search**:
  - Evaluate $\eta_{\max} \in \{5.0\times 10^{-5}, 1.0\times 10^{-4}\}$.

---

### Ablation E5-C: Modality Capacity Rebalancing
*Hypothesis*: sMRI accounts for $64.49\%$ of total model parameters ($2.08\text{M}$), while the Connectome GNN accounts for only $0.23\%$ ($7.3\text{K}$). The network over-indexes on high-capacity anatomical volume memorization.
- **C1: Branch-Specific Dropout**:
  - Apply asymmetric dropout: higher dropout ($0.35$) on 3D ResNet-18 feature maps, and low dropout ($0.10$) on Connectome GNN node features.
- **C2: Connectome GNN Capacity Elevation**:
  - Evaluate expanding Connectome GNN hidden dimensions ($32 \rightarrow 48$), raising its parameter count modestly while boosting functional topology representation.
- **C3: Unimodal & Bimodal Diagnostic Ablations**:
  - Measure validation loss and accuracy under modality-isolated configurations:
    1. sMRI-only (3D ResNet-18)
    2. fMRI-only (ConvNeXtV2)
    3. FC-only (Connectome GNN)
    4. Tri-modal (CDCo-MambaNet V5)
  - Quantifies exact multi-modal synergy and verifies that each modality provides non-redundant diagnostic information.

---

### Ablation E5-D: Scanner-Site Robustness Analysis
*Hypothesis*: NYU comprises $22.0\%$ of training and $24.1\%$ of validation, but only $3.6\%$ ($1$ subject) of test. Training models without site-invariance causes the network to learn acquisition artifacts.
- **D1: Validation Site Disaggregation**:
  - Disaggregate validation performance across sites (NYU vs UM_1 vs UCLA_1 vs others) to detect site-specific accuracy disparity.
- **D2: Feature-Level Instance Normalization**:
  - Evaluate adding Instance Normalization (`nn.InstanceNorm1d`) in feature projection adapters (`smri_change`, `fmri_change`) to strip subject- and scanner-specific intensity offsets prior to SSM fusion without requiring site labels (zero target leakage).

---

### Ablation E5-E: Multimodal Fusion Optimization
*Hypothesis*: Spatial attention pooling query dimension and contrastive perturbation scaling directly regulate the latent biomarker space.
- **E1: Spatial Query Multi-Head Expansion**:
  - Evaluate 8-head spatial pooling ($h=8, d_h=6$) vs 4-head baseline ($h=4, d_h=12$).
- **E2: Dynamic Feature Dimensioning (DFD) Bounds**:
  - Evaluate minimum kept dimension $min\_dim \in [12, 24]$.
- **E3: Contrastive Perturbation Weight ($\lambda_{\text{ct}}$)**:
  - Evaluate $\lambda_{\text{ct}} \in [0.5, 1.0]$.

---

## 4. Multi-Seed Validation Protocol (Seeds 1, 2, 3, 4, 5)

Once the best configuration is identified on Seed 0 (using training and validation data exclusively), the unified configuration will be benchmarked across **all 5 independent seeds**:

- **Splits Used**:
  - `result/phase6_multiseed/splits/exact_splits_seed0.json` (Seed 0)
  - `result/phase6_multiseed/splits/exact_splits_seed1.json` (Seed 1)
  - `result/phase6_multiseed/splits/exact_splits_seed2.json` (Seed 2)
  - `result/phase6_multiseed/splits/exact_splits_seed3.json` (Seed 3)
  - `result/phase6_multiseed/splits/exact_splits_seed4.json` (Seed 4)
- **Cohort per Seed**: Exactly $227$ Train, $29$ Validation, $28$ Held-Out Test ($0$ leakage, stratified $14/14$).
- **Reported Statistics**:
  - Mean $\pm$ Standard Deviation across seeds:
    $$\bar{X} = \frac{1}{S}\sum_{s=1}^S X_s, \quad \sigma = \sqrt{\frac{1}{S-1}\sum_{s=1}^S (X_s - \bar{X})^2}$$
  - Metrics tracked: Accuracy, Balanced Accuracy, ASD Sensitivity, Specificity, Macro Precision, Macro F1, ROC-AUC, ECE, Brier Score, Peak VRAM, Wall-clock Duration.

---

## 5. Execution Manifest & Deliverables Directory Structure

All E5 artifacts will be isolated in dedicated directories:

```
CDCO-Mamba-Research-main/
├── conf/
│   ├── model/
│   │   └── cdcomambanet_v5_generalization.yaml    # E5 Model Config
│   ├── optimizer/
│   │   └── adamw_e5_cosine.yaml                  # E5 Monotonic Cosine Optimizer
│   └── training/
│       └── e5_generalization_training.yaml        # E5 Training Config (Grad Clipping, Patience)
├── result/
│   └── experiment5_generalization/                # Checkpoints & raw logs (Seeds 0-4)
├── reports/
│   └── Finalization/
│       └── E5_Generalization/                     # All final CSVs, reports, and plots
│           ├── E5_Ablation_Matrix_Results.csv
│           ├── E5_MultiSeed_Benchmark_Summary.csv
│           ├── E5_Generalization_Analysis_Report.md
│           ├── E5_Training_Curves.png
│           ├── E5_Validation_Comparison.png
│           └── E5_Scientific_Review.md
└── tools/
    ├── run_e5_ablation.py                         # Single-factor ablation harness
    └── run_e5_multiseed_benchmark.py              # 5-seed benchmark runner
```

---

## 6. Pre-Execution Review & Verification Checklist

- [x] E4 code, checkpoints, and reports verified frozen and protected.
- [x] Isolated directories defined for E5 models, configs, checkpoints, and deliverables.
- [x] Zero test data access guaranteed during training, tuning, and checkpoint selection.
- [x] Checkpoint selection metric strictly locked to minimum validation loss (`val_loss`).
- [x] One-factor-at-a-time ablation plan structured across E5-A, E5-B, E5-C, E5-D, E5-E.
- [x] 5-seed statistical validation protocol formalized using pre-existing split files.
- [x] No promises of artificial 80% accuracy; focus strictly on reproducible, balanced generalization gains.

---

**Approval Request**:  
*This configuration plan is ready for your review. In accordance with strict instructions, no code changes or training runs will occur until you approve this experimental design.*
