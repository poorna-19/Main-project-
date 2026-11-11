# CDCo-MambaNet V2: Scientific Accuracy Improvement Plan

**Target Architecture**: CDCo-MambaNet V2 (Cross-Domain Collaborative Mamba & Cross-Attention Network)  
**Primary Objective**: Systematically improve ASD classification generalization from ~50–64% towards an 80% experimental target on ABIDE I  
**Author**: Senior Deep Learning Research Scientist, Neuroimaging ML Engineer, and PyTorch Specialist  
**Document Status**: Stage A Proposal — Prepared for User Clearance Prior to Stage B Implementation  

---

## 1. Core Principles: Preserving the Existing Project

1. **Non-Destructive Development**:
   - The entire existing codebase (`models/CDCo_MambaNet`, `dataset`, `conf`, `result`, `demo`) remains completely unmodified and preserved.
   - V2 improvements will reside in a dedicated directory: `models/CDCo_MambaNet_V2/` or parallel modules, leaving existing pipelines functional.
2. **Preserving Core Novelty Components**:
   - **RAF (Reliability-Aware Adaptive Fusion)**: Preserved and enhanced.
   - **DFD (Dynamic Feature Dimension)**: Preserved with adaptive dimension scaling.
   - **Nested Dropout**: Preserved for ordered latent representation.
   - **EDL Weighting**: Dirichlet Evidential uncertainty modeling preserved and temperature-calibrated.
   - **Head Regularization**: LayerNorm and Dropout symmetry retained.
3. **Scientific Grounding Over Hype**:
   - 80% accuracy is an ambitious experimental benchmark. In the High-Dimension Low-Sample-Size (HDLSS) regime of neuroimaging ($N=284$, 17 sites), every 1% gain must be backed by multi-seed statistical significance and zero data leakage.

---

## 2. Six-Pillar Accuracy Improvement Roadmap

```
+---------------------------------------------------------------------------------------------------+
|                                  SIX-PILLAR ACCURACY ROADMAP                                      |
+---------------------------------------------------------------------------------------------------+
|  Pillar 1: Explicit Functional Connectivity Integration                                           |
|  - Connectome Graph Attention / GNN over the 6,670 Pearson edges (currently ignored by deep net).  |
|  - Harnesses the proven 65.5% - 71.4% classical inductive bias directly into deep latents.       |
+---------------------------------------------------------------------------------------------------+
|  Pillar 2: Architecture Parameter Right-Sizing                                                    |
|  - Compress fMRI encoder from 24.7M params down to compact Temporal Transformer / SSM (<1.5M).    |
|  - Reduces total parameter count from 27.7M to ~4M, slashing parameter-to-sample ratio by 85%.    |
+---------------------------------------------------------------------------------------------------+
|  Pillar 3: Domain-Adversarial Scanner-Site Invariance                                             |
|  - Gradient Reversal Layer (GRL) site discriminator across 17 clinical imaging sites.             |
|  - Prevents the network from learning scanner-specific baseline drifts as ASD shortcuts.          |
+---------------------------------------------------------------------------------------------------+
|  Pillar 4: Unified Calibrated Evidential Head                                                     |
|  - Eliminate dual-head discordance where Head 1 and Head 2 cancel each other out.                 |
|  - Deploy temperature-calibrated Dirichlet Evidential Head with uncertainty gating.               |
+---------------------------------------------------------------------------------------------------+
|  Pillar 5: Regularization & Manifold Augmentation                                                 |
|  - Spatio-temporal DropBlock, DropPath stochastic depth, and Manifold Mixup.                      |
|  - SWA (Stochastic Weight Averaging) to find flat, generalizable loss minima.                     |
+---------------------------------------------------------------------------------------------------+
|  Pillar 6: Principled ABIDE I + ABIDE II Dataset Expansion                                        |
|  - External validation on ABIDE II; followed by harmonized multi-site cohort scaling to N > 1,000.|
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Pillar-by-Pillar Scientific Specification

### Pillar 1: Incorporating the Discarded Functional Connectivity (Graph Inductive Bias)
- **Problem**: CDCo-MambaNet takes `fmri_con` in `forward()` but never feeds it into any layer. Yet, static Pearson FC alone achieves **65.48% mean test accuracy** with $L_2$ Logistic Regression and reaches **75.00%** when hybridized with deep latents.
- **Solution**: Introduce a lightweight **Brain Graph Attention Network (BrainGNN)** or **FC Edge MLP**:
  $$\mathbf{H}_{\text{FC}} = \text{GAT}\left(\mathbf{A}_{\text{corr}}, \mathbf{X}_{\text{ROI}}\right) \in \mathbb{R}^{116 \times d}$$
- **Hypothesis**: Giving the model direct access to static topology prevents it from trying to re-learn connectome statistics from scratch on 227 noisy BOLD time-series.

---

### Pillar 2: Parameter-Efficient Model Architecture (Combating Overfitting)
- **Problem**: 27.7M parameters fitting 227 subjects ($122,000 : 1$ parameter-to-sample ratio). The fMRI ConvNeXtV2-Mamba branch alone consumes 24.7M parameters.
- **Solution**:
  - Replace heavy depthwise ConvNeXt blocks with a **Lightweight Temporal Transformer / SSM** ($d_{\text{model}} = 128$, 3 layers, 4 heads, $\sim 0.5\text{M}$ params).
  - Retain 3D ResNet for sMRI with width pruning ($d_{\text{planes}} = [32, 64, 128, 256]$, $\sim 1.5\text{M}$ params).
  - Target model complexity: **$\sim 4.0\text{M}$ total parameters** (an $85.6\%$ parameter reduction).
- **Hypothesis**: Slashing model capacity dramatically narrows the $99\%$ train vs. $50\%$ validation overfitting gap.

---

### Pillar 3: Domain-Adversarial Scanner-Site Invariance
- **Problem**: The 284 subjects originate from 17 independent hospitals and scanners with distinct repetition times ($\text{TR}$), field strengths (3T/1.5T), and coil sensitivities. Deep models easily memorize site identifiers instead of pathology.
- **Solution**: Add a **Gradient Reversal Layer (GRL)** site discriminator:
  $$\mathcal{L}_{\text{site}} = \text{CrossEntropy}\left( \text{Discriminator}_{\text{site}}(\text{GRL}(\mathbf{z}_{\text{fused}})), \mathbf{y}_{\text{site}} \right)$$
  During backpropagation, gradients from the site classifier are reversed by $-\lambda_{\text{GRL}}$, forcing the feature extractors to learn representations that are maximally diagnostic for ASD while minimally predictive of scanner site.
- **Hypothesis**: Removes scanner confounding, improving out-of-site generalization.

---

### Pillar 4: Calibrated Unified Classification Head (Eliminating Discordance)
- **Problem**: Dual heads (`cls_linear` vs. `mlp_cls`) compete and contradict each other. On Seed 3, Head 2 reached 57.14% while Head 1 collapsed to 35.71%, dragging ensemble accuracy down to 39.29%.
- **Solution**:
  - Replace the divergent dual heads with a **single unified Dirichlet Evidential Head**:
    $$\mathbf{\alpha} = \text{Softplus}(\text{MLP}(\mathbf{z}_{\text{fused}})) + 1, \qquad S = \sum_{k=1}^K \alpha_k$$
  - Implement **temperature calibration** ($T$) to prevent negative logit bias and align confidence with empirical risk.
- **Hypothesis**: Eliminates inter-head cancellation and negative probability drift.

---

### Pillar 5: Manifold Regularization & Stochastic Weight Averaging
- **Techniques**:
  1. **DropBlock-3D**: Drops contiguous $3 \times 3 \times 3$ voxel blocks during sMRI encoding to prevent the network from relying on localized skull-stripping artifacts.
  2. **Stochastic Depth (DropPath)**: Randomly skips residual blocks during training.
  3. **Manifold Mixup**: Interpolates intermediate hidden states between subjects to smooth the classification boundary in representation space.
  4. **Stochastic Weight Averaging (SWA)**: Averages model weights across the last 10 epochs to find wider, flatter loss basins that generalize better.

---

## 4. Phase 4: ABIDE I & ABIDE II Dataset Expansion Plan

To move genuinely towards 80% without sample-size constraints, expanding from $N=284$ to the full ABIDE cohort is planned:

| Cohort | Public Availability | Modalities Available | Subject Count | Expected Download Size | Preprocessed Space |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **ABIDE I (Current)** | S3 / PCP Public | sMRI (T1w) + fMRI (DPARSF) | 284 | 1.05 GB (precomputed on E:) | MNI152 (AAL-116) |
| **ABIDE I (Full)** | S3 / PCP Public | sMRI + fMRI | 1,112 | ~45 GB (Raw) / 4.2 GB (NPY) | DPARSF / FreeSurfer |
| **ABIDE II** | NITRC / S3 Public | sMRI (T1w) + rs-fMRI | 1,114 | ~65 GB (Raw) / 4.8 GB (NPY) | CPAC / DPARSF |
| **Combined Target** | Public Open Science | Harmonized Multimodal | **> 2,000** | **~9.0 GB (Processed NPY)** | Unified AAL-116 Atlas |

### Strict Expansion Protocol:
1. **Step 1 (Zero-Shot External Test)**: Train CDCo-MambaNet V2 on ABIDE I ($N=284$), freeze weights, and evaluate on an independent cohort of ABIDE II subjects without retraining to benchmark pure out-of-domain transfer.
2. **Step 2 (Harmonized Pooling)**: Harmonize demographic covariates (Age, Sex, FIQ) and combine ABIDE I + ABIDE II for 5-fold cross-validated scaling.

---

## 5. Experiment Management & Phased Execution Matrix

```
E1: Baseline Reproduction & Multi-Seed Verification (Completed Seeds 0-4)
 └── E2: Explicit Functional Connectivity (GNN / Edge-MLP Integration)
      └── E3: Architecture Right-Sizing (<4M Parameter Compact Model)
           └── E4: Domain-Adversarial Site Invariant Training (GRL)
                └── E5: Unified Calibrated Evidential Head (Zero Discordance)
                     └── E6: Regularization Suite (DropBlock + Manifold Mixup + SWA)
                          └── E7: Complete CDCo-MambaNet V2 5-Seed Benchmark
                               └── E8: Independent ABIDE II Generalization Validation
```

---

## 6. Beginner-Friendly Execution Guide

### What Each Evaluation & Audit Script Does:
1. `tools/evaluate_v2_audit.py`:
   - Loads the exact certified checkpoint (`result/phase6_experiment2_novelty/10-28-13-02-58/best_model.pt`).
   - Evaluates the 28 held-out test patients without retraining or changing any weight.
   - Computes Accuracy, Sensitivity, Specificity, F1, and ROC-AUC.
2. `tools/visualize_audit_metrics.py`:
   - Generates high-resolution confusion matrix heatmaps and ROC curves comparing Baseline, Novelty, and Classical models.
   - Saves clean publication-quality PNG charts.
