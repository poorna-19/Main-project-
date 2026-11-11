# CDCo-MambaNet Accuracy Improvement & Scientific Research Plan

**Author:** Senior Deep Learning Research Scientist  
**Focus Area:** Multimodal Neuroimaging, Autism Spectrum Disorder (ASD) Classification, Mamba & Transformer Architectures  
**Dataset:** ABIDE I Cohort ($N=284$, 140 ASD, 144 Healthy Control, 17 Scanner Sites)  
**Hardware Target:** NVIDIA GeForce MX450 (2,048 MB VRAM budget)  
**Date:** November 1, 2026  
**Status:** Working Proposal (Pre-Training Clearance Required)

---

## 1. Executive Summary & Diagnostic Findings

Classification of Autism Spectrum Disorder (ASD) from the Autism Brain Imaging Data Exchange (ABIDE I) dataset represents one of the most notoriously difficult benchmarks in computational neuroscience. With only $N=284$ subjects distributed across 17 heterogeneous clinical scanner sites, deep neural architectures face severe challenges.

Our independent empirical audit and multi-seed execution of **CDCo-MambaNet** (Phase 6 Experiment 2) have surfaced critical, definitive findings:

| Model Paradigm | Seed 0 Test | Seed 1 Test | Seed 2 Test | Mean (Seeds 0–2) | Mechanism / Primary Feature Space |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CDCo-MambaNet (End-to-End Deep)** | 64.29% | 50.00% | 50.00% | **54.76%** | ConvNeXtV2-Mamba + 3D ResNet-50 + RAF |
| **Classical Ridge Classifier** | 71.43% | 67.86% | 57.14% | **65.48%** | 6,670 static Pearson Functional Connectivity |
| **Classical L2 Logistic Regression** | 71.43% | 64.29% | 60.71% | **65.48%** | 6,670 static Pearson Functional Connectivity |
| **Hybrid FC + Deep (L2 Logistic)** | **67.86%** | **64.29%** | **75.00%** | **69.05%** | 6,670 FC + 424 Deep Latent Representations |

### Key Empirical Revelations
1. **The Overfitting Catastrophe (Generalization Gap > 45%):**  
   In Seed 2 and Seed 3, training accuracy reaches **96.5% – 99.2%** within 6–7 epochs, while validation accuracy stagnates between **37.9% and 58.6%**, and held-out test accuracy drops to **50.0%** (chance level). The model has 27.7 million parameters fitting only 227 training subjects.
2. **Static Functional Connectivity Dominance:**  
   Classical regularized linear classifiers (Ridge, L2 Logistic Regression) operating strictly on the 6,670 upper-triangle Pearson correlation features consistently outperform the 27.7M-parameter deep network by **+10.7 percentage points** on average ($65.48\%$ vs $54.76\%$).
3. **The Hybrid Latent Synergy:**  
   Concatenating the static 6,670 Pearson FC features with the 424 deep latent features extracted by CDCo-MambaNet yields **69.05% mean accuracy** across seeds (peaking at **75.00%** on Seed 2). This demonstrates that CDCo-MambaNet *does* extract complementary structural and temporal representations, but its end-to-end classification heads (`cls_linear` and `mlp_cls`) overfit and fail to generalize on the small training sample size.

---

## 2. Root Cause Analysis: The Neuroimaging Deep Learning Trilemma

```
                      [High-Dimensional Modalities]
                     (sMRI: 96^3, fMRI: 116x115)
                                  /\
                                 /  \
                                /    \
                               /      \
    [Extreme Sample Scarcity] /________\ [Multi-Site Heterogeneity]
     (N = 227 Train Subjects)             (17 Clinical Scanner Sites)
```

### 2.1 The High-Dimension Low-Sample-Size (HDLSS) Regime
In ABIDE I, each subject possesses:
- **sMRI**: $96 \times 96 \times 96 = 884,736$ spatial voxels.
- **fMRI**: $116 \times 115 = 13,340$ time-series points.
- **Pearson FC**: $116 \times 115 / 2 = 6,670$ correlation coefficients.

With only 227 training subjects per seed split, the input space dimensionality ($D \approx 9 \times 10^5$) exceeds the number of observations by nearly $4,000\times$. Unconstrained 3D convolutions and bidirectional Mamba state-space blocks rapidly memorize individual subject idiosyncrasies rather than invariant neurobiological biomarkers of ASD.

### 2.2 Scanner Site Covariate Shift
The 284 subjects originate from 17 independent imaging centers (e.g., NYU, KKI, UCLA, USM, Stanford, Pitt). Each site utilizes different:
- Scanner hardware (Siemens Allegra 3T, Trio 3T, GE Signa 3T, Philips Achieva 3T).
- Repetition times ($\text{TR}$ ranging from $1,500\,\text{ms}$ to $3,000\,\text{ms}$).
- Voxel resolutions, head coil configurations, and pre-processing artifacts.
- Demographics (age distributions, IQ ranges, male/female ratios).

Standard empirical risk minimization (ERM) forces the deep feature extractor to learn site-discriminative shortcuts (e.g., identifying scanner baseline drift or slice timing differences) instead of ASD pathology.

### 2.3 Dual-Head Asymmetry & Probability Bias
CDCo-MambaNet deploys two classification heads:
- **Head 1 (`cls_linear`):** A linear projection on the flattened spatio-temporal features.
- **Head 2 (`mlp_cls`):** A multi-layer perceptron receiving fused multimodal tokens after RAF.

Our audit revealed that Head 1 and Head 2 often predict conflicting classes. Furthermore, in the baseline demo model, Head 2 suffered from a negative probability bias, predicting Control for 100% of inputs. In Phase 6 Experiment 2, adding LayerNorm and Dropout (0.5) to Head 1 stabilized training, but the ensemble still collapses when both heads overfit the 227 training samples.

---

## 3. Systematic Accuracy Improvement Strategy

To elevate held-out test generalization from ~55% to $\ge 72\%$, we propose a multi-pillar research strategy:

```
+---------------------------------------------------------------------------------------------------+
|                                  SIX-PILLAR ACCURACY ROADMAP                                      |
+---------------------------------------------------------------------------------------------------+
|  Pillar 1: Hybrid Feature Integration                                                             |
|  - Retain 6,670 static Pearson FC as explicit inductive bias.                                     |
|  - Combine with compact deep morphological and temporal embeddings.                               |
+---------------------------------------------------------------------------------------------------+
|  Pillar 2: CDCo-CrossAttentionNet Architecture                                                    |
|  - Bidirectional Cross-Modal Attention between sMRI morphometry and fMRI dynamics.                |
|  - Graph Convolutional / Attention Network for Functional Connectivity topology.                  |
+---------------------------------------------------------------------------------------------------+
|  Pillar 3: Domain-Adversarial Scanner Invariance                                                  |
|  - Gradient Reversal Layer (GRL) site discriminator to remove scanner-specific features.          |
+---------------------------------------------------------------------------------------------------+
|  Pillar 4: Structural Regularization & Manifold Augmentation                                      |
|  - CutMix / Manifold Mixup in latent space; DropBlock-3D; Stochastic Depth.                      |
+---------------------------------------------------------------------------------------------------+
|  Pillar 5: Calibrated Evidential Uncertainty & Single Unified Head                                |
|  - Replace conflicting dual heads with a single unified, temperature-calibrated Dirichlet head.   |
+---------------------------------------------------------------------------------------------------+
|  Pillar 6: Strict Multi-Seed Zero-Leakage Experimental Governance                                 |
|  - Retain locked 5-seed splits; strictly evaluate held-out test sets only post-training.          |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Execution Plan & Phased Milestones

### Phase A: Complete & Audit Ongoing 5-Seed Execution (CURRENT)
- **Status:** Background task `task-114` active. Seeds 0, 1, 2 completed; Seed 3 running; Seed 4 pending.
- **Action:** Allow `task-114` to complete naturally. Zero interference. Zero GPU competition.
- **Deliverable:** Generate comprehensive statistical verification via `tools/verify_fiveseed_results.py`.

### Phase B: Performance Analysis & Failure Mode Characterization
- Compile all five seeds: Mean $\pm$ Pop SD $\pm$ Sample SD.
- Cross-tabulate site-by-site error rates across all 17 sites.
- Deliverable: `reports/five_seed_performance_analysis.md`.

### Phase C: Isolated Architecture Implementation (`CDCo-CrossAttentionNet`)
- Implement the new model in an isolated directory: `models/CDCo_CrossAttentionNet/`.
- Ensure strict parameter budget ($\le 5\text{M}$ parameters vs $27.7\text{M}$ previously) to prevent HDLSS memorization.
- Fit within the 2,048 MB VRAM budget on NVIDIA GeForce MX450.
- Deliverable: `reports/new_architecture_design.md`.

### Phase D: Rigorous Comparative Benchmarking Protocol
- Design identical 5-seed, stratified, zero-leakage evaluation protocol.
- Integrate classical baselines (Ridge, SVM, Logistic Regression) alongside Hybrid models and deep architectures.
- Deliverable: `reports/model_comparison_protocol.md`.

### Phase E: Experimental Clearance Gate
- **MANDATORY:** Present all designs and reports to the user.
- **No training of new models will begin without explicit user authorization.**

---

## 5. Risk Assessment & Scientific Guardrails

1. **Risk of Data Leakage:**  
   *Mitigation:* All preprocessing (z-scoring, whitening, graph construction) must be computed exclusively on the training fold of each seed and applied out-of-sample to validation and test folds.
2. **Risk of VRAM Overflow (Out of Memory):**  
   *Mitigation:* The NVIDIA MX450 has 2,048 MB VRAM. Full 3D attention across $96^3$ voxels is computationally impossible. We enforce spatial feature pooling ($8\times 8\times 8$ or global anatomical ROI pooling) before cross-attention.
3. **Risk of Benchmark Fabrication:**  
   *Mitigation:* Every metric reported must trace back to verifiable JSON artifacts (`final.json`, `linear_probe_seed*.json`) generated during execution.
