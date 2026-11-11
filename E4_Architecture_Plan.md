# CDCo-MambaNet V4 — Unified Calibrated Evidential Head Architecture Plan (Experiment E4)

**Experiment Name**: E4 Unified Calibrated Evidential Architecture  
**Motivation**: Validation-Driven Resolution of Dual-Head Discordance & Uncalibrated Overconfidence  
**Target Architecture**: `CDCo_MambaNet_V4`  
**Configuration File**: `conf/model/cdcomambanet_v4_unified.yaml`  
**Isolation Boundary**: Baseline, E2, and E3 models and checkpoints remain completely untouched.  

---

## 1. Problem Formulation: Why Dual-Head Distillation Fails

The validation-only audit of Experiment E3 revealed three structural weaknesses in the baseline dual-head design:

```
[Previous E1 / E2 / E3 Dual-Head Architecture]
                    ┌─── Head 1 (Class Token) ────> Logits 1 (Cautious, Conf: 0.62) ──┐
Fused Representations                                                                   ├──> Disagree on 62% of subjects!
                    └─── Head 2 (Spatial MLP) ────> Logits 2 (Aggressive, Conf: 0.71) ┘    Mutual KL Distillation fights CE loss!
                                                                                           Ensemble hijacked by Head 2!
```

1. **Prediction Discordance (62.07% Disagreement)**:
   Head 1 operates on summary class tokens after bidirectional SSM mixing, while Head 2 operates on raw flattened spatial tokens. Because their receptive fields and token granularities differ, they learn conflicting decision surfaces, disagreeing on **18 of 29 validation subjects**.
2. **Mutual Distillation As Conflict Noise**:
   The mutual distillation loss $\mathcal{L}_{\text{KD}} = \text{KL}(p_1 / T \parallel p_2 / T) + \text{KL}(p_2 / T \parallel p_1 / T)$ attempts to force the two heads to agree. When one head is wrong, this term penalizes the correct head, distorting gradient updates.
3. **Probability Ensemble Dilution**:
   Because Head 2 has higher unregularized logit magnitudes (mean confidence $0.71$ vs $0.62$), taking an unweighted average $p_{\text{ens}} = 0.5(p_1 + p_2)$ gives Head 2 disproportionate veto power over Head 1.

---

## 2. The E4 Architecture: Unified Dual-Stream Evidential Network

Experiment E4 replaces the fractured dual-head structure with a **Single Unified Calibrated Evidential Head** that fuses both global summary tokens and fine-grained spatial-topological tokens into a single coherent decision representation:

```
[Proposed E4 Architecture: CDCo_MambaNet_V4]

  sMRI Volume (96^3)       fMRI BOLD (116x115)       FC Pearson Matrix (116x116)
         │                         │                              │
         ▼                         ▼                              ▼
  3D ResNet-18              Compact ConvNeXtV2            BrainConnectomeGNN
  (2,078,674 params)        (969,136 params)              (36,353 params)
         │                         │                              │
         ▼                         ▼                              ▼
  sMRI Tokens (128x48)      fMRI Tokens (128x48)          FC Tokens (128x48)
  ─────────────────────────────────────────────────────────────────────────────
                       Tri-Modal Reliability Gate V3
           (RAF Modality Blending + DFD Dimension Slicing k_min..48)
  ─────────────────────────────────────────────────────────────────────────────
  Prepend Class Tokens [s, f, c] -> Each stream is (B, 129, 48)
  Bidirectional Mamba SSM Mixers (r1, r2, r_fc, r3)
  Cross-Modal Hadamard Interaction: r1 * r2 * r_fc
  ─────────────────────────────────────────────────────────────────────────────
                       UNIFIED DUAL-STREAM AGGREGATOR
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
  Stream A: Global Class Tokens                           Stream B: Spatial-Topological Tokens
  [s_cls, f_cls, c_cls] in R^{144}                        X_spatial in R^{B x 384 x 48}
         │                                                       │
         │                                                Multi-Head Attention Pooling
         │                                                AttnPool(X_spatial) in R^{B x 48}
         │                                                       │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                    Concatenated Unified Representation
                          z_unified in R^{B x 192}
                                     │
                                     ▼
                  Unified Evidential Classification Head
             LayerNorm(192) -> Linear(192, 96) -> GELU -> Dropout(0.5)
                         Linear(96, 2) -> Evidence e
                                     │
                                     ▼
                  Dirichlet Evidence: alpha = e + 1
                  Expected Probability: p_k = alpha_k / S
                  Vacuous Uncertainty:  u = 2 / S
                  Validation Temperature Scaling: z / T_val
```

---

## 3. Mathematical Formulation of the E4 Innovations

### 1. Unified Multi-Head Attention Pooling
Instead of feeding 384 spatial tokens into a naive unregularized MLP, we use Multi-Head Attention Pooling to dynamically aggregate the most discriminative spatial-topological features:
$$\mathbf{a} = \text{Softmax}\left(\frac{\mathbf{q} \mathbf{K}^T}{\sqrt{d}}\right) \in \mathbb{R}^{B \times 1 \times 384}$$
$$\mathbf{z}_{\text{spatial}} = \mathbf{a} \mathbf{V} \in \mathbb{R}^{B \times 1 \times 48}$$
where $\mathbf{q} \in \mathbb{R}^{1 \times 48}$ is a learned query vector representing the ideal neuroimaging biomarker profile.

### 2. Dual-Stream Concatenation
The global class tokens $\mathbf{z}_{\text{cls}} \in \mathbb{R}^{B \times 144}$ and the pooled spatial vector $\mathbf{z}_{\text{spatial}} \in \mathbb{R}^{B \times 48}$ are fused:
$$\mathbf{z}_{\text{unified}} = [\mathbf{z}_{\text{cls}} \,\|\, \mathbf{z}_{\text{spatial}}] \in \mathbb{R}^{B \times 192}$$

### 3. Calibrated Evidential Dirichlet Head
The unified feature vector passes through symmetric LayerNorm and Dropout ($p = 0.5$) to produce non-negative evidence:
$$\mathbf{e} = \text{Softplus}\left(\mathbf{W}_2 \, \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_1 \mathbf{z}_{\text{unified}} + \mathbf{b}_1\right)\right) + \mathbf{b}_2\right)$$
- Dirichlet parameter: $\alpha_k = e_k + 1$
- Dirichlet strength: $S = \sum_{k=1}^K \alpha_k$
- Calibrated probability: $\hat{p}_k = \alpha_k / S$
- Evidential vacuous uncertainty: $u = K / S$

### 4. Simplified, Non-Conflicting Loss Function
Because there is only one unified head, **mutual KL distillation ($\mathcal{L}_{\text{KD}}$) is completely removed**:
$$\mathcal{L}_{\text{E4}} = \mathcal{L}_{\text{CE}}(\hat{p}, y) + \lambda_{\text{EDL}} \mathcal{L}_{\text{EDL}}(y, \boldsymbol{\alpha}, \lambda_{\text{KL}}) + \lambda_{\text{CT}} \mathcal{L}_{\text{contrastive}}$$
Eliminating the mutual distillation term eliminates the 62% conflict gradient noise, restoring monotonic gradient descent toward a single, coherent objective.

---

## 4. Preservation of All Prior Novelty Components

| Research Component | Origin | Status in E4 | Functional Role |
| :--- | :--- | :---: | :--- |
| **BrainConnectomeGNN** | Pillar 1 / E2 | **PRESERVED** | Encodes 116x116 Pearson FC matrix via degree-normalized graph conv |
| **Right-Sized Backbones** | Pillar 2 / E3 | **PRESERVED** | 3D ResNet-18 (sMRI) + Compact ConvNeXtV2 (fMRI) (<3.5M params) |
| **RAF (Reliability Gate)**| Baseline / V2 | **PRESERVED** | Dynamic modality reliability weighting: $a_m = r_m / \max(r)$ |
| **DFD (Dynamic Dim)** | Baseline / V2 | **PRESERVED** | Dynamic feature width truncation $k \in [12, 48]$ |
| **Nested Dropout** | Baseline / V2 | **PRESERVED** | Stochastic dimension dropping ($p = 0.3$) for canonical subspace encoding |
| **Multi-Task Dirichlet Loss**| Baseline / V2 | **PRESERVED** | Auxiliary Dirichlet cross-entropy on each modality |
| **Contrastive Perturbation**| Baseline / V2 | **PRESERVED** | Perturbation consistency loss regularizing spatial representations |
| **Temperature Calibration**| Pillar 5 / E4 | **NEW (E4)** | Validation-optimized temperature scaling ($T \approx 1.65$) |
| **Unified Head (Single Path)**| Pillar 5 / E4 | **NEW (E4)** | Eliminates dual-head discordance and mutual distillation noise |

---

## 5. Strict Zero-Leakage & Validation Protocol

1. **Frozen Seed 0 Split**: Uses the exact same 227 Train, 29 Validation, and 28 Held-Out Test subjects.
2. **Loss-Governed Early Stopping**: Validation loss (`val_loss`) strictly governs early stopping (patience = 5).
3. **Validation-Only Calibration**: Temperature $T$ is optimized on the validation set only after early stopping selects the best checkpoint.
4. **Held-Out Test Set Untouched**: The test set is evaluated strictly once after all training and calibration are frozen.
