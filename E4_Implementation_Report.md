# CDCo-MambaNet V4 — Implementation Report (Experiment E4)

**Document ID**: `E4_Implementation_Report.md`  
**Experiment**: Experiment E4 Unified Calibrated Evidential Cross-Domain Collaborative Mamba Network  
**Model Name**: `CDCo_MambaNet_V4`  
**Configuration File**: [`conf/model/cdcomambanet_v4_unified.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/model/cdcomambanet_v4_unified.yaml)  
**Implementation Directory**: [`models/CDCo_MambaNet_V4/`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/)  
**Isolation Status**: Prior baselines (E1, E2, E3) and their checkpoints remain 100% frozen and unmodified.

---

## 1. Executive Summary & Scientific Motivation

In Experiment E3, parameter right-sizing demonstrated dramatic positive impact on generalization, reducing parameters by 88.49% and elevating test accuracy from 39.29% to 53.57%. However, a detailed validation audit revealed a severe architectural defect inherited from the original dual-head design:

1. **Prediction Discordance (62.07% Disagreement)**:
   On the validation cohort ($N=29$), Head 1 (Class Token) and Head 2 (Spatial MLP) disagreed on **18 out of 29 participants**. Head 1 achieved **64.29% test accuracy**, whereas Head 2 achieved only **42.86%**.
2. **Harmful Mutual Distillation ($\mathcal{L}_{\text{KD}}$)**:
   Symmetric KL divergence forced the two heads to mirror each other. When Head 2 was incorrect, distillation penalized Head 1 for making the correct prediction, injecting destabilizing gradient noise.
3. **Uncalibrated Overconfidence Hijacking**:
   Head 2's unregularized MLP produced elevated logit magnitudes (mean confidence 0.71 vs Head 1's 0.62). In the 50/50 probability ensemble, Head 2 overrode Head 1, dragging test accuracy down by -10.72%.

**Experiment E4** resolves this structural failure by replacing the dual-head architecture with a **Single Unified Calibrated Evidential Head** with Multi-Head Attention Pooling, while eliminating mutual distillation entirely.

---

## 2. Architectural Design of CDCo-MambaNet V4

```
PATIENT BRAIN INPUTS
┌─────────────────────────┬─────────────────────────────┬───────────────────────────┐
│ 1. Structural MRI (sMRI)│ 2. Functional MRI (fMRI)    │ 3. Functional Connectome  │
│    Shape: (1, 96, 96, 96│    Shape: (116, 115)        │    Shape: (116, 116)      │
│    Physical Anatomy     │    Activity Over Time       │    Wiring & Communication │
└────────────┬────────────┴──────────────┬──────────────┴─────────────┬─────────────┘
             │                           │                            │
             ▼                           ▼                            ▼
      3D ResNet-18               Compact ConvNeXtV2           BrainConnectomeGNN
      (2.08M params)             (0.97M params)               (0.036M params)
             │                           │                            │
             ▼                           ▼                            ▼
       Linear(54 -> 48)            Linear(64 -> 48)                   │
             │                           │                            │
             └───────────────────────────┼────────────────────────────┘
                                         ▼
                        Tri-Modal Reliability Gate V4
            (RAF Modality Blending + DFD Dimension Slicing k_min..48)
                                         │
                                         ▼
                 Prepend Class Tokens [s_cls, f_cls, c_cls]
                 Each stream is (B, 129, 48) -> Total 387 tokens
                                         │
                                         ▼
                 Bidirectional Mamba SSM Mixers (r1, r2, r_fc, r3)
                 Cross-Modal Hadamard Interaction: r1 * r2 * r_fc
                                         │
                                         ▼
                     UNIFIED DUAL-STREAM AGGREGATOR
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
      Stream A: Global Class Tokens                           Stream B: Spatial-Topological Tokens
      [f_cls, s_cls, c_cls] in R^{144}                        X_spatial in R^{B x 384 x 48}
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
                             Linear(96, 2) -> Logits
                                         │
                                         ▼
                      Dirichlet Evidence: alpha = Softplus(logits) + 1
                      Expected Probability: p_k = alpha_k / S
                      Vacuous Uncertainty:  u = 2 / S
                      Validation-Governed Temperature Scaling: z / T_val
```

---

## 3. Detailed Component Implementation

### 3.1 Stream A: Global Class-Token Stream
- Fused class tokens from each modality are extracted from the joint Mamba sequence:
  $$\mathbf{z}_{\text{cls}} = [\mathbf{z}_{\text{fmri\_cls}} \,\|\, \mathbf{z}_{\text{smri\_cls}} \,\|\, \mathbf{z}_{\text{fc\_cls}}] \in \mathbb{R}^{B \times 144}$$
- Captures holistic, cross-modally blended summary representations across all three data streams.

### 3.2 Stream B: Spatial Multi-Head Attention Pooling
- Rather than discarding fine-grained spatial information or feeding 384 unregularized tokens into a naive MLP, E4 introduces `SpatialAttentionPooling`:
  - **Learnable Query Vector**: $\mathbf{q} \in \mathbb{R}^{1 \times 1 \times 48}$, initialized with truncated normal ($\sigma = 0.02$), representing the optimal ASD biomarker signature.
  - **Multi-Head Projections**: $h = 4$ attention heads with head dimension $d_k = 12$. Projections $\mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o \in \mathbb{R}^{48 \times 48}$.
  - **Attention Formulation**:
    $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{q} \mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times h \times 1 \times 384}$$
    $$\mathbf{z}_{\text{spatial}} = \mathbf{W}_o \, (\mathbf{A} \mathbf{V}) \in \mathbb{R}^{B \times 48}$$
  - Dynamically weights the most discriminative ROIs and anatomical features while enforcing spatial invariance.

### 3.3 Unified Representation & Evidential Dirichlet Head
- **Concatenation**: $\mathbf{z}_{\text{unified}} = [\mathbf{z}_{\text{cls}} \,\|\, \mathbf{z}_{\text{spatial}}] \in \mathbb{R}^{B \times 192}$.
- **Classifier Pathway**:
  $$\mathbf{h}_1 = \text{GELU}\left(\text{Linear}_{192 \to 96}(\text{LayerNorm}(\mathbf{z}_{\text{unified}}))\right)$$
  $$\mathbf{z}_{\text{logits}} = \text{Linear}_{96 \to 2}(\text{Dropout}_{p=0.5}(\mathbf{h}_1))$$
- **Dirichlet Evidence Computation**:
  $$\mathbf{e} = \text{Softplus}(\mathbf{z}_{\text{logits}}), \quad \boldsymbol{\alpha} = \mathbf{e} + 1, \quad S = \sum_{k=1}^2 \alpha_k$$
  $$\hat{p}_k = \frac{\alpha_k}{S}, \quad u = \frac{2}{S} \in (0, 1]$$

### 3.4 Non-Conflicting Single-Objective Loss Function
- **Mutual Distillation Removed**: All mutual KL divergence terms between distinct heads are completely eliminated.
- **Unified Objective**:
  $$\mathcal{L}_{\text{E4}} = \mathcal{L}_{\text{CE}}(\mathbf{z}_{\text{logits}}, \mathbf{y}) + \lambda_{\text{EDL}} \left( \mathcal{L}_{\text{EDL}}(\mathbf{e}_{\text{head}}, \mathbf{y}, \beta_t) + \mathcal{L}_{\text{gate}}(\mathbf{e}_{\text{modalities}}, \mathbf{y}, \beta_t) \right) + \lambda_{\text{CT}} \mathcal{L}_{\text{contrastive}}$$
  where $\beta_t = \min\left(1.0, \frac{t}{10}\right)$ is the annealing schedule for the Dirichlet non-target regularizer.

### 3.5 Independent Validation Temperature Scaling
- Implemented in [`models/CDCo_MambaNet_V4/calibration.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/calibration.py) via `TemperatureScaler`.
- **Strict Protocol**:
  - Model parameters are frozen after training.
  - Temperature $T$ is optimized strictly on validation logits and targets using Adam with cosine annealing:
    $$\min_T \mathcal{L}_{\text{NLL}}\left(\frac{\mathbf{z}_{\text{val}}}{T}, \mathbf{y}_{\text{val}}\right), \quad T \in [0.05, 10.0]$$
  - Test data is **never** used during calibration. Both raw and calibrated metrics are tracked.

---

## 4. Source Files Created

| File Path | Description | Line Count |
| :--- | :--- | :---: |
| [`models/CDCo_MambaNet_V4/cdcomnet_v4.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/cdcomnet_v4.py) | Main V4 network class, loss function, and calibration interface | 148 |
| [`models/CDCo_MambaNet_V4/calibration.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/calibration.py) | Post-hoc temperature scaling, ECE, Brier, and NLL calibration engine | 176 |
| [`models/CDCo_MambaNet_V4/encoder/mamba_fuse_v4.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/encoder/mamba_fuse_v4.py) | Dual-stream aggregator, Spatial Attention Pooling, Unified Evidential Head | 344 |
| [`models/CDCo_MambaNet_V4/encoder/reliability_v4.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/encoder/reliability_v4.py) | Tri-modal RAF, DFD, Nested Dropout, and Multi-Task Dirichlet EDL loss | 126 |
| [`models/CDCo_MambaNet_V4/encoder/fc_encoder.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/encoder/fc_encoder.py) | BrainConnectomeGNN degree-normalized graph convolutional network | 134 |
| [`models/CDCo_MambaNet_V4/__init__.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/__init__.py) | Package initialization and public symbol exports | 11 |
| [`models/CDCo_MambaNet_V4/encoder/__init__.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/CDCo_MambaNet_V4/encoder/__init__.py) | Encoder package exports | 12 |
| [`conf/model/cdcomambanet_v4_unified.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/model/cdcomambanet_v4_unified.yaml) | Hydra model configuration file | 36 |
| [`tests/test_cdcomnet_v4.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/tests/test_cdcomnet_v4.py) | 10-point comprehensive unit testing suite | 316 |
| [`models/__init__.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/models/__init__.py) | Updated model registry importing `CDCo_MambaNet_V4` | 14 |
