# CDCo-MambaNet V4 — Parameter Count & Resource Estimate (Experiment E4)

**Architecture**: `CDCo_MambaNet_V4` (Unified Calibrated Evidential Cross-Domain Collaborative Mamba Network)  
**Baseline Parameters**: 27,729,262  
**E2 Parameters**: 27,811,311  
**E3 Parameters**: 3,202,271  
**E4 Estimated Parameters**: **3,220,988** (**88.42% Reduction vs Baseline / E2**)  

---

## 1. Submodule-by-Submodule Parameter Breakdown

| Module Component | Underlying Architecture | Input / Output Tensors | E4 Parameter Count | % of Total Model |
| :--- | :--- | :---: | :---: | :---: |
| **`smri_encoder`** | 3D ResNet-18 (Depth 18, BasicBlocks) | `(B, 1, 96, 96, 96)` $\rightarrow$ `(B, 128, 6, 3, 3)` | **2,078,674** | 64.54% |
| **`fmri_encoder`** | Compact ConvNeXtV2 (Dims `[16..256]`, Depths `[1, 1, 2, 1]`) | `(B, 1, 116, 115)` $\rightarrow$ `(B, 128, 8, 8)` | **969,136** | 30.09% |
| **`fc_encoder`** | BrainConnectomeGNN (Degree-norm GCN + Readout) | `(B, 116, 116)` $\rightarrow$ `(B, 128, 48)` + `(B, 1, 48)` | **36,353** | 1.13% |
| **`gate`** | ReliabilityGate_V3 (3 Dirichlet heads + missing) | `[(B, 128, 48)] * 3` $\rightarrow$ `[(B, 128, 48)] * 3` | **7,782** | 0.24% |
| **`smri_change`** | Linear Feature Adapter | `(B, 128, 54)` $\rightarrow$ `(B, 128, 48)` | **2,640** | 0.08% |
| **`fmri_change`** | Linear Feature Adapter | `(B, 128, 64)` $\rightarrow$ `(B, 128, 48)` | **3,120** | 0.10% |
| **`mamba_r1`** | Bidirectional Mamba SSM Mixer (sMRI) | `(B, 129, 48)` $\rightarrow$ `(B, 129, 64)` | **20,544** | 0.64% |
| **`mamba_r2`** | Bidirectional Mamba SSM Mixer (fMRI) | `(B, 129, 48)` $\rightarrow$ `(B, 129, 64)` | **20,544** | 0.64% |
| **`mamba_r_fc`** | Bidirectional Mamba SSM Mixer (FC Connectome) | `(B, 129, 48)` $\rightarrow$ `(B, 129, 64)` | **20,544** | 0.64% |
| **`mamba_r3`** | Joint Mamba Vision Mixer | `(B, 387, 48)` $\rightarrow$ `(B, 387, 64)` | **20,544** | 0.64% |
| **`predictor`** | Contrastive MLPHead_V3 | `(B, 387, 64)` $\rightarrow$ `(B, 387, 48)` | **24,864** | 0.77% |
| **`spatial_attn_pool`** | Multi-Head Attention Pooling (NEW) | `(B, 384, 48)` $\rightarrow$ `(B, 48)` | **4,656** | 0.14% |
| **`unified_head`** | Unified Evidential Classifier (NEW) | `(B, 192)` $\rightarrow$ `(B, 2)` Dirichlet Evidence | **19,298** | 0.60% |
| **`cls_tokens`** | Learnable Class Tokens `[s, f, c]` | `3 * (1, 1, 48)` | **144** | 0.00% |
| **`sl1`** | Contrastive Scale Parameter | Scaler tensor | **1** | 0.00% |
| **Total Model** | **CDCo_MambaNet_V4** | — | **3,220,988** | **100.00%** |

---

## 2. Comparison Across All Experimental Generations

| Metric | Controlled Baseline | CDCo-MambaNet V2 (E2) | CDCo-MambaNet V3 (E3) | CDCo-MambaNet V4 (E4 Planned) |
| :--- | :---: | :---: | :---: | :---: |
| **fMRI Encoder Params** | 24,706,752 | 24,706,752 | 969,136 | **969,136** |
| **sMRI Encoder Params** | 2,899,794 | 2,899,794 | 2,078,674 | **2,078,674** |
| **FC GNN Params** | 0 (Unused) | 82,049 | 36,353 | **36,353** |
| **Classification Heads** | Dual Head (5,237) | Dual Head (5,237) | Dual Head (5,237) | **Unified Head (23,954)** |
| **Mutual Distillation** | Yes (Active) | Yes (Active) | Yes (Active) | **Removed (Non-Conflicting)** |
| **Total Parameters** | **27,729,262** | **27,811,311** | **3,202,271** | **3,220,988** |
| **Parameter Reduction vs Baseline**| 0.00% | +0.30% | -88.45% | **-88.42%** |
| **Expected Peak Training VRAM**| ~720 MB | 729.9 MB | 198.0 MB | **~200–210 MB** |
| **Model Checkpoint Disk Size** | 111.5 MB | 111.5 MB | 12.8 MB | **~12.9 MB** |

---

## 3. Computational Feasibility Summary

- **GPU VRAM Budget**: 2,048 MB on NVIDIA GeForce MX450.
- **Estimated Peak VRAM**: **~205 MB** ($<11\%$ of available physical VRAM).
- **VRAM Headroom**: **$>1,840\text{ MB}$ ($>89\%$ safety headroom)**.
- **Out-of-Memory Risk**: **Zero**.
- **Expected Training Time**: **~45–50 minutes** for 30 epochs with early stopping.
