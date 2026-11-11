# CDCo-MambaNet V4 — Parameter Count & Memory Report (Experiment E4)

**Document ID**: `E4_Parameter_Report.md`  
**Model Name**: `CDCo_MambaNet_V4`  
**Total Parameter Count**: **3,226,317**  
**Trainable Parameters**: **3,226,317** (100.0%)  
**Non-Trainable Parameters**: **0**  
**Parameter Reduction vs Baseline**: **-88.36%** (from 27,729,262 down to 3,226,317)  
**Parameter Reduction vs E2**: **-88.40%** (from 27,811,311 down to 3,226,317)

---

## 1. Exact Submodule-by-Submodule Parameter Breakdown

| Module | Submodule / Class | Tensor Transformation | Parameter Count | % of Total Model |
| :--- | :--- | :---: | :---: | :---: |
| **`smri_encoder`** | 3D ResNet-18 (Depth 18, BasicBlocks) | `(B, 1, 96, 96, 96)` $\rightarrow$ `(B, 128, 6, 3, 3)` | **2,078,674** | 64.43% |
| **`fmri_encoder`** | Compact ConvNeXtV2 (Depths `[1,1,2,1]`, Dims `[16..256]`) | `(B, 1, 116, 115)` $\rightarrow$ `(B, 128, 8, 8)` | **969,136** | 30.04% |
| **`fc_encoder`** | BrainConnectomeGNN (Degree-norm GCN + Readout) | `(B, 116, 116)` $\rightarrow$ `(B, 128, 48)` + `(B, 1, 48)` | **36,353** | 1.13% |
| **`predictor`** | Contrastive MLPHead_V3 | `(B, 387, 64)` $\rightarrow$ `(B, 387, 48)` | **20,080** | 0.62% |
| **`mamba_r1`** | Bidirectional Mamba SSM Mixer (sMRI) | `(B, 129, 48)` $\rightarrow$ `(B, 129, 64)` | **20,544** | 0.64% |
| **`mamba_r2`** | Bidirectional Mamba SSM Mixer (fMRI) | `(B, 129, 48)` $\rightarrow$ `(B, 129, 64)` | **20,544** | 0.64% |
| **`mamba_r_fc`** | Bidirectional Mamba SSM Mixer (FC Connectome) | `(B, 129, 48)` $\rightarrow$ `(B, 129, 64)` | **20,544** | 0.64% |
| **`mamba_r3`** | Joint Mamba Vision Mixer | `(B, 387, 48)` $\rightarrow$ `(B, 387, 64)` | **20,544** | 0.64% |
| **`unified_head`** | Unified Evidential Classifier (NEW in E4) | `(B, 192)` $\rightarrow$ `(B, 2)` Dirichlet Evidence | **19,106** | 0.59% |
| **`gate`** | ReliabilityGate_V4 (3 Dirichlet heads + missing tokens) | `[(B, 128, 48)] * 3` $\rightarrow$ `[(B, 128, 48)] * 3` | **7,782** | 0.24% |
| **`spatial_attn_pool`**| Multi-Head Attention Pooling (NEW in E4) | `(B, 384, 48)` $\rightarrow$ `(B, 48)` | **7,104** | 0.22% |
| **`fmri_change`** | Linear Feature Adapter | `(B, 128, 64)` $\rightarrow$ `(B, 128, 48)` | **3,120** | 0.10% |
| **`smri_change`** | Linear Feature Adapter | `(B, 128, 54)` $\rightarrow$ `(B, 128, 48)` | **2,640** | 0.08% |
| **`cls_tokens`** | Learnable Class Tokens `[s, f, c]` | $3 \times (1, 1, 48)$ | **144** | 0.00% |
| **`sl1`** | Contrastive Scale Parameter | Scalar parameter | **1** | 0.00% |
| **`temperature_scaler`** | Post-hoc Temperature Parameter | Scalar parameter | **1** | 0.00% |
| **Total Model** | **`CDCo_MambaNet_V4`** | — | **3,226,317** | **100.00%** |

---

## 2. Cross-Generational Architecture Comparison

| Architectural Property | Baseline (E1) | V2 Full-Scale (E2) | V3 Compact (E3) | V4 Unified (E4) |
| :--- | :---: | :---: | :---: | :---: |
| **fMRI Encoder** | ConvNeXtV2 (24.7M) | ConvNeXtV2 (24.7M) | Compact ConvNeXt (969K) | **Compact ConvNeXt (969K)** |
| **sMRI Encoder** | 3D ResNet-50 (2.9M) | 3D ResNet-50 (2.9M) | 3D ResNet-18 (2.08M) | **3D ResNet-18 (2.08M)** |
| **FC GNN Branch** | None (Unused) | BrainConnectomeGNN (82K) | BrainConnectomeGNN (36K) | **BrainConnectomeGNN (36K)** |
| **Spatial Aggregation**| Raw Flattened Tokens | Raw Flattened Tokens | Raw Flattened Tokens | **Multi-Head Attention Pooling (7K)**|
| **Classification Heads**| Dual Disagreeing Heads | Dual Disagreeing Heads | Dual Disagreeing Heads | **Single Unified Evidential Head (19K)**|
| **Mutual Distillation**| Active ($\mathcal{L}_{\text{KD}}$) | Active ($\mathcal{L}_{\text{KD}}$) | Active ($\mathcal{L}_{\text{KD}}$) | **Completely Removed** |
| **Calibration Module** | None | None | Post-hoc validation audit | **Validation TemperatureScaler** |
| **Total Parameters** | **27,729,262** | **27,811,311** | **3,202,271** | **3,226,317** |
| **Parameter Delta** | Baseline Reference | +0.30% | -88.45% | **-88.36%** |
| **Ratio to Cohort (N=227)**| 122,155 : 1 | 122,517 : 1 | 14,106 : 1 | **14,212 : 1** |

---

## 3. Hardware Resource & GPU Memory Assessment

- **Host GPU**: NVIDIA GeForce MX450 (Dedicated VRAM: 2,048 MB).
- **Batch Size**: 2 (with gradient accumulation = 8, effective batch size = 16).
- **Gradient Checkpointing**: Active across all ResNet-18 and ConvNeXtV2 stages.
- **Estimated Peak VRAM Usage**: **~205 MB** (approx. 10.0% of available GPU memory).
- **Available VRAM Headroom**: **>1,840 MB** (>89.9% safety margin).
- **Out-of-Memory Risk**: **Zero**.
- **Checkpoint File Size on Disk**: **~12.9 MB** (down from 111.5 MB in baseline/E2).
