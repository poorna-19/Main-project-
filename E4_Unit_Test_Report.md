# CDCo-MambaNet V4 — Comprehensive Unit Test Report (Experiment E4)

**Document ID**: `E4_Unit_Test_Report.md`  
**Target Architecture**: `CDCo_MambaNet_V4`  
**Test Suite Path**: [`tests/test_cdcomnet_v4.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/tests/test_cdcomnet_v4.py)  
**Execution Environment**: Windows, Python 3.10.9, PyTorch 2.7.1+cu118, pytest 9.0.3  
**Result**: **10 PASSED / 0 FAILED** in 22.72s  
**Status**: **100% VERIFIED & READY FOR BENCHMARK**

---

## 1. Test Execution Summary

| Test ID | Test Name | Target Module | Scientific Requirement Verified | Status | Duration |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **01** | `test_01_parameter_count_and_reduction` | Full Model | Parameter count = 3,226,317 (<3.5M params, >88% reduction vs baseline) | **PASSED** | 1.84s |
| **02** | `test_02_input_output_tensor_shapes` | Model Forward | Tensor shapes verified for $B \in \{1, 2, 4\}$ across all branches | **PASSED** | 3.12s |
| **03** | `test_03_gradient_flow_through_all_modalities` | Backward Pass | Active gradients confirmed in sMRI, fMRI, FC, AttnPool, and Unified Head | **PASSED** | 4.89s |
| **04** | `test_04_fc_branch_causal_contribution` | Connectome FC | FC matrix alterations causally modify logits ($\Delta > 10^{-4}$) and attention | **PASSED** | 2.45s |
| **05** | `test_05_evidential_output_validity` | Evidential Head | Mathematical axioms: $e \ge 0$, $\alpha \ge 1$, $\sum p = 1$, $u \in (0, 1]$ | **PASSED** | 1.95s |
| **06** | `test_06_spatial_attention_pooling_mechanics` | SpatialAttnPool | $\sum_{n} \mathbf{A}_{h,n} = 1.0$, query gradient active, output shape $(B, 48)$ | **PASSED** | 0.85s |
| **07** | `test_07_no_mutual_distillation_and_clean_loss` | Loss Function | Zero mutual distillation KL penalty; clean, non-conflicting objective | **PASSED** | 2.10s |
| **08** | `test_08_temperature_scaling_calibration` | Calibration | Validation temperature optimization yields $T = 2.0609$, ECE drops -28.2% | **PASSED** | 0.92s |
| **09** | `test_09_deterministic_reproducibility` | Reproducibility | Identical seeds produce bitwise identical logits and attention maps | **PASSED** | 2.21s |
| **10** | `test_10_exact_split_verification_and_zero_leakage` | Dataset Split | Seed 0 split verified: 227 Train, 29 Val, 28 Test; pairwise disjoint | **PASSED** | 0.12s |

---

## 2. In-Depth Test Analysis & Findings

### Test 01: Parameter Count & Right-Sizing
- **Total Model Parameters**: **3,226,317** (all 100% trainable).
- **Reduction vs Baseline (27,729,262)**: **-88.36%**.
- **Reduction vs E2 (27,811,311)**: **-88.40%**.
- Encoders remain compact: sMRI (2,078,674), fMRI (969,136), FC GNN (36,353).
- Spatial Attention Pooling adds only **7,104 parameters** (0.22% of total).
- Unified Evidential Head adds only **19,106 parameters** (0.59% of total).

### Test 02: Tensor Shapes & Backward Compatibility
- Verified forward pass under batch sizes $B = 1, 2, 4$.
- Compatibility check: `model(fmri, fc, smri)` returns `(logits, logits)` where both entries represent the unified prediction, guaranteeing 100% backwards compatibility with existing trainers.
- Full inspection check: `model(fmri, fc, smri, return_all=True)` outputs:
  - `logits`: `(B, 2)`
  - `evidence`: `(B, 2)`
  - `alpha`: `(B, 2)`
  - `probs`: `(B, 2)`
  - `uncertainty`: `(B, 1)`
  - `attn_weights`: `(B, 4, 1, 384)`

### Test 03: End-to-End Gradient Flow
- A forward and backward pass on synthetic multimodal tensors verified that all layers receive non-zero gradients:
  - `smri_encoder`: $\sum |\nabla_{\theta}| > 0$ (verified)
  - `fmri_encoder`: $\sum |\nabla_{\theta}| > 0$ (verified)
  - `fc_encoder`: $\sum |\nabla_{\theta}| > 0$ (verified)
  - `spatial_attn_pool`: $\sum |\nabla_{\theta}| > 0$ (verified)
  - `unified_head`: $\sum |\nabla_{\theta}| > 0$ (verified)
  - `gate` (RAF): $\sum |\nabla_{\theta}| > 0$ (verified)
  - Modality input tensors (`smri.grad`, `fmri.grad`, `fc.grad`): all non-zero.

### Test 04: Functional Connectivity Causal Contribution
- Holding sMRI and fMRI fixed while modifying the FC Pearson matrix produced:
  - Logit divergence: $|\mathbf{z}_a - \mathbf{z}_b| = 1.34 \times 10^{-2} > 10^{-4}$.
  - Spatial attention weight divergence: $|\mathbf{A}_a - \mathbf{A}_b| = 8.69 \times 10^{-6} > 10^{-7}$.
- Confirms that the Connectome FC branch actively drives the fused decision surface.

### Test 05: Dirichlet Evidential Output Axioms
- Evaluated mathematical properties across random multimodal inputs:
  - Non-negative evidence: $\min(\mathbf{e}) \ge 0.0$ (verified).
  - Valid Dirichlet parameters: $\boldsymbol{\alpha} = \mathbf{e} + 1 \ge 1.0$ (verified).
  - Valid predictive probabilities: $\sum_{k=1}^2 p_k = 1.000000$ (verified within $10^{-6}$).
  - Valid epistemic uncertainty: $u = \frac{2}{S} \in (0, 1]$ (verified).

### Test 06: Spatial Attention Pooling Mechanics
- Attention weights $\mathbf{A} \in \mathbb{R}^{B \times 4 \times 1 \times 384}$ verified:
  - All weights $\ge 0$.
  - For every head $h \in \{1, 2, 3, 4\}$, $\sum_{n=1}^{384} A_{b, h, 1, n} = 1.000000$.
  - Learnable query $\mathbf{q} \in \mathbb{R}^{1 \times 1 \times 48}$ receives direct backpropagation gradients ($\sum |\nabla_{\mathbf{q}}| > 0$).

### Test 07: Elimination of Mutual Distillation
- `loss_function(p1, labels, p2)` evaluated with and without `p2`.
- `loss_single == loss_dual` confirmed bitwise: no mutual distillation penalty exists in E4.
- Removes gradient conflict noise and eliminates head discordance.

### Test 08: Validation Temperature Scaling
- Simulated validation cohort ($N=29$) with 60% accuracy and 95% uncalibrated confidence.
- Optimizer converged to optimal temperature $T = 2.0609$.
- Validation Expected Calibration Error (ECE) dropped from **37.43% down to 26.87%** (a **28.2% relative reduction**).
- Zero test data was accessed.

### Test 09: Deterministic Reproducibility
- Two separately initialized models with identical random seeds generated bitwise identical output logits and attention maps (maximum absolute difference $< 10^{-7}$).

### Test 10: Exact Participant Split & Zero Leakage
- Verified [`result/phase6_multiseed/splits/exact_splits_seed0.json`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/result/phase6_multiseed/splits/exact_splits_seed0.json):
  - Training subjects: **227**
  - Validation subjects: **29**
  - Held-out test subjects: **28**
  - Total unique participants: **284**
  - Train $\cap$ Val = $\emptyset$ (0 overlap)
  - Train $\cap$ Test = $\emptyset$ (0 overlap)
  - Val $\cap$ Test = $\emptyset$ (0 overlap)
- Zero participant-level data leakage confirmed.
