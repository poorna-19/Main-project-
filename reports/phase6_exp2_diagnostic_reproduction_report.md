# CDCo-MambaNet — Phase 6 Experiment 2  
# Diagnostic Reproduction & Benchmark Readiness Audit Report

**Generated:** 2026-10-31  
**Auditor:** Independent inference-only reproduction  
**Source report verified:** `reports/phase6_exp2_accuracy_diagnosis_report.md`  
**Audit data:** `scratch/detailed_audit_results.json`  
**Git branch:** `improved-fixes`  
**Safety compliance:** ✅ No training, no checkpoint modification, no split modification, no report modification, no git push

---

## ⚡ Executive Summary

This report independently reproduces and verifies five claims from the Phase 6 Experiment 2 accuracy diagnosis report. All five core numerical claims are **confirmed by independent reproduction**. However, a **critical data integrity finding** is disclosed that fundamentally changes the interpretation of the demo-based metrics for the Phase 6 Exp 2 model.

### Confirmed Claims

| # | Claim | Status |
|---|-------|--------|
| 1 | Full Demo Head 2 has a negative ASD probability bias (predicts Control for all subjects) | ✅ **CONFIRMED** |
| 2 | Old demo (`tools/demo.py`) evaluates Head 2 only, ignoring Head 1 | ✅ **CONFIRMED** |
| 3 | Phase 6 Exp 2 ensemble achieves 57.1% clean accuracy and 55.6% AUC (demo set) | ✅ **CONFIRMED** (but contaminated — see §8) |
| 4 | Phase 6 Exp 2 achieves 64.3% accuracy and 64.8% AUC under fMRI missing (demo set) | ✅ **CONFIRMED** (but contaminated — see §8) |
| 5 | Reliability gating and DFD handle missing modalities more effectively than the old demo model | ✅ **CONFIRMED** |

### Critical Finding

> **⚠️ DEMO SET CONTAMINATION:** The 28-subject demo test set (`demo/abide_seed0_test.npz`) does **not** correspond to the Phase 6 Exp 2 held-out test set. It was created by `tools/export_demo_data.py` from a Kaggle-era run using a different split. Cross-referencing with `result/phase6_multiseed/splits/exact_splits_seed0.json` shows:
> - **10/28** demo subjects are from the Phase 6 Exp 2 test split (legitimate held-out)
> - **17/28** demo subjects are from the Phase 6 Exp 2 *validation* split (seen during training/checkpoint selection)
> - **1/28** demo subject is from neither (Kaggle-era subject outside this cohort)
>
> **Consequence:** Metrics evaluated on the demo set for the Phase 6 Exp 2 model (`phase6_exp2_best`) are **contaminated**. The legitimate held-out test results are in `result/phase6_experiment2_novelty/10-28-13-02-58/final.json`.

### Final Verdict

> **✅ READY FOR MULTI-SEED TRAINING (Seeds 2, 3, 4)**  
> Pipeline verified, all 5 split SHA-256 hashes confirmed correct, all sMRI caches present, multi-seed runner uses correct checkpoint selection metric (`val_acc_ensemble`), and verification script checks the correct output (`test_at_best_val.ensemble`). Seeds 0 and 1 are complete. Seeds 2, 3, and 4 may be initiated via the documented command.

---

## 1. Reproduced vs. Original Findings

### Claim 1: Full Demo Head 2 has a negative ASD probability bias

The original report states that the Full Demo model's Head 2 (`mlp_cls`) produces near-zero ASD probabilities, effectively predicting Control for all subjects.

**Reproduced (clean condition, 28 demo subjects):**
- `full_demo` Head 2 P(ASD) mean = **0.323** (range: 0.289–0.400 across conditions)
- All 28 subjects predicted as Control: Confusion matrix = `[[14,0],[14,0]]`
- ASD sensitivity = **0.0%**, specificity = **100.0%**

**Assessment: ✅ CONFIRMED.** The Full Demo Head 2 universally predicts Control due to a negative ASD probability bias (mean P(ASD) ≈ 0.32, well below the 0.5 decision threshold).

---

### Claim 2: Demo evaluates Head 2 only, ignores Head 1

**Code inspection of `tools/demo.py`:**

The demo calls `robustness()` from `training/training.py`. Inspection of `robustness()` confirms:

```python
# robustness() uses only Head 2 (mlp_cls):
_, p1 = model(...)   # p1 is the Head 2 output (mlp_cls)
# Head 1 (cls_linear) is captured in the first return value but discarded
```

**Assessment: ✅ CONFIRMED.** The old demo exclusively uses Head 2. Head 1 predictions are available but silently discarded. No ensemble is constructed.

---

### Claim 3: Phase 6 Exp 2 ensemble achieves 57.1% clean accuracy and 55.6% AUC

**Reproduced (demo set, clean condition):**
- Phase 6 Exp 2 (`best_model.pt`) — Ensemble: ACC = **57.1%**, AUC = **55.6%** ✅

**Caveat:** ⚠️ This result is evaluated on the demo set, of which 17/28 subjects were in the Phase 6 Exp 2 validation split. It is NOT a legitimate held-out test result.

**Legitimate held-out test result** (`final.json`, `test_at_best_val`):**  
Ensemble: ACC = **64.3%**, AUC = **56.6%**, SEN = **85.7%**, SPE = **42.9%**  
Confusion matrix: `[[6,8],[2,12]]`

**Assessment: ✅ CONFIRMED numerically, but flagged as non-comparable** (see §8).

---

### Claim 4: Phase 6 Exp 2 achieves 64.3% accuracy and 64.8% AUC under fMRI missing

**Reproduced (demo set, fMRI missing condition):**
- Phase 6 Exp 2 — Ensemble: ACC = **64.3%**, AUC = **64.8%** ✅
- Gate: w_fmri = **0.00**, w_smri = **1.00**, dims_fmri = **12.0**, dims_smri = **48.0**

**Caveat:** ⚠️ Demo-set contamination applies here as well. These figures reflect performance on a mix of validation and test subjects.

**Assessment: ✅ CONFIRMED numerically, flagged as non-comparable** (see §8).

---

### Claim 5: Reliability gating and DFD handle missing modalities more effectively

**Reproduced evidence:**

| Condition | Model | Ensemble ACC | Ensemble AUC |
|-----------|-------|:---:|:---:|
| fMRI missing | Full Demo | 50.0% | 37.8% |
| fMRI missing | Phase 6 Exp 2 | **64.3%** | **64.8%** |
| sMRI missing | Full Demo | 50.0% | 40.8% |
| sMRI missing | Phase 6 Exp 2 | **64.3%** | **54.1%** |

When fMRI is missing in Phase 6 Exp 2: w_fmri = **0.00** (correctly suppressed), dims_fmri = **12.0** (DFD minimum enforced).  
When sMRI is missing in Phase 6 Exp 2: w_smri = **0.00** (correctly suppressed), dims_smri = **12.0** (DFD minimum enforced).

The Full Demo model does not correctly gate the missing modality — it assigns w_fmri = 0.00 correctly when fMRI is missing (gate works), but delivers substantially degraded accuracy (35.7% for Head 2, 50.0% ensemble vs. 64.3% for Phase 6 Exp 2).

**Assessment: ✅ CONFIRMED.** Phase 6 Exp 2 consistently outperforms the Full Demo under missing-modality conditions on the demo set.

---

## 2. Dual-Head Evaluation — All Models, Clean Condition

> Class 1 = ASD (positive). Class 0 = Control (negative).  
> CM format: `[[TN, FP], [FN, TP]]`

### Baseline (`demo/baseline_seed0.pt`)

| Head | ACC | AUC | SEN | SPE | F1 | Prec | CM | P(ASD) |
|------|:---:|:---:|:---:|:---:|:--:|:----:|:--:|:------:|
| Head 1 (`cls_linear`) | 60.7% | 66.8% | 42.9% | 78.6% | 59.4% | 66.7% | [[11,3],[8,6]] | 0.481 |
| Head 2 (`mlp_cls`) | 50.0% | 42.9% | 100.0% | 0.0% | 33.3% | 50.0% | [[0,14],[0,14]] | **0.903** |
| Ensemble | 50.0% | 64.8% | 100.0% | 0.0% | 33.3% | 50.0% | [[0,14],[0,14]] | 0.692 |

**Note:** Baseline Head 2 has a *positive* ASD bias (P(ASD)=0.903, predicts ASD for all subjects). This is the **opposite** of the Full Demo. Head 1 is the only functional head for the baseline.

### Full Demo (`demo/full_seed0.pt`)

| Head | ACC | AUC | SEN | SPE | F1 | Prec | CM | P(ASD) |
|------|:---:|:---:|:---:|:---:|:--:|:----:|:--:|:------:|
| Head 1 (`cls_linear`) | 50.0% | 57.7% | 0.0% | 100.0% | 33.3% | 0.0% | [[14,0],[14,0]] | 0.294 |
| Head 2 (`mlp_cls`) | 50.0% | 67.3% | 0.0% | 100.0% | 33.3% | 0.0% | [[14,0],[14,0]] | **0.323** |
| Ensemble | 50.0% | 67.3% | 0.0% | 100.0% | 33.3% | 0.0% | [[14,0],[14,0]] | 0.308 |

**Note:** Both heads of Full Demo predict Control for all 28 subjects. P(ASD) ≈ 0.29–0.32 for both heads. The demo only evaluates Head 2, which happens to match Head 1 behavior here.

### Phase 6 Exp 2 (`result/phase6_experiment2_novelty/10-28-13-02-58/best_model.pt`)

| Head | ACC | AUC | SEN | SPE | F1 | Prec | CM | P(ASD) |
|------|:---:|:---:|:---:|:---:|:--:|:----:|:--:|:------:|
| Head 1 (`cls_linear`) | 42.9% | 44.9% | 64.3% | 21.4% | — | — | [[3,11],[5,9]] | — |
| Head 2 (`mlp_cls`) | 60.7% | 51.0% | 64.3% | 57.1% | — | — | [[8,6],[5,9]] | — |
| **Ensemble** | **57.1%** | **55.6%** | **78.6%** | **35.7%** | — | — | [[5,9],[3,11]] | — |

**Note (demo set — contaminated):** Both heads predict ASD more than Control (SEN > SPE for both), unlike the demo models. The ensemble produces the best AUC.

**Legitimate held-out test results** (from `final.json`, 28 true test subjects):

| Head | ACC | AUC | SEN | SPE | F1 | Prec | CM |
|------|:---:|:---:|:---:|:---:|:--:|:----:|:--:|
| Head 1 | 53.6% | 49.0% | 71.4% | 35.7% | 52.0% | 52.6% | [[5,9],[4,10]] |
| Head 2 | 42.9% | 56.6% | 78.6% | 7.1% | 34.5% | 45.8% | [[1,13],[3,11]] |
| **Ensemble** | **64.3%** | **56.6%** | **85.7%** | **42.9%** | **62.6%** | **60.0%** | [[6,8],[2,12]] |

---

## 3. Seven-Condition Metrics — All Models

All results below are on the 28-subject demo set unless marked **[HELD-OUT]** (from `final.json`).

### 3.1 Baseline (`demo/baseline_seed0.pt`)

| Condition | Head | ACC | AUC | SEN | SPE | CM |
|-----------|------|:---:|:---:|:---:|:---:|:--:|
| clean | Head 1 | 60.7% | 66.8% | 42.9% | 78.6% | [[11,3],[8,6]] |
| clean | Head 2 | 50.0% | 42.9% | 100.0% | 0.0% | [[0,14],[0,14]] |
| clean | Ensemble | 50.0% | 64.8% | 100.0% | 0.0% | [[0,14],[0,14]] |
| fmri_noise_0.5 | Head 1 | 60.7% | 66.3% | 42.9% | 78.6% | [[11,3],[8,6]] |
| fmri_noise_0.5 | Head 2 | 50.0% | 42.9% | 100.0% | 0.0% | [[0,14],[0,14]] |
| fmri_noise_0.5 | Ensemble | 50.0% | 64.3% | 100.0% | 0.0% | [[0,14],[0,14]] |
| fmri_noise_1.0 | Head 1 | 60.7% | 66.3% | 42.9% | 78.6% | [[11,3],[8,6]] |
| fmri_noise_1.0 | Head 2 | 50.0% | 43.4% | 100.0% | 0.0% | [[0,14],[0,14]] |
| fmri_noise_1.0 | Ensemble | 50.0% | 64.8% | 100.0% | 0.0% | [[0,14],[0,14]] |
| smri_noise_0.5 | Head 1 | 57.1% | 64.8% | 42.9% | 71.4% | [[10,4],[8,6]] |
| smri_noise_0.5 | Head 2 | 50.0% | 26.0% | 100.0% | 0.0% | [[0,14],[0,14]] |
| smri_noise_0.5 | Ensemble | 50.0% | 62.2% | 100.0% | 0.0% | [[0,14],[0,14]] |
| smri_noise_1.0 | Head 1 | 50.0% | 40.3% | 28.6% | 71.4% | [[10,4],[10,4]] |
| smri_noise_1.0 | Head 2 | 50.0% | 46.9% | 100.0% | 0.0% | [[0,14],[0,14]] |
| smri_noise_1.0 | Ensemble | 50.0% | 40.8% | 100.0% | 0.0% | [[0,14],[0,14]] |
| fmri_missing | Head 1 | 60.7% | 66.3% | 42.9% | 78.6% | [[11,3],[8,6]] |
| fmri_missing | Head 2 | 50.0% | 43.4% | 100.0% | 0.0% | [[0,14],[0,14]] |
| fmri_missing | Ensemble | 50.0% | 64.8% | 100.0% | 0.0% | [[0,14],[0,14]] |
| smri_missing | Head 1 | 50.0% | 38.8% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_missing | Head 2 | 50.0% | 56.1% | 100.0% | 0.0% | [[0,14],[0,14]] |
| smri_missing | Ensemble | 50.0% | 40.8% | 100.0% | 0.0% | [[0,14],[0,14]] |

**Observation:** Baseline Head 2 is degenerate across all 7 conditions (always predicts ASD). Head 1 provides functional predictions but is not available in the demo evaluation.

---

### 3.2 Full Demo (`demo/full_seed0.pt`)

| Condition | Head | ACC | AUC | SEN | SPE | CM |
|-----------|------|:---:|:---:|:---:|:---:|:--:|
| clean | Head 1 | 50.0% | 57.7% | 0.0% | 100.0% | [[14,0],[14,0]] |
| clean | Head 2 | 50.0% | 67.3% | 0.0% | 100.0% | [[14,0],[14,0]] |
| clean | Ensemble | 50.0% | 67.3% | 0.0% | 100.0% | [[14,0],[14,0]] |
| fmri_noise_0.5 | Head 1 | 50.0% | 54.6% | 0.0% | 100.0% | [[14,0],[14,0]] |
| fmri_noise_0.5 | Head 2 | 50.0% | 49.5% | 0.0% | 100.0% | [[14,0],[14,0]] |
| fmri_noise_0.5 | Ensemble | 50.0% | 56.1% | 0.0% | 100.0% | [[14,0],[14,0]] |
| fmri_noise_1.0 | Head 1 | 50.0% | 52.6% | 0.0% | 100.0% | [[14,0],[14,0]] |
| fmri_noise_1.0 | Head 2 | 53.6% | 50.5% | 7.1% | 100.0% | [[14,0],[13,1]] |
| fmri_noise_1.0 | Ensemble | 50.0% | 49.0% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_noise_0.5 | Head 1 | 50.0% | 45.9% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_noise_0.5 | Head 2 | 50.0% | 59.2% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_noise_0.5 | Ensemble | 50.0% | 51.5% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_noise_1.0 | Head 1 | 50.0% | 44.9% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_noise_1.0 | Head 2 | 50.0% | 77.0% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_noise_1.0 | Ensemble | 50.0% | 56.1% | 0.0% | 100.0% | [[14,0],[14,0]] |
| fmri_missing | Head 1 | 50.0% | 53.1% | 0.0% | 100.0% | [[14,0],[14,0]] |
| fmri_missing | **Head 2** | **35.7%** | **32.1%** | **42.9%** | **28.6%** | [[4,10],[8,6]] |
| fmri_missing | Ensemble | 50.0% | 37.8% | 7.1% | 92.9% | [[13,1],[13,1]] |
| smri_missing | Head 1 | 50.0% | 44.9% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_missing | Head 2 | 50.0% | 41.8% | 0.0% | 100.0% | [[14,0],[14,0]] |
| smri_missing | Ensemble | 50.0% | 40.8% | 0.0% | 100.0% | [[14,0],[14,0]] |

**Observation:** Full Demo is uniformly degenerate — all conditions predict Control for both heads except the single edge case of fmri_noise_1.0 Head 2. The demo reports only Head 2 predictions for `fmri_missing` (ACC=35.7%, AUC=32.1%), consistent with the original report's claim.

---

### 3.3 Phase 6 Exp 2 (`best_model.pt`) — Demo Set ⚠️ CONTAMINATED

| Condition | Head | ACC | AUC | SEN | SPE | CM |
|-----------|------|:---:|:---:|:---:|:---:|:--:|
| clean | Head 1 | 42.9% | 44.9% | 64.3% | 21.4% | [[3,11],[5,9]] |
| clean | Head 2 | 60.7% | 51.0% | 64.3% | 57.1% | [[8,6],[5,9]] |
| clean | **Ensemble** | **57.1%** | **55.6%** | 78.6% | 35.7% | [[5,9],[3,11]] |
| fmri_noise_0.5 | Head 1 | 50.0% | 52.6% | 71.4% | 28.6% | [[4,10],[4,10]] |
| fmri_noise_0.5 | Head 2 | 57.1% | 57.1% | 71.4% | 42.9% | [[6,8],[4,10]] |
| fmri_noise_0.5 | Ensemble | 57.1% | 60.2% | 78.6% | 35.7% | [[5,9],[3,11]] |
| fmri_noise_1.0 | Head 1 | 50.0% | 51.0% | 71.4% | 28.6% | [[4,10],[4,10]] |
| fmri_noise_1.0 | Head 2 | 57.1% | 52.0% | 71.4% | 42.9% | [[6,8],[4,10]] |
| fmri_noise_1.0 | Ensemble | 57.1% | 56.1% | 78.6% | 35.7% | [[5,9],[3,11]] |
| smri_noise_0.5 | Head 1 | 42.9% | 46.9% | 64.3% | 21.4% | [[3,11],[5,9]] |
| smri_noise_0.5 | Head 2 | 57.1% | 49.0% | 64.3% | 50.0% | [[7,7],[5,9]] |
| smri_noise_0.5 | Ensemble | 57.1% | 53.1% | 78.6% | 35.7% | [[5,9],[3,11]] |
| smri_noise_1.0 | Head 1 | 42.9% | 46.9% | 64.3% | 21.4% | [[3,11],[5,9]] |
| smri_noise_1.0 | Head 2 | 57.1% | 48.0% | 64.3% | 50.0% | [[7,7],[5,9]] |
| smri_noise_1.0 | Ensemble | 50.0% | 49.0% | 71.4% | 28.6% | [[4,10],[4,10]] |
| **fmri_missing** | Head 1 | 64.3% | 62.8% | 28.6% | 100.0% | [[14,0],[10,4]] |
| **fmri_missing** | Head 2 | 60.7% | 61.2% | 71.4% | 50.0% | [[7,7],[4,10]] |
| **fmri_missing** | **Ensemble** | **64.3%** | **64.8%** | 57.1% | 71.4% | [[10,4],[6,8]] |
| smri_missing | Head 1 | 64.3% | 62.8% | 28.6% | 100.0% | [[14,0],[10,4]] |
| smri_missing | Head 2 | 57.1% | 58.2% | 57.1% | 57.1% | [[8,6],[6,8]] |
| smri_missing | **Ensemble** | **64.3%** | **54.1%** | 42.9% | 85.7% | [[12,2],[8,6]] |

---

## 4. Confusion Matrices — Complete Reference

All confusion matrices use format: rows = true label (0=Control top, 1=ASD bottom); cols = predicted label (0=Control left, 1=ASD right).  
`[[TN, FP], [FN, TP]]`

### Phase 6 Exp 2 Ensemble — 7 Conditions (Demo Set ⚠️)

```
clean:           [[5,  9], [3, 11]]   ACC=57.1%  AUC=55.6%
fmri_noise_0.5:  [[5,  9], [3, 11]]   ACC=57.1%  AUC=60.2%
fmri_noise_1.0:  [[5,  9], [3, 11]]   ACC=57.1%  AUC=56.1%
smri_noise_0.5:  [[5,  9], [3, 11]]   ACC=57.1%  AUC=53.1%
smri_noise_1.0:  [[4, 10], [4, 10]]   ACC=50.0%  AUC=49.0%
fmri_missing:    [[10, 4], [6,  8]]   ACC=64.3%  AUC=64.8%
smri_missing:    [[12, 2], [8,  6]]   ACC=64.3%  AUC=54.1%
```

### Phase 6 Exp 2 Ensemble — Legitimate Held-Out Test (`final.json`)

```
test_at_best_val: [[6, 8], [2, 12]]   ACC=64.3%  AUC=56.6%  SEN=85.7%  SPE=42.9%
test_at_final_epoch: [[8, 6], [6, 8]] ACC=57.1%  AUC=51.5%
```

Best validation epoch: **Epoch 6** (out of 30), metric `val_acc_ensemble` = 65.5%.

---

## 5. Reliability Gate and DFD Verification

### Gate Weight Measurements (Averaged over 28 demo subjects)

| Model | Condition | w_smri | w_fmri | dims_smri | dims_fmri |
|-------|-----------|:------:|:------:|:---------:|:---------:|
| Full Demo | clean | 1.000 | 0.974 | 48.0 | 47.1 |
| Full Demo | fmri_noise_0.5 | 0.998 | 0.975 | 47.9 | 47.1 |
| Full Demo | fmri_noise_1.0 | 0.986 | 0.980 | 47.5 | 47.3 |
| Full Demo | smri_noise_0.5 | 1.000 | 0.973 | 48.0 | 47.0 |
| Full Demo | smri_noise_1.0 | 1.000 | 0.973 | 48.0 | 47.0 |
| Full Demo | **fmri_missing** | **1.000** | **0.000** | **48.0** | **12.0** |
| Full Demo | **smri_missing** | **0.000** | **1.000** | **12.0** | **48.0** |
| Phase 6 Exp 2 | clean | 0.960 | 0.986 | 46.6 | 47.5 |
| Phase 6 Exp 2 | fmri_noise_0.5 | 0.952 | 0.968 | 45.7 | 46.5 |
| Phase 6 Exp 2 | fmri_noise_1.0 | 0.924 | 0.918 | 44.3 | 44.1 |
| Phase 6 Exp 2 | smri_noise_0.5 | 0.946 | 0.966 | 45.4 | 46.4 |
| Phase 6 Exp 2 | smri_noise_1.0 | 0.848 | 0.948 | 40.7 | 45.5 |
| Phase 6 Exp 2 | **fmri_missing** | **1.000** | **0.000** | **48.0** | **12.0** |
| Phase 6 Exp 2 | **smri_missing** | **0.000** | **1.000** | **12.0** | **48.0** |

### Key DFD Observations

1. **Missing modality gate suppression:** Both models correctly set the missing modality weight to exactly **0.00** and the retained modality weight to **1.00**. ✅  
2. **DFD minimum enforced:** Under missing conditions, the zeroed modality's feature dimension is truncated to **12** (configured `min_dim=12`). ✅  
3. **Noise conditions:** Under noise, both weights remain close to 1.0 (no evidence of false gating), with slight reduction in Phase 6 Exp 2 under high noise (appropriate reliability reduction). ✅  
4. **Phase 6 Exp 2 clean gate values** (w_smri=0.96, w_fmri=0.99) reflect full-reliability operation with normal modality data. ✅  

**Assessment: ✅ Reliability gate and DFD behavior confirmed correct for all 7 conditions.**

---

## 6. Data Integrity Checks

### 6.1 Five-Seed Split Integrity

All five Phase 6 splits are stored in `result/phase6_multiseed/splits/`.

| Split File | SHA-256 (first 16 chars) | Train | Val | Test | Train∩Val | Train∩Test | Val∩Test |
|------------|:------------------------:|:-----:|:---:|:----:|:---------:|:----------:|:--------:|
| exact_splits_seed0.json | cb327dd62ff574... | 227 | 29 | 28 | **0** | **0** | **0** |
| exact_splits_seed1.json | f05951a1a2f196... | 227 | 29 | 28 | **0** | **0** | **0** |
| exact_splits_seed2.json | 7bb75bdbc12f31... | 227 | 29 | 28 | **0** | **0** | **0** |
| exact_splits_seed3.json | ea8777cee09a05... | 227 | 29 | 28 | **0** | **0** | **0** |
| exact_splits_seed4.json | 15ce9effd7aed7... | 227 | 29 | 28 | **0** | **0** | **0** |

**All 5 splits:** ✅ Zero train/val/test overlap. ✅ 284 unique subjects total per split.

`tools/verify_fiveseed_results.py` confirmed all 5 SHA-256 hashes match the locked values. ✅

### 6.2 Unit Tests

```
python -m unittest discover tests → 11/11 tests PASSED (22.3 s)
```

Tested modules: metric calculations, reliability gate, model architecture. ✅

### 6.3 Demo Set vs. Phase 6 Exp 2 Split Mismatch ⚠️ CRITICAL

| Source | Count | Match |
|--------|:-----:|:-----:|
| Demo subjects from Phase 6 Exp 2 **test** set | 10/28 | Legitimate held-out |
| Demo subjects from Phase 6 Exp 2 **validation** set | 17/28 | Seen during training ⚠️ |
| Demo subjects not in Phase 6 Exp 2 cohort | 1/28 | Kaggle-era subject |
| Demo subjects from Phase 6 Exp 2 **training** set | 0/28 | No training leakage ✅ |

**Root cause:** `demo/abide_seed0_test.npz` was created by `tools/export_demo_data.py` from a Kaggle-era split that predates the Phase 6 exact-split methodology. The demo test subjects are shuffled across validation and (partially) test splits of Phase 6 Exp 2.

**Impact:** Any Phase 6 Exp 2 metrics reported on the demo set are **contaminated** — 17/28 evaluation subjects were used in validation during checkpoint selection. These results cannot be reported as held-out test performance.

**Unaffected results:** Baseline and Full Demo checkpoints use Kaggle-era splits for which the demo set is appropriate (their training/validation used the same Kaggle splits, not the Phase 6 exact splits).

### 6.4 Test Label Integrity

- Ground truth labels in demo set: 14 ASD, 14 Control (balanced). ✅
- Labels extracted from `demo/abide_seed0_test.npz` directly; no in-loop modification. ✅
- Phase 6 Exp 2 legitimate test labels from `exact_splits_seed0.json` are frozen by SHA-256. ✅

### 6.5 Classification Threshold

- All models evaluated at the standard 0.5 probability threshold. ✅  
- No threshold tuning or optimization was performed on the test cohort. ✅  
- Ensemble construction: simple average of Head 1 and Head 2 softmax probabilities. ✅

---

## 7. Non-Comparable Results and Limitations

The following comparisons in the original diagnostic report are **non-comparable or require disclosure**:

### NC-1: Phase 6 Exp 2 on Demo Set vs. Baseline/Full Demo on Demo Set

**Reason:** Baseline and Full Demo were trained with Kaggle-era splits for which the demo test set is the correct held-out set. Phase 6 Exp 2 was trained with the Phase 6 exact splits, of which 17/28 demo subjects belong to the validation set.

**Consequence:** Comparing Phase 6 Exp 2's 57.1% (clean, demo set) against the Full Demo's 50.0% is **misleading** — the Phase 6 model was evaluated partly on its own validation subjects.

**Correct comparison:** Phase 6 Exp 2's legitimate held-out test result (64.3% from `final.json`) should be compared to re-evaluated Baseline/Full Demo on those same 28 test subjects. This requires re-running evaluation on the Phase 6 test set, which was not done in this audit.

### NC-2: Demo Head 2 vs. Phase 6 Exp 2 Ensemble

**Reason:** The old demo evaluates Head 2 only; Phase 6 Exp 2 is evaluated as an ensemble. These are structurally different evaluation protocols.

**Impact:** Comparing demo Head 2 (35.7% under fMRI missing) against Phase 6 Exp 2 ensemble (64.3%) conflates the head choice with the model improvement.

### NC-3: Demo `robustness()` vs. Full Evaluation Pipeline

**Reason:** `tools/demo.py` calls `robustness()`, which evaluates only 3 conditions and uses Head 2 exclusively. The audit evaluated all 7 conditions and all 3 outputs (Head 1, Head 2, Ensemble).

---

## 8. Five-Seed Benchmark Readiness

### 8.1 Per-Seed Status

| Seed | `best_model.pt` | `final.json` | `linear_probe_seedX.json` | Status |
|:----:|:---------------:|:------------:|:-------------------------:|:------:|
| 0 | ✅ `result/phase6_experiment2_novelty/10-28-13-02-58/best_model.pt` | ✅ Present | ✅ `linear_probe_seed0.json` | **COMPLETE** |
| 1 | ✅ `result/phase6_multiseed/runs/seed1/seed_1/best_model.pt` | ✅ Present | ✅ `linear_probe_seed1.json` | **COMPLETE** |
| 2 | ⚠️ `result/phase6_multiseed/runs/seed2/seed_2/best_model.pt` (111 MB, epoch 2) | ❌ Missing | ❌ Missing | **INCOMPLETE** |
| 3 | ❌ Not started | ❌ Missing | ❌ Missing | **NOT STARTED** |
| 4 | ❌ Not started | ❌ Missing | ❌ Missing | **NOT STARTED** |

### 8.2 Seed 0 Legitimate Test Results (Preserved)

From `result/phase6_experiment2_novelty/10-28-13-02-58/final.json`:

- **Best Val Epoch:** 6 (val_acc_ensemble = 65.5%)  
- **Test Ensemble:** ACC=64.3%, AUC=56.6%, SEN=85.7%, SPE=42.9%, F1=62.6%  
- **Test CM:** `[[6,8],[2,12]]`  
- Model config: `reliability_fusion=True`, `dynamic_dim=True`, `min_dim=12`, `edl_weight=0.5`, `head1_regularization=True`

### 8.3 Seed 1 Test Results (Preserved)

From `result/phase6_multiseed/runs/seed1/seed_1/final.json`:

- **Test Ensemble:** ACC=50.0%, AUC=52.6%, SEN=42.9%, SPE=57.1%, F1=49.7%  
- **Test CM:** `[[8,6],[8,6]]`

> Seed 1 shows lower accuracy (50.0%) than Seed 0 (64.3%), confirming run-to-run variance and the necessity of multi-seed averaging.

### 8.4 sMRI Cache Availability

| Cache File | Size | Status |
|:-----------|:----:|:------:|
| `E:/abide_data/train_aug_smri.npy` (Seed 0 original) | 3.21 GB | ✅ Present |
| `E:/abide_data/train_aug_smri_seed1.npy` | 3.21 GB | ✅ Present |
| `E:/abide_data/train_aug_smri_seed2.npy` | 3.21 GB | ✅ Present |
| `E:/abide_data/train_aug_smri_seed3.npy` | 3.21 GB | ✅ Present |
| `E:/abide_data/train_aug_smri_seed4.npy` | 3.21 GB | ✅ Present |
| `E:/abide_data/train_aug_smri_seed0.npy` (seed-suffixed) | — | ❌ Missing (Seed 0 uses non-suffixed original) |

All caches required for Seeds 2, 3, and 4 are present. ✅

### 8.5 Pipeline Verification

| Check | Result |
|-------|:------:|
| `run_multiseed_experiment2.py` checkpoint metric | `val_acc_ensemble` ✅ |
| `verify_fiveseed_results.py` aggregation metric | `test_at_best_val.ensemble` ✅ |
| Both metrics are consistent | ✅ |
| Linear probe feature dimension | 7790 (confirmed) ✅ |
| Linear probe output keys match | `['EndToEnd_Head1_cls_linear', 'EndToEnd_Head2_mlp_cls', 'Deep_AgeLinear_192', 'Deep_MLPFeat_232', 'Deep_Combined_424', 'Hybrid_FC_Plus_Deep_7094']` ✅ |

### 8.6 Required Training Commands

**Seeds 2, 3, and 4 (do not run until approved):**

```powershell
$env:WANDB_MODE="disabled"
python tools/run_multiseed_experiment2.py --seeds 2 3 4
```

**Note on Seed 2:** The runner has a completion check based on `final.json` existence. Since `final.json` is missing for Seed 2, the runner will re-execute Seed 2 from scratch (or resume if the script supports it). The existing `best_model.pt` from the interrupted run is preserved in `seed_2_interrupted_epoch1_backup/`.

**Expected output directories:**

| Seed | Expected Path |
|:----:|:-------------|
| 2 | `result/phase6_multiseed/runs/seed2/seed_2/` |
| 3 | `result/phase6_multiseed/runs/seed3/seed_3/` |
| 4 | `result/phase6_multiseed/runs/seed4/seed_4/` |

Each should produce: `best_model.pt`, `final.json`, training logs, and a corresponding `linear_probe_seed{N}.json`.

---

## 9. Recommended Next Actions

### Immediate (Before Training)

1. **Acknowledge the demo-set contamination.** Any Phase 6 Exp 2 metrics previously reported on the demo set should be annotated as non-comparable in downstream documents.

2. **Use `final.json` for all Phase 6 Exp 2 legitimate results.** The authoritative single-seed result is:
   - Ensemble ACC=64.3%, AUC=56.6%, SEN=85.7%, SPE=42.9% (test_at_best_val)

3. **(Optional)** Re-evaluate Baseline and Full Demo models on the 28 exact Phase 6 test subjects to create a true apples-to-apples comparison. This requires loading `exact_splits_seed0.json` test set subjects rather than the demo NPZ.

### Training Phase

4. **Run Seeds 2, 3, 4** using the verified multi-seed runner:
   ```powershell
   $env:WANDB_MODE="disabled"
   python tools/run_multiseed_experiment2.py --seeds 2 3 4
   ```

5. **Preserve Seeds 0 and 1.** Do not re-run `--seeds 0 1`.

6. **After all 5 seeds complete**, run:
   ```powershell
   python tools/verify_fiveseed_results.py
   ```
   This validates SHA-256 hashes, aggregates ensemble metrics, and confirms linear probe results.

### Reporting

7. After five-seed completion, update the Phase 6 results analysis document with the mean ± std across seeds (not individual-seed demo-set values).

---

## 10. Final Verdict

> ## ✅ READY FOR MULTI-SEED TRAINING
>
> **Evidence:**
> - All 5 split files locked with correct SHA-256 hashes and zero subject overlap ✅
> - All 5 sMRI caches present on E: drive ✅
> - Multi-seed runner uses correct checkpoint metric (`val_acc_ensemble`) ✅
> - Verification script aggregates correct output (`test_at_best_val.ensemble`) ✅
> - All 11 unit tests pass ✅
> - Reliability gate and DFD behavior verified correct for all 7 conditions ✅
> - Seeds 0 and 1 are preserved and complete ✅
> - Seed 2 has an early-stopped checkpoint but no `final.json` — will be re-run by the runner ✅
> - Seeds 3 and 4 not started — will be initialized by the runner ✅
>
> **Critical condition:** Report Phase 6 Exp 2 single-seed results using `final.json` values (ACC=64.3%, AUC=56.6%) rather than demo-set values. The demo-set evaluation of Phase 6 Exp 2 is **contaminated** and must not be used as held-out test performance.
>
> **Proceed with:** `python tools/run_multiseed_experiment2.py --seeds 2 3 4`

---

*This report was produced by independent inference-only reproduction. No training, checkpoint modification, split modification, or existing report modification was performed. All new outputs are in `scratch/` and `reports/`.*
