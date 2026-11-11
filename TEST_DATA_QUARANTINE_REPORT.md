# CDCo-MambaNet V4 — Held-Out Test Set Quarantine Report

**Author:** Senior Deep Learning Research Engineer  
**Date:** November 11, 2026  
**Subject:** Scientific Protocol for Held-Out Test Data Isolation & Quarantine  

---

## 1. Background & Identification of Risk

The file `demo/abide_seed0_test.npz` ($45,590,895$ bytes) contains multimodal neuroimaging data (fMRI timeseries, Pearson correlation matrices, and 3D sMRI volumes) for 28 real human subjects from the ABIDE I cohort.

In prior setup instructions, researchers were advised to execute `python tools/demo.py` as an immediate post-installation smoke test.

### Scientific Violation Identified:
In translational clinical AI and medical imaging research, **the held-out test cohort must be treated as a strict scientific quarantine zone.**
1. Repeatedly executing inference against the held-out test set during routine environment configuration, debugging, or portability testing introduces **subconscious observer bias** and risks **test-set wear**.
2. If hyperparameter adjustments or configuration fixes are informed by performance observed on `demo/abide_seed0_test.npz`, the generalization claims of the research are invalidated.
3. Furthermore, diagnostic forensic analysis revealed that `demo/abide_seed0_test.npz` was exported from a Kaggle-era split that predated the Phase 6 frozen split methodology (`splits/exact_splits_seed0.json`).

---

## 2. Inventory of Scripts Accessing Test Patient Data

| Script Path | Purpose | Access Mode | Quarantine Action Taken |
| :--- | :--- | :---: | :--- |
| `tools/demo.py` | Legacy Kaggle reproduction demo | Real Patients | **DEPRECATED for setup verification.** Retained strictly as historical Kaggle artifact. |
| `tools/evaluate_v2_audit.py` | Historical audit tool | Real Patients | Archived in research reports. |
| `tools/export_demo_data.py` | Demo export utility | Real Patients | Documented as legacy utility. |
| `tools/generate_audit_artifacts.py`| Visualization generator | Real Patients | Retained for historical audit figures only. |
| `tools/safe_fresh_system_check.py` | **NEW Primary Setup Tool** | **100% Synthetic** | **APPROVED.** Completely eliminates access to real test data. |

---

## 3. Strict Frozen Research Protocol for Held-Out Test Data

To guarantee pristine scientific integrity for Phase 2 and all future research:

1. **Strictly Single-Pass Evaluation:**
   The 28 held-out test participants in `splits/exact_splits_seed0.json` shall be evaluated **exactly once** per experiment, strictly after all training epochs are complete, all early-stopping decisions are final, and the checkpoint has been frozen based solely on minimum validation loss on the 29 validation subjects.
2. **Zero In-Loop Test Evaluation:**
   Neither the training loop (`training/training.py`), validation hooks, nor the early stopping monitor may compute test loss or test metrics during training.
3. **No Threshold or Hyperparameter Tuning on Test Data:**
   Post-hoc calibration (such as temperature scaling and evidential uncertainty thresholding) must be fitted exclusively on the frozen validation cohort.
4. **Routine Portability & Smoke Testing:**
   All environment verifications, hardware checks, CI/CD pipelines, and fresh-computer setup testing must invoke `tools/safe_fresh_system_check.py`, which generates random synthetic tensors in memory and discards them immediately.

---

## 4. Preservation of Historical Research Evidence

In accordance with forensic audit protocols, `demo/abide_seed0_test.npz` and historical evaluation logs are **not deleted or altered**, preserving the reproducibility chain for past publications. However, they are formally quarantined and removed from the active onboarding and execution paths.
