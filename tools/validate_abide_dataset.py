"""Lightweight validation script for the prepared ABIDE cohort on Drive E:.

Checks:
1. Existence of abide_284.npy and abide_smri96.npy.
2. File sizes, array shapes, and data types.
3. Subject counts and class balance (ASD vs Control).
4. Finite values (no NaNs, no Infs).
5. Integration with load_abide_data().
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.abide import load_abide_data

DATA_DIR = Path(r"E:\abide_data")
DATASET_PATH = DATA_DIR / "abide_284.npy"
SMRI_PATH = DATA_DIR / "abide_smri96.npy"


def main():
    print(f"=== Validating ABIDE Dataset in {DATA_DIR} ===")
    errors = []

    # 1. Existence
    if not DATASET_PATH.exists():
        print(f"[FAIL] {DATASET_PATH} does not exist!")
        return 1
    if not SMRI_PATH.exists():
        print(f"[FAIL] {SMRI_PATH} does not exist!")
        return 1

    size_dict_mb = DATASET_PATH.stat().st_size / (1024 * 1024)
    size_smri_mb = SMRI_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] Found abide_284.npy ({size_dict_mb:.2f} MB)")
    print(f"[OK] Found abide_smri96.npy ({size_smri_mb:.2f} MB)")

    # 2. Data contents
    try:
        data = np.load(DATASET_PATH, allow_pickle=True).item()
    except Exception as e:
        print(f"[FAIL] Failed to load {DATASET_PATH}: {e}")
        return 1

    required_keys = ["timeseires", "corr", "label", "site", "file_id", "smri"]
    for k in required_keys:
        if k not in data:
            errors.append(f"Missing key in dictionary: '{k}'")
        else:
            print(f"  Key '{k}': present")

    ts = data["timeseires"]
    corr = data["corr"]
    label = data["label"]
    site = data["site"]
    file_id = data["file_id"]

    # 3. Shapes & dtypes
    print("\n--- Array Shapes & Data Types ---")
    print(f"  timeseries shape: {ts.shape}, dtype: {ts.dtype}")
    print(f"  corr shape:       {corr.shape}, dtype: {corr.dtype}")
    print(f"  label shape:      {label.shape}, dtype: {label.dtype}")
    print(f"  site count:       {len(site)}, unique sites: {len(np.unique(site))}")
    print(f"  file_id count:    {len(file_id)}")

    n_subjects = len(label)
    if n_subjects != 284:
        errors.append(f"Expected 284 subjects, got {n_subjects}")

    if ts.shape != (n_subjects, 116, 115):
        errors.append(f"Unexpected timeseries shape: {ts.shape}, expected ({n_subjects}, 116, 115)")
    if corr.shape != (n_subjects, 116, 116):
        errors.append(f"Unexpected corr shape: {corr.shape}, expected ({n_subjects}, 116, 116)")

    # 4. Class Balance
    n_controls = int(np.sum(label == 0))
    n_asd = int(np.sum(label == 1))
    print(f"\n--- Class Distribution ---")
    print(f"  Controls (0): {n_controls} (expected: 144)")
    print(f"  Autism   (1): {n_asd} (expected: 140)")
    print(f"  Total:        {n_controls + n_asd}")

    if n_controls != 144:
        errors.append(f"Expected 144 controls, got {n_controls}")
    if n_asd != 140:
        errors.append(f"Expected 140 ASD subjects, got {n_asd}")

    # 5. Check Finite Values (No NaNs, No Infs)
    print("\n--- Numerical Integrity Checks ---")
    nan_ts = np.isnan(ts).sum()
    inf_ts = np.isinf(ts).sum()
    nan_corr = np.isnan(corr).sum()
    inf_corr = np.isinf(corr).sum()
    print(f"  timeseries NaNs: {nan_ts}, Infs: {inf_ts}")
    print(f"  corr NaNs:       {nan_corr}, Infs: {inf_corr}")

    if nan_ts > 0 or inf_ts > 0:
        errors.append("Timeseries contains NaNs or Infs")
    if nan_corr > 0 or inf_corr > 0:
        errors.append("Correlation matrices contain NaNs or Infs")

    # 6. Check sMRI Memmap
    print("\n--- sMRI Memmap Checks ---")
    try:
        smri_mmap = np.load(SMRI_PATH, mmap_mode="r")
        print(f"  sMRI memmap shape: {smri_mmap.shape}, dtype: {smri_mmap.dtype}")
        if smri_mmap.shape != (n_subjects, 1, 96, 96, 96):
            errors.append(f"Unexpected sMRI shape: {smri_mmap.shape}, expected ({n_subjects}, 1, 96, 96, 96)")
        
        # Sample check first and last volume for finite values
        v0 = smri_mmap[0]
        v_last = smri_mmap[-1]
        if np.isnan(v0).any() or np.isnan(v_last).any():
            errors.append("sMRI sample volume contains NaNs")
        else:
            print("  sMRI sample volume finite check: PASSED")
    except Exception as e:
        errors.append(f"Failed to open sMRI memmap: {e}")

    # 7. Check Dataloader Integration via load_abide_data()
    print("\n--- Pipeline Integration Check (load_abide_data) ---")
    try:
        cfg = OmegaConf.create({
            "dataset": {
                "name": "abide",
                "path": str(DATASET_PATH),
                "batch_size": 2,
                "node_sz": 116,
                "node_feature_sz": 116,
                "timeseries_sz": 115,
            }
        })
        f_time, f_pc, f_labels, f_smri = load_abide_data(cfg)
        print(f"  load_abide_data() returned tensors:")
        print(f"    f_time:   {f_time.shape}, dtype: {f_time.dtype}")
        print(f"    f_pc:     {f_pc.shape}, dtype: {f_pc.dtype}")
        print(f"    f_labels: {f_labels.shape}, dtype: {f_labels.dtype}")
        print(f"    f_smri:   {f_smri.shape}, dtype: {f_smri.dtype}")
        print("  Pipeline Integration: SUCCESSFUL")
    except Exception as e:
        errors.append(f"load_abide_data() failed: {e}")

    print("\n==================================")
    if errors:
        print("[FAIL] Validation failed with errors:")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("[SUCCESS] All dataset validation checks PASSED!")
        print("ABIDE cohort on Drive E: is complete, verified, and ready for training.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
