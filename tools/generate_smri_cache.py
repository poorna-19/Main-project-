"""
Pre-generates isolated 4x sMRI augmentation caches (train_aug_smri_seed{s}.npy)
for multi-seed runs.
Guarantees:
1. Strict isolation per seed (uses exact split train indices).
2. Disk-backed memmap on external drive (E:/abide_data) to fit within RAM constraints.
3. Fast execution (~1 minute per seed).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.abide import load_abide_data
from dataset.data_argument import random_rotate_3d, random_erase_3d

def generate_cache_for_seed(seed, dataset_path="E:/abide_data/abide_284.npy", out_cache=None):
    if out_cache is None:
        out_cache = f"E:/abide_data/train_aug_smri_seed{seed}.npy"
        
    split_path = f"result/phase6_multiseed/splits/exact_splits_seed{seed}.json"
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing split file: {split_path}")
        
    with open(split_path, "r") as f:
        splits = json.load(f)
    train_indices = np.array(splits["train"]["indices"])
    length = len(train_indices)
    
    cfg = OmegaConf.create({
        "dataset": {
            "path": dataset_path,
            "node_sz": 116,
            "node_feature_sz": 116,
            "timeseries_sz": 115
        }
    })
    
    print(f"\n[Seed {seed}] Loading ABIDE dataset from {dataset_path}...")
    _, _, _, smri = load_abide_data(cfg)
    smri_train = smri[train_indices]
    
    expected_shape = (4 * length,) + tuple(smri_train.shape[1:])
    expected_bytes = int(np.prod(expected_shape)) * 4
    
    if os.path.exists(out_cache):
        cur_bytes = os.path.getsize(out_cache)
        if cur_bytes in (expected_bytes, expected_bytes + 128):
            print(f"[Seed {seed}] Cache already exists and is valid ({cur_bytes} bytes): {out_cache}")
            return
        else:
            print(f"[Seed {seed}] Existing cache has invalid size ({cur_bytes} vs expected {expected_bytes}). Regenerating...")
            os.remove(out_cache)
            
    print(f"[Seed {seed}] Creating 4x sMRI memmap cache at: {out_cache} ({expected_bytes / (1024**3):.2f} GB)...")
    os.makedirs(os.path.dirname(out_cache), exist_ok=True)
    out = np.lib.format.open_memmap(out_cache, mode='w+', dtype=np.float32, shape=expected_shape)
    
    aug_times = 3
    for j in range(aug_times):
        print(f"  Generating augmentation pass {j+1}/{aug_times}...")
        for i in range(length):
            smri1 = smri_train[i]
            if j == 0:
                out[length + i] = random_rotate_3d(smri1).numpy()
            else:
                out[(j + 1) * length + i] = random_erase_3d(smri1.clone()).numpy()
                
    out[:length] = smri_train.numpy()
    out.flush()
    print(f"[Seed {seed}] Cache successfully generated and flushed to disk: {out_cache}")

def main():
    parser = argparse.ArgumentParser(description="Pre-generate sMRI augmentation caches for multi-seed runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4], help="Seeds to generate caches for")
    args = parser.parse_args()
    
    print("Pre-generating sMRI caches for seeds:", args.seeds)
    for s in args.seeds:
        generate_cache_for_seed(s)
    print("\nAll requested caches ready!")

if __name__ == "__main__":
    main()
