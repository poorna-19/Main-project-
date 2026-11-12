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

def generate_cache_for_seed(seed, dataset_path="E:/abide_data/abide_284.npy", out_cache=None, split_path=None):
    if out_cache is None:
        out_cache = f"E:/abide_data/train_aug_smri_seed{seed}.npy"
        
    if split_path is None:
        candidate_splits = [
            f"result/phase6_multiseed/splits/exact_splits_seed{seed}.json",
            f"splits/exact_splits_seed{seed}.json"
        ]
        for p in candidate_splits:
            if os.path.exists(p):
                split_path = p
                break
        if split_path is None:
            split_path = candidate_splits[0]

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
    parser.add_argument("--seed", type=int, default=None, help="Single seed to generate cache for (e.g. 0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to generate caches for")
    parser.add_argument("--dataset_path", type=str, default="E:/abide_data/abide_284.npy", help="Path to abide_284.npy")
    parser.add_argument("--out_cache", type=str, default=None, help="Custom output path for generated cache .npy")
    parser.add_argument("--split_path", type=str, default=None, help="Path to exact split JSON file")
    args = parser.parse_args()
    
    if args.seed is not None:
        target_seeds = [args.seed]
    elif args.seeds is not None:
        target_seeds = args.seeds
    else:
        target_seeds = [2, 3, 4]

    print("Pre-generating sMRI caches for seeds:", target_seeds)
    for s in target_seeds:
        generate_cache_for_seed(
            s,
            dataset_path=args.dataset_path,
            out_cache=args.out_cache if len(target_seeds) == 1 else None,
            split_path=args.split_path
        )
    print("\nAll requested caches ready!")

if __name__ == "__main__":
    main()
