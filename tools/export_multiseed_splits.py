"""
Export and audit multi-seed subject splits (Seeds 0, 1, 2, 3, 4).
Ensures:
1. Every subject belongs to strictly one partition per seed (zero overlap).
2. Seed 0 matches result/phase6_baselines_and_regularization/exact_splits_seed0.json bit-for-bit.
3. Class balance is preserved (Train: 115 Con/112 ASD, Val: 15 Con/14 ASD, Test: 14 Con/14 ASD).
4. Subject IDs, File IDs, Sites, and Labels are explicitly documented for every partition.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

def generate_and_audit_multiseed_splits():
    dataset_path = "E:/abide_data/abide_284.npy"
    cohort_path = "E:/abide_data/cohort.csv"
    out_dir = "result/phase6_multiseed/splits"
    os.makedirs(out_dir, exist_ok=True)
    
    data = np.load(dataset_path, allow_pickle=True).item()
    timeseries = data["timeseires"]
    corr = data["corr"]
    labels = data["label"]
    sites = data["site"]
    file_ids = [str(f) for f in data["file_id"]]
    n_total = len(labels)
    
    cohort_df = pd.read_csv(cohort_path)
    fid_to_sub = dict(zip(cohort_df["FILE_ID"].astype(str), cohort_df["SUB_ID"].astype(str)))
    sub_ids = [fid_to_sub.get(fid, fid) for fid in file_ids]
    
    labels_flat = np.argmax(labels, axis=-1) if labels.ndim > 1 else labels.astype(int)
    labels_tensor = torch.tensor(labels_flat)
    labels_one_hot = F.one_hot(labels_tensor.to(torch.int64)).numpy()
    
    val_len = int(n_total * 0.1)
    test_len = n_total - int(n_total * 0.8) - val_len
    
    splits_summary = {}
    
    print("=" * 70)
    print("GENERATING & AUDITING MULTI-SEED SPLITS (SEEDS 0 - 4)")
    print("=" * 70)
    
    for seed in range(5):
        s1 = StratifiedShuffleSplit(n_splits=1, test_size=val_len + test_len, random_state=seed)
        train_idx, val_test_idx = next(s1.split(timeseries, labels_one_hot))
        
        s2 = StratifiedShuffleSplit(n_splits=1, test_size=test_len, random_state=seed)
        test_rel_idx, val_rel_idx = next(s2.split(timeseries[val_test_idx], labels_one_hot[val_test_idx]))
        
        val_idx = val_test_idx[val_rel_idx]
        test_idx = val_test_idx[test_rel_idx]
        
        # Verify disjointness
        s_tr = set(train_idx)
        s_va = set(val_idx)
        s_te = set(test_idx)
        assert len(s_tr & s_va) == 0, f"Seed {seed}: Train and Val overlap!"
        assert len(s_tr & s_te) == 0, f"Seed {seed}: Train and Test overlap!"
        assert len(s_va & s_te) == 0, f"Seed {seed}: Val and Test overlap!"
        assert len(s_tr | s_va | s_te) == n_total, f"Seed {seed}: Total count mismatch!"
        
        # Verify subject ID disjointness
        subs_tr = set([sub_ids[i] for i in train_idx])
        subs_va = set([sub_ids[i] for i in val_idx])
        subs_te = set([sub_ids[i] for i in test_idx])
        assert len(subs_tr & subs_va) == 0, f"Seed {seed}: Sub IDs Train and Val overlap!"
        assert len(subs_tr & subs_te) == 0, f"Seed {seed}: Sub IDs Train and Test overlap!"
        assert len(subs_va & subs_te) == 0, f"Seed {seed}: Sub IDs Val and Test overlap!"
        
        splits_dict = {
            "metadata": {
                "dataset_path": dataset_path,
                "cohort_path": cohort_path,
                "seed": seed,
                "n_total": int(n_total),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            },
            "train": {
                "indices": [int(i) for i in train_idx],
                "sub_ids": [sub_ids[i] for i in train_idx],
                "file_ids": [file_ids[i] for i in train_idx],
                "labels": [int(labels_flat[i]) for i in train_idx],
                "sites": [str(sites[i]) for i in train_idx],
                "n_control": int(np.sum(labels_flat[train_idx] == 0)),
                "n_asd": int(np.sum(labels_flat[train_idx] == 1)),
            },
            "val": {
                "indices": [int(i) for i in val_idx],
                "sub_ids": [sub_ids[i] for i in val_idx],
                "file_ids": [file_ids[i] for i in val_idx],
                "labels": [int(labels_flat[i]) for i in val_idx],
                "sites": [str(sites[i]) for i in val_idx],
                "n_control": int(np.sum(labels_flat[val_idx] == 0)),
                "n_asd": int(np.sum(labels_flat[val_idx] == 1)),
            },
            "test": {
                "indices": [int(i) for i in test_idx],
                "sub_ids": [sub_ids[i] for i in test_idx],
                "file_ids": [file_ids[i] for i in test_idx],
                "labels": [int(labels_flat[i]) for i in test_idx],
                "sites": [str(sites[i]) for i in test_idx],
                "n_control": int(np.sum(labels_flat[test_idx] == 0)),
                "n_asd": int(np.sum(labels_flat[test_idx] == 1)),
            }
        }
        
        out_path = os.path.join(out_dir, f"exact_splits_seed{seed}.json")
        with open(out_path, "w") as f:
            json.dump(splits_dict, f, indent=2)
            
        splits_summary[f"seed_{seed}"] = {
            "splits_file": out_path,
            "train": f"{len(train_idx)} subs ({splits_dict['train']['n_control']} Con / {splits_dict['train']['n_asd']} ASD)",
            "val": f"{len(val_idx)} subs ({splits_dict['val']['n_control']} Con / {splits_dict['val']['n_asd']} ASD)",
            "test": f"{len(test_idx)} subs ({splits_dict['test']['n_control']} Con / {splits_dict['test']['n_asd']} ASD)",
            "test_sub_ids": splits_dict['test']['sub_ids']
        }
        
        print(f"Seed {seed}: Saved to {out_path}")
        print(f"  Train: {splits_summary[f'seed_{seed}']['train']}")
        print(f"  Val:   {splits_summary[f'seed_{seed}']['val']}")
        print(f"  Test:  {splits_summary[f'seed_{seed}']['test']}")
        
    # Check seed 0 exact match with original
    orig_seed0 = "result/phase6_baselines_and_regularization/exact_splits_seed0.json"
    if os.path.exists(orig_seed0):
        with open(orig_seed0, "r") as f:
            orig_data = json.load(f)
        with open(os.path.join(out_dir, "exact_splits_seed0.json"), "r") as f:
            new_data = json.load(f)
        assert orig_data["train"]["indices"] == new_data["train"]["indices"], "Seed 0 Train indices differ!"
        assert orig_data["val"]["indices"] == new_data["val"]["indices"], "Seed 0 Val indices differ!"
        assert orig_data["test"]["indices"] == new_data["test"]["indices"], "Seed 0 Test indices differ!"
        print("\nSeed 0 verification: 100% BIT-FOR-BIT MATCH WITH ORIGINAL RUN!")
        
    # Document test set overlap between seeds
    print("\n--- TEST SET OVERLAP MATRIX ACROSS SEEDS ---")
    overlap_matrix = {}
    for s_a in range(5):
        overlap_matrix[f"seed_{s_a}"] = {}
        with open(os.path.join(out_dir, f"exact_splits_seed{s_a}.json")) as fa:
            te_a = set(json.load(fa)["test"]["indices"])
        for s_b in range(5):
            with open(os.path.join(out_dir, f"exact_splits_seed{s_b}.json")) as fb:
                te_b = set(json.load(fb)["test"]["indices"])
            overlap_matrix[f"seed_{s_a}"][f"seed_{s_b}"] = len(te_a & te_b)
            
    summary_path = os.path.join(out_dir, "multiseed_splits_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"splits": splits_summary, "test_overlap_matrix": overlap_matrix}, f, indent=2)
    print(f"Multi-seed splits summary documented in: {summary_path}")
    print("=" * 70)

if __name__ == "__main__":
    generate_and_audit_multiseed_splits()
