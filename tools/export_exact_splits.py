"""
Export and verify exact subject splits used across all experiments (Seed 0).
Guarantees bit-for-bit identical subjects across deep learning and classical baselines.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

def export_splits():
    dataset_path = "E:/abide_data/abide_284.npy"
    cohort_path = "E:/abide_data/cohort.csv"
    out_path = "result/phase6_baselines_and_regularization/exact_splits_seed0.json"
    
    data = np.load(dataset_path, allow_pickle=True).item()
    timeseries = data["timeseires"]
    corr = data["corr"]
    labels = data["label"]
    sites = data["site"]
    file_ids = [str(f) for f in data["file_id"]]
    n_total = len(labels)
    
    cohort_df = pd.read_csv(cohort_path)
    # Map file_id to sub_id
    fid_to_sub = dict(zip(cohort_df["FILE_ID"].astype(str), cohort_df["SUB_ID"].astype(str)))
    sub_ids = [fid_to_sub.get(fid, fid) for fid in file_ids]
    
    labels_flat = np.argmax(labels, axis=-1) if labels.ndim > 1 else labels.astype(int)
    labels_tensor = torch.tensor(labels_flat)
    labels_one_hot = F.one_hot(labels_tensor.to(torch.int64)).numpy()
    
    seed = 0
    val_len = int(n_total * 0.1)
    test_len = n_total - int(n_total * 0.8) - val_len
    
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=val_len + test_len, random_state=seed)
    train_idx, val_test_idx = next(split1.split(timeseries, labels_one_hot))
    
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_len, random_state=seed)
    test_rel_idx, val_rel_idx = next(split2.split(timeseries[val_test_idx], labels_one_hot[val_test_idx]))
    
    val_idx = val_test_idx[val_rel_idx]
    test_idx = val_test_idx[test_rel_idx]
    
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
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits_dict, f, indent=2)
        
    print(f"Successfully exported exact splits to {out_path}")
    print(f"Train subjects: {len(train_idx)} (Control: {splits_dict['train']['n_control']}, ASD: {splits_dict['train']['n_asd']})")
    print(f"Val subjects:   {len(val_idx)} (Control: {splits_dict['val']['n_control']}, ASD: {splits_dict['val']['n_asd']})")
    print(f"Test subjects:  {len(test_idx)} (Control: {splits_dict['test']['n_control']}, ASD: {splits_dict['test']['n_asd']})")

if __name__ == "__main__":
    export_splits()
