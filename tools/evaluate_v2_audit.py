"""
Reproducible Independent Evaluation Script for CDCo-MambaNet V2 Audit
Evaluates certified model checkpoints strictly on the held-out test cohort without training.

Usage:
    python tools/evaluate_v2_audit.py
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score
)
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.CDCo_MambaNet.cdcomnet import CDCo_MambaNet


def evaluate_checkpoint(checkpoint_path, data_path):
    print("=" * 80)
    print("CDCo-MambaNet Stage A Independent Evaluation Audit")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Cohort: {data_path}")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test cohort file not found at: {data_path}")

    # 1. Load Data
    test_data = np.load(data_path, allow_pickle=True)
    ts_key = "timeseries" if "timeseries" in test_data else ("fmri" if "fmri" in test_data else "timeseires")
    fmri = torch.tensor(test_data[ts_key], dtype=torch.float32)
    corr = torch.tensor(test_data["corr"], dtype=torch.float32)
    smri = torch.tensor(test_data["smri"], dtype=torch.float32)
    labels = torch.tensor(test_data["label"], dtype=torch.long)

    n_samples = len(labels)
    n_controls = int((labels == 0).sum())
    n_asd = int((labels == 1).sum())
    print(f"Loaded {n_samples} held-out test subjects: {n_controls} Controls (0), {n_asd} ASD (1)")

    # 2. Build Model Architecture
    conf = OmegaConf.load(ROOT / "conf" / "config.yaml")
    model_conf = OmegaConf.load(ROOT / "conf" / "model" / "cdcomambanet_novelty.yaml")
    conf.model = model_conf
    conf.dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "ABIDE.yaml")

    model = CDCo_MambaNet(conf)

    # 3. Load State Dict
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Run Forward Inference (Zero Gradients)
    with torch.no_grad():
        avail = torch.ones(n_samples, 2, dtype=torch.bool)
        out_h1, out_h2 = model(fmri, corr, smri, avail)

        prob_h1 = F.softmax(out_h1, dim=-1)[:, 1].numpy()
        prob_h2 = F.softmax(out_h2, dim=-1)[:, 1].numpy()
        prob_ens = 0.5 * (prob_h1 + prob_h2)

        pred_h1 = (prob_h1 >= 0.5).astype(int)
        pred_h2 = (prob_h2 >= 0.5).astype(int)
        pred_ens = (prob_ens >= 0.5).astype(int)
        y_true = labels.numpy()

    # 5. Compute Comprehensive Metrics
    results = {}
    for name, pred, prob in [
        ("Head 1 (cls_linear)", pred_h1, prob_h1),
        ("Head 2 (mlp_cls)", pred_h2, prob_h2),
        ("Ensemble (Dual Head)", pred_ens, prob_ens),
    ]:
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, pred)
        bacc = balanced_accuracy_score(y_true, pred)
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, pred, average="macro")
        auc = roc_auc_score(y_true, prob)

        results[name] = {
            "Accuracy": acc,
            "Balanced_Accuracy": bacc,
            "ASD_Sensitivity": sen,
            "Control_Specificity": spe,
            "F1_Score": f1,
            "ROC_AUC": auc,
            "Confusion_Matrix": cm.tolist(),
        }

        print(f"\n--- {name} Results ---")
        print(f"  Accuracy:            {acc * 100:.2f}% ({tp + tn}/{n_samples} correct)")
        print(f"  Balanced Accuracy:   {bacc * 100:.2f}%")
        print(f"  ASD Sensitivity:     {sen * 100:.2f}% ({tp}/{tp + fn} ASD detected)")
        print(f"  Control Specificity: {spe * 100:.2f}% ({tn}/{tn + fp} Controls confirmed)")
        print(f"  F1 Score (Macro):    {f1 * 100:.2f}%")
        print(f"  ROC-AUC:             {auc * 100:.2f}%")
        print(f"  Confusion Matrix:    TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results


if __name__ == "__main__":
    ckpt = ROOT / "result" / "phase6_experiment2_novelty" / "10-28-13-02-58" / "best_model.pt"
    if not ckpt.exists():
        ckpt = ROOT / "demo" / "full_seed0.pt"

    cohort = ROOT / "demo" / "abide_seed0_test.npz"
    evaluate_checkpoint(str(ckpt), str(cohort))
