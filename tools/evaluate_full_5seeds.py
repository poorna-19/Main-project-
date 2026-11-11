"""
Evaluate the 5 existing Full Gated model checkpoints (27.8M params)
on their exact respective held-out test splits (N=28).
"""
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score, precision_score, recall_score
)
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import dataset_factory
from models import model_factory

FULL_CHECKPOINTS = {
    0: ROOT / "demo" / "full_seed0.pt",
    1: ROOT / "result" / "phase6_multiseed" / "runs" / "seed1" / "seed_1" / "best_model.pt",
    2: ROOT / "result" / "phase6_multiseed" / "runs" / "seed2" / "seed_2" / "best_model.pt",
    3: ROOT / "result" / "phase6_multiseed" / "runs" / "seed3" / "seed_3" / "best_model.pt",
    4: ROOT / "result" / "phase6_multiseed" / "runs" / "seed4" / "seed_4" / "best_model.pt",
}

def evaluate_full_models():
    results = {}
    print("=" * 80)
    print("Evaluating 5-Seed Full Gated Checkpoints (27.8M parameters)")
    print("=" * 80)

    for seed in range(5):
        ckpt_path = FULL_CHECKPOINTS[seed]
        split_path = ROOT / "result" / "phase6_multiseed" / "splits" / f"exact_splits_seed{seed}.json"
        
        print(f"\n--- Evaluating Seed {seed} ---")
        print(f"Checkpoint: {ckpt_path.relative_to(ROOT)}")
        print(f"Split: {split_path.relative_to(ROOT)}")

        conf = OmegaConf.load(ROOT / "conf" / "config.yaml")
        conf.model = OmegaConf.load(ROOT / "conf" / "model" / "cdcomambanet_novelty.yaml")
        conf.dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "ABIDE.yaml")
        conf.training = OmegaConf.load(ROOT / "conf" / "training" / "exp2_novelty_training.yaml")
        conf.optimizer = OmegaConf.load(ROOT / "conf" / "optimizer" / "adamw_wd05.yaml")
        conf.datasz = OmegaConf.load(ROOT / "conf" / "datasz" / "100p.yaml")
        conf.preprocess = OmegaConf.load(ROOT / "conf" / "preprocess" / "non_mixup.yaml")

        with open_dict(conf):
            conf.seed = seed
            conf.repeat_time = 1
            conf.log_path = "result/audit_full"
            conf.unique_id = f"seed_{seed}"
            conf.dataset.path = "E:/abide_data/abide_284.npy"
            conf.dataset.exact_splits_path = str(split_path)
            conf.dataset.batch_size = 2
            conf.dataset.val_batch_size = 2
            conf.dataset.test_batch_size = 2
            conf.dataset.expand = False  # No training augmentation needed for eval

        dataloaders = dataset_factory(conf)
        val_loader, test_loader = dataloaders[1], dataloaders[2]

        model = model_factory(conf)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        model.cuda()

        # Step 1: Validation Logits for Threshold Calibration
        val_preds_h1, val_preds_h2, val_targets = [], [], []
        with torch.no_grad():
            for fmri, corr, label, smri in val_loader:
                fmri, corr, smri = fmri.cuda(), corr.cuda(), smri.cuda()
                p1, p2 = model(fmri, corr, smri)
                val_preds_h1.extend(p1.cpu().numpy())
                val_preds_h2.extend(p2.cpu().numpy())
                y = torch.argmax(label, dim=-1).cpu().numpy() if label.dim() > 1 else label.cpu().numpy()
                val_targets.extend(y)

        val_probs_h2 = F.softmax(torch.tensor(np.array(val_preds_h2)), dim=-1)[:, 1].numpy()
        val_targets = np.array(val_targets)

        # Optimal Youden threshold on validation only
        best_j, best_thresh = -1.0, 0.50
        for th in np.linspace(0.10, 0.90, 801):
            pred_bin = (val_probs_h2 >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(val_targets, pred_bin, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sens + spec - 1.0
            if j > best_j:
                best_j = j
                best_thresh = th

        print(f"Validation Calibrated Threshold: theta* = {best_thresh:.4f} (Max J = {best_j:.4f})")

        # Step 2: Test Evaluation
        test_preds_h1, test_preds_h2, test_targets = [], [], []
        with torch.no_grad():
            for fmri, corr, label, smri in test_loader:
                fmri, corr, smri = fmri.cuda(), corr.cuda(), smri.cuda()
                p1, p2 = model(fmri, corr, smri)
                test_preds_h1.extend(p1.cpu().numpy())
                test_preds_h2.extend(p2.cpu().numpy())
                y = torch.argmax(label, dim=-1).cpu().numpy() if label.dim() > 1 else label.cpu().numpy()
                test_targets.extend(y)

        test_probs_h2 = F.softmax(torch.tensor(np.array(test_preds_h2)), dim=-1)[:, 1].numpy()
        test_targets = np.array(test_targets)

        # Default 0.50
        test_pred_def = (test_probs_h2 >= 0.50).astype(int)
        cm_def = confusion_matrix(test_targets, test_pred_def, labels=[0, 1])
        acc_def = accuracy_score(test_targets, test_pred_def)
        bacc_def = balanced_accuracy_score(test_targets, test_pred_def)
        f1_def = f1_score(test_targets, test_pred_def, average="macro", zero_division=0)
        sen_def = recall_score(test_targets, test_pred_def, pos_label=1, zero_division=0)
        spe_def = recall_score(test_targets, test_pred_def, pos_label=0, zero_division=0)
        prec_def = precision_score(test_targets, test_pred_def, pos_label=1, zero_division=0)
        auc = roc_auc_score(test_targets, test_probs_h2)

        # Calibrated
        test_pred_cal = (test_probs_h2 >= best_thresh).astype(int)
        cm_cal = confusion_matrix(test_targets, test_pred_cal, labels=[0, 1])
        acc_cal = accuracy_score(test_targets, test_pred_cal)
        bacc_cal = balanced_accuracy_score(test_targets, test_pred_cal)
        f1_cal = f1_score(test_targets, test_pred_cal, average="macro", zero_division=0)
        sen_cal = recall_score(test_targets, test_pred_cal, pos_label=1, zero_division=0)
        spe_cal = recall_score(test_targets, test_pred_cal, pos_label=0, zero_division=0)
        prec_cal = precision_score(test_targets, test_pred_cal, pos_label=1, zero_division=0)

        print(f"Test Default:    Acc={acc_def*100:.2f}%, BAcc={bacc_def*100:.2f}%, F1={f1_def*100:.2f}%, AUC={auc*100:.2f}%")
        print(f"Test Calibrated: Acc={acc_cal*100:.2f}%, BAcc={bacc_cal*100:.2f}%, F1={f1_cal*100:.2f}%, CM={cm_cal.tolist()}")

        results[f"seed_{seed}"] = {
            "val_calibrated_threshold": float(best_thresh),
            "test_metrics_default": {
                "accuracy": float(acc_def),
                "balanced_accuracy": float(bacc_def),
                "sensitivity": float(sen_def),
                "specificity": float(spe_def),
                "precision": float(prec_def),
                "macro_f1": float(f1_def),
                "auc": float(auc),
                "confusion_matrix": cm_def.tolist()
            },
            "test_metrics_calibrated": {
                "accuracy": float(acc_cal),
                "balanced_accuracy": float(bacc_cal),
                "sensitivity": float(sen_cal),
                "specificity": float(spe_cal),
                "precision": float(prec_cal),
                "macro_f1": float(f1_cal),
                "auc": float(auc),
                "confusion_matrix": cm_cal.tolist()
            }
        }

    out_file = ROOT / "result" / "full_model_5seeds_test_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full model 5-seed evaluation to {out_file}")

if __name__ == "__main__":
    evaluate_full_models()
