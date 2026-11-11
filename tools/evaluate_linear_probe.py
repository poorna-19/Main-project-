"""
Phase 6: Linear Probe & Representation Evaluation on Frozen CDCo-MambaNet
Extracts deep representations from best_model.pt and tests whether:
1. Deep spatio-temporal representations contain generalizable ASD signal when classified with regularized linear heads.
2. Fusing deep Mamba representations with static Pearson FC (6,670 edges) improves test accuracy beyond 71.43%.

Strict Zero-Leakage Protocol:
- Identical subject splits (Train=227, Val=29, Test=28 from exact_splits_seed0.json)
- Feature scalers fit strictly on Train
- Hyperparameters tuned strictly on Val
- Test evaluated exactly once per finalized model
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score, recall_score
)
from dataset.abide import load_abide_data
from models.CDCo_MambaNet.cdcomnet import CDCo_MambaNet
from tools.train_classical_baselines import extract_upper_tri, calc_all_metrics

def extract_deep_features(model, dataloader, device="cuda"):
    """
    Extracts intermediate representations from CDCo_MambaNet:
    - age_linear_feat: 192-dim vector before cls_linear
    - mlp_feat: 232-dim vector before mlp_cls.cls
    - end_to_end_preds: raw logits from head 1 and head 2
    """
    model.eval()
    age_feats = []
    mlp_feats = []
    logits1_list = []
    logits2_list = []
    labels_list = []
    
    # Register forward hooks to capture intermediate latent features
    captured = {}
    
    def hook_fuse(module, input, output):
        # In Mamba_fusion.forward:
        # age_out_linear = torch.cat((fmri_age, smri_age), dim=-1)
        # cls_linear is called with age_out_linear
        pass

    # We can inspect the exact forward logic cleanly by hooking cls_linear and mlp_cls.cls
    def hook_cls_linear(module, input, output):
        captured['age_linear_feat'] = input[0].detach().cpu()
        
    def hook_mlp_cls(module, input, output):
        captured['mlp_feat'] = input[0].detach().cpu()
        
    h1 = model.fuse_encoder.cls_linear.register_forward_hook(hook_cls_linear)
    h2 = model.fuse_encoder.mlp_cls.cls.register_forward_hook(hook_mlp_cls)
    
    with torch.no_grad():
        for batch in dataloader:
            fmri, corr, label, smri = batch
            fmri = fmri.to(device)
            corr = corr.to(device)
            smri = smri.to(device)
            
            p1, p2 = model(fmri, corr, smri)
            
            age_feats.append(captured['age_linear_feat'].numpy())
            mlp_feats.append(captured['mlp_feat'].numpy())
            logits1_list.append(p1.detach().cpu().numpy())
            logits2_list.append(p2.detach().cpu().numpy())
            
            if label.ndim > 1:
                labels_flat = torch.argmax(label, dim=-1).cpu().numpy()
            else:
                labels_flat = label.cpu().numpy().astype(int)
            labels_list.append(labels_flat)
            
    h1.remove()
    h2.remove()
    
    return {
        "age_feats": np.concatenate(age_feats, axis=0),
        "mlp_feats": np.concatenate(mlp_feats, axis=0),
        "logits1": np.concatenate(logits1_list, axis=0),
        "logits2": np.concatenate(logits2_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0),
    }

class SubjectDataset(torch.utils.data.Dataset):
    def __init__(self, timeseries, corr, labels, smri, indices):
        self.indices = indices
        self.timeseries = timeseries[indices]
        self.corr = corr[indices]
        self.labels = labels[indices]
        self.smri = smri[indices]
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        return (
            self.timeseries[idx],
            self.corr[idx],
            self.labels[idx],
            self.smri[idx]
        )

import argparse

def run_linear_probe():
    parser = argparse.ArgumentParser(description="Evaluate linear probes on frozen CDCo-MambaNet representations.")
    parser.add_argument("--checkpoint", type=str, default="result/controlled_run_resumed/10-26-23-09-22/best_model.pt")
    parser.add_argument("--config", type=str, default="result/controlled_run_resumed/10-26-23-09-22/config.yaml")
    parser.add_argument("--splits", type=str, default="result/phase6_baselines_and_regularization/exact_splits_seed0.json")
    parser.add_argument("--out", type=str, default="result/phase6_baselines_and_regularization/linear_probe_results.json")
    args = parser.parse_args()

    splits_path = args.splits
    checkpoint_path = args.checkpoint
    config_path = args.config
    out_file = args.out
    
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    train_idx = np.array(splits["train"]["indices"])
    val_idx = np.array(splits["val"]["indices"])
    test_idx = np.array(splits["test"]["indices"])
    
    print(f"Loaded splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    cfg = OmegaConf.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load dataset
    final_timeseires, final_pearson, labels, smri = load_abide_data(cfg)
    
    # Create non-shuffled unaugmented dataloaders for feature extraction
    batch_size = 4
    train_ds = SubjectDataset(final_timeseires, final_pearson, labels, smri, train_idx)
    val_ds = SubjectDataset(final_timeseires, final_pearson, labels, smri, val_idx)
    test_ds = SubjectDataset(final_timeseires, final_pearson, labels, smri, test_idx)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = CDCo_MambaNet(cfg).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print("Extracting deep representations...")
    train_extracted = extract_deep_features(model, train_loader, device=device)
    val_extracted = extract_deep_features(model, val_loader, device=device)
    test_extracted = extract_deep_features(model, test_loader, device=device)
    
    y_train = train_extracted["labels"]
    y_val = val_extracted["labels"]
    y_test = test_extracted["labels"]
    
    # Also extract classical Pearson FC upper triangle features
    X_fc_raw = extract_upper_tri(final_pearson.cpu().numpy())
    X_fc_train = X_fc_raw[train_idx]
    X_fc_val = X_fc_raw[val_idx]
    X_fc_test = X_fc_raw[test_idx]
    
    # Representations to evaluate
    feature_sets = {
        "Deep_AgeLinear_192": {
            "train": train_extracted["age_feats"],
            "val": val_extracted["age_feats"],
            "test": test_extracted["age_feats"],
            "desc": "96-dim multimodal fused token representation (before cls_linear; 48 fMRI + 48 sMRI)"
        },
        "Deep_MLPFeat_232": {
            "train": train_extracted["mlp_feats"],
            "val": val_extracted["mlp_feats"],
            "test": test_extracted["mlp_feats"],
            "desc": "1,024-dim nonlinear projection representation (before mlp_cls.cls; 2x512 channels)"
        },
        "Deep_Combined_424": {
            "train": np.hstack([train_extracted["age_feats"], train_extracted["mlp_feats"]]),
            "val": np.hstack([val_extracted["age_feats"], val_extracted["mlp_feats"]]),
            "test": np.hstack([test_extracted["age_feats"], test_extracted["mlp_feats"]]),
            "desc": "1,120-dim concatenated deep representations (96 age_linear + 1,024 mlp_cls)"
        },
        "Hybrid_FC_Plus_Deep_7094": {
            "train": np.hstack([X_fc_train, train_extracted["age_feats"], train_extracted["mlp_feats"]]),
            "val": np.hstack([X_fc_val, val_extracted["age_feats"], val_extracted["mlp_feats"]]),
            "test": np.hstack([X_fc_test, test_extracted["age_feats"], test_extracted["mlp_feats"]]),
            "desc": "7,790-dim multimodal fusion: 6,670 static Pearson FC + 1,120 deep Mamba representations"
        }
    }
    
    probe_results = {}
    
    # Check end-to-end performance on these exact subsets
    for head_idx, head_name in enumerate(["Head1_cls_linear", "Head2_mlp_cls"]):
        logits_key = "logits1" if head_idx == 0 else "logits2"
        v_preds = np.argmax(val_extracted[logits_key], axis=-1)
        v_probs = F.softmax(torch.tensor(val_extracted[logits_key]), dim=-1)[:, 1].numpy()
        t_preds = np.argmax(test_extracted[logits_key], axis=-1)
        t_probs = F.softmax(torch.tensor(test_extracted[logits_key]), dim=-1)[:, 1].numpy()
        
        vm = calc_all_metrics(y_val, v_preds, v_probs)
        tm = calc_all_metrics(y_test, t_preds, t_probs)
        print(f"\n[End-to-End {head_name}]")
        print(f"  Val Acc: {vm['accuracy']*100:.2f}% | Val BAcc: {vm['balanced_accuracy']*100:.2f}% | Val AUC: {vm['auc']*100:.2f}%")
        print(f"  Test Acc: {tm['accuracy']*100:.2f}% | Test BAcc: {tm['balanced_accuracy']*100:.2f}% | Test AUC: {tm['auc']*100:.2f}%")
        
        probe_results[f"EndToEnd_{head_name}"] = {
            "val_metrics": vm,
            "test_metrics": tm
        }
    
    # Train linear probes
    for feat_name, feats in feature_sets.items():
        print("\n" + "=" * 65)
        print(f"LINEAR PROBE: {feat_name} ({feats['desc']})")
        print("=" * 65)
        
        # Scaling fit strictly on train
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feats["train"])
        X_va = scaler.transform(feats["val"])
        X_te = scaler.transform(feats["test"])
        
        # 1. Logistic Regression (L2)
        c_grid = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0, 10.0]
        best_c = None
        best_val_score = -1.0
        best_clf = None
        
        for c in c_grid:
            clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=0)
            clf.fit(X_tr, y_train)
            val_preds = clf.predict(X_va)
            val_probs = clf.predict_proba(X_va)[:, 1]
            m = calc_all_metrics(y_val, val_preds, val_probs)
            score = m["balanced_accuracy"] + 0.01 * m["auc"]
            if score > best_val_score:
                best_val_score = score
                best_c = c
                best_clf = clf
                
        test_preds = best_clf.predict(X_te)
        test_probs = best_clf.predict_proba(X_te)[:, 1]
        best_val_m = calc_all_metrics(y_val, best_clf.predict(X_va), best_clf.predict_proba(X_va)[:, 1])
        test_m = calc_all_metrics(y_test, test_preds, test_probs)
        
        print(f"  Logistic Regression (L2, C={best_c}):")
        print(f"    Val Acc:  {best_val_m['accuracy']*100:.2f}% | Val BAcc: {best_val_m['balanced_accuracy']*100:.2f}% | Val AUC: {best_val_m['auc']*100:.2f}%")
        print(f"    Test Acc: {test_m['accuracy']*100:.2f}% | Test BAcc: {test_m['balanced_accuracy']*100:.2f}% | Test AUC: {test_m['auc']*100:.2f}%")
        print(f"    Confusion Matrix: {test_m['confusion_matrix']}")
        
        # 2. Ridge Classifier
        alpha_grid = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
        best_alpha = None
        best_val_score_r = -1.0
        best_ridge = None
        for a in alpha_grid:
            rclf = RidgeClassifier(alpha=a, random_state=0)
            rclf.fit(X_tr, y_train)
            val_preds = rclf.predict(X_va)
            val_dec = rclf.decision_function(X_va)
            val_probs = 1.0 / (1.0 + np.exp(-val_dec))
            m = calc_all_metrics(y_val, val_preds, val_probs)
            score = m["balanced_accuracy"] + 0.01 * m["auc"]
            if score > best_val_score_r:
                best_val_score_r = score
                best_alpha = a
                best_ridge = rclf
                
        test_preds_r = best_ridge.predict(X_te)
        test_dec_r = best_ridge.decision_function(X_te)
        test_probs_r = 1.0 / (1.0 + np.exp(-test_dec_r))
        val_dec_best = best_ridge.decision_function(X_va)
        val_probs_best = 1.0 / (1.0 + np.exp(-val_dec_best))
        best_val_mr = calc_all_metrics(y_val, best_ridge.predict(X_va), val_probs_best)
        test_mr = calc_all_metrics(y_test, test_preds_r, test_probs_r)
        
        print(f"  Ridge Classifier (alpha={best_alpha}):")
        print(f"    Val Acc:  {best_val_mr['accuracy']*100:.2f}% | Val BAcc: {best_val_mr['balanced_accuracy']*100:.2f}% | Val AUC: {best_val_mr['auc']*100:.2f}%")
        print(f"    Test Acc: {test_mr['accuracy']*100:.2f}% | Test BAcc: {test_mr['balanced_accuracy']*100:.2f}% | Test AUC: {test_mr['auc']*100:.2f}%")
        print(f"    Confusion Matrix: {test_mr['confusion_matrix']}")
        
        probe_results[feat_name] = {
            "description": feats["desc"],
            "dim": int(X_tr.shape[1]),
            "LogisticRegression_L2": {
                "best_C": best_c,
                "val_metrics": best_val_m,
                "test_metrics": test_m
            },
            "RidgeClassifier": {
                "best_alpha": best_alpha,
                "val_metrics": best_val_mr,
                "test_metrics": test_mr
            }
        }
        
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(probe_results, f, indent=2)
    print(f"\nAll Linear Probe results saved to: {out_file}")

if __name__ == "__main__":
    run_linear_probe()
