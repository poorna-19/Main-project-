"""
Phase 6: Classical Baseline Training & Evaluation on Identical Subject Splits
Models:
1. Logistic Regression (L2 penalty)
2. Logistic Regression (L1 penalty - Sparse)
3. Ridge Classifier (L2 penalty)
4. Linear SVM (LinearSVC)
5. Logistic Regression trained on 4x Augmented Train Set

All feature scaling and hyperparameter selection strictly use Train/Val only.
Held-out test set is evaluated ONLY ONCE per finalized model.
"""
import os
import sys
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score, recall_score
)

def extract_upper_tri(corr_matrices):
    """Extract upper-triangle off-diagonal features (6670 features for 116x116)."""
    n_subs = len(corr_matrices)
    triu_idx = np.triu_indices(116, k=1)
    features = np.zeros((n_subs, len(triu_idx[0])), dtype=np.float32)
    for i in range(n_subs):
        mat = np.array(corr_matrices[i])
        features[i] = mat[triu_idx]
    return features

def calc_all_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # ASD recall
    spe = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0        # Control recall
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "sensitivity": float(sen),
        "specificity": float(spe),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

def run_baselines():
    splits_path = "result/phase6_baselines_and_regularization/exact_splits_seed0.json"
    dataset_path = "E:/abide_data/abide_284.npy"
    
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    data = np.load(dataset_path, allow_pickle=True).item()
    corr = data["corr"]
    labels = data["label"]
    labels_flat = np.argmax(labels, axis=-1) if labels.ndim > 1 else labels.astype(int)
    
    train_idx = np.array(splits["train"]["indices"])
    val_idx = np.array(splits["val"]["indices"])
    test_idx = np.array(splits["test"]["indices"])
    
    y_train = labels_flat[train_idx]
    y_val = labels_flat[val_idx]
    y_test = labels_flat[test_idx]
    
    print(f"Cohort check: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # 1. Feature extraction
    X_raw = extract_upper_tri(corr)
    X_train = X_raw[train_idx]
    X_val = X_raw[val_idx]
    X_test = X_raw[test_idx]
    print(f"Feature shape: {X_train.shape} (6,670 functional connectivity features per subject)")
    
    # 2. Feature scaling strictly fit on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Baseline 1: Logistic Regression (L2 Regularization)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 1: Logistic Regression (L2)")
    print("=" * 60)
    c_grid_l2 = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0, 10.0]
    best_c_l2 = None
    best_val_score_l2 = -1.0
    best_model_l2 = None
    val_records_l2 = []
    
    for c in c_grid_l2:
        clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=0)
        clf.fit(X_train_scaled, y_train)
        val_preds = clf.predict(X_val_scaled)
        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        m = calc_all_metrics(y_val, val_preds, val_probs)
        # Model selection: prioritize balanced accuracy with AUC tie-breaker
        score = m["balanced_accuracy"] + 0.01 * m["auc"]
        val_records_l2.append({"C": c, **m})
        print(f"  C={c:<7} | Val Acc: {m['accuracy']*100:.2f}% | Val BAcc: {m['balanced_accuracy']*100:.2f}% | Val AUC: {m['auc']*100:.2f}% | Val F1: {m['f1']*100:.2f}%")
        if score > best_val_score_l2:
            best_val_score_l2 = score
            best_c_l2 = c
            best_model_l2 = clf
            
    # Evaluate best L2 model on Test
    test_preds_l2 = best_model_l2.predict(X_test_scaled)
    test_probs_l2 = best_model_l2.predict_proba(X_test_scaled)[:, 1]
    best_val_m_l2 = calc_all_metrics(y_val, best_model_l2.predict(X_val_scaled), best_model_l2.predict_proba(X_val_scaled)[:, 1])
    test_m_l2 = calc_all_metrics(y_test, test_preds_l2, test_probs_l2)
    print(f"-> Selected C={best_c_l2}: Val Acc={best_val_m_l2['accuracy']*100:.2f}%, Test Acc={test_m_l2['accuracy']*100:.2f}%, Test AUC={test_m_l2['auc']*100:.2f}%")
    
    results["LogisticRegression_L2"] = {
        "best_hyperparameter": {"C": best_c_l2},
        "validation_metrics": best_val_m_l2,
        "test_metrics": test_m_l2,
        "tuning_history": val_records_l2
    }

    # -------------------------------------------------------------------------
    # Baseline 2: Logistic Regression (L1 Regularization - Sparse Lasso)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 2: Logistic Regression (L1 - Sparse Feature Selection)")
    print("=" * 60)
    c_grid_l1 = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.2, 0.5, 1.0]
    best_c_l1 = None
    best_val_score_l1 = -1.0
    best_model_l1 = None
    val_records_l1 = []
    
    for c in c_grid_l1:
        clf = LogisticRegression(C=c, penalty='l1', solver='liblinear', max_iter=2000, random_state=0)
        clf.fit(X_train_scaled, y_train)
        val_preds = clf.predict(X_val_scaled)
        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        non_zero = int(np.sum(clf.coef_ != 0))
        m = calc_all_metrics(y_val, val_preds, val_probs)
        score = m["balanced_accuracy"] + 0.01 * m["auc"]
        val_records_l1.append({"C": c, "nonzero_features": non_zero, **m})
        print(f"  C={c:<7} | Active Features: {non_zero:>4}/6670 | Val Acc: {m['accuracy']*100:.2f}% | Val BAcc: {m['balanced_accuracy']*100:.2f}% | Val AUC: {m['auc']*100:.2f}%")
        if score > best_val_score_l1:
            best_val_score_l1 = score
            best_c_l1 = c
            best_model_l1 = clf

    test_preds_l1 = best_model_l1.predict(X_test_scaled)
    test_probs_l1 = best_model_l1.predict_proba(X_test_scaled)[:, 1]
    best_val_m_l1 = calc_all_metrics(y_val, best_model_l1.predict(X_val_scaled), best_model_l1.predict_proba(X_val_scaled)[:, 1])
    test_m_l1 = calc_all_metrics(y_test, test_preds_l1, test_probs_l1)
    print(f"-> Selected C={best_c_l1}: Nonzero Features={int(np.sum(best_model_l1.coef_ != 0))}, Val Acc={best_val_m_l1['accuracy']*100:.2f}%, Test Acc={test_m_l1['accuracy']*100:.2f}%, Test AUC={test_m_l1['auc']*100:.2f}%")
    
    results["LogisticRegression_L1"] = {
        "best_hyperparameter": {"C": best_c_l1, "nonzero_features": int(np.sum(best_model_l1.coef_ != 0))},
        "validation_metrics": best_val_m_l1,
        "test_metrics": test_m_l1,
        "tuning_history": val_records_l1
    }

    # -------------------------------------------------------------------------
    # Baseline 3: Ridge Classifier
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 3: Ridge Classifier (L2 Linear Classifier)")
    print("=" * 60)
    alpha_grid = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    best_alpha = None
    best_val_score_ridge = -1.0
    best_model_ridge = None
    val_records_ridge = []
    
    for alpha in alpha_grid:
        clf = RidgeClassifier(alpha=alpha, random_state=0)
        clf.fit(X_train_scaled, y_train)
        val_preds = clf.predict(X_val_scaled)
        val_decision = clf.decision_function(X_val_scaled)
        # Sigmoid of decision function for AUC calculation
        val_probs = 1.0 / (1.0 + np.exp(-val_decision))
        m = calc_all_metrics(y_val, val_preds, val_probs)
        score = m["balanced_accuracy"] + 0.01 * m["auc"]
        val_records_ridge.append({"alpha": alpha, **m})
        print(f"  alpha={alpha:<7} | Val Acc: {m['accuracy']*100:.2f}% | Val BAcc: {m['balanced_accuracy']*100:.2f}% | Val AUC: {m['auc']*100:.2f}%")
        if score > best_val_score_ridge:
            best_val_score_ridge = score
            best_alpha = alpha
            best_model_ridge = clf

    test_preds_ridge = best_model_ridge.predict(X_test_scaled)
    test_decision_ridge = best_model_ridge.decision_function(X_test_scaled)
    test_probs_ridge = 1.0 / (1.0 + np.exp(-test_decision_ridge))
    val_decision_best = best_model_ridge.decision_function(X_val_scaled)
    best_val_m_ridge = calc_all_metrics(y_val, best_model_ridge.predict(X_val_scaled), 1.0 / (1.0 + np.exp(-val_decision_best)))
    test_m_ridge = calc_all_metrics(y_test, test_preds_ridge, test_probs_ridge)
    print(f"-> Selected alpha={best_alpha}: Val Acc={best_val_m_ridge['accuracy']*100:.2f}%, Test Acc={test_m_ridge['accuracy']*100:.2f}%, Test AUC={test_m_ridge['auc']*100:.2f}%")
    
    results["RidgeClassifier"] = {
        "best_hyperparameter": {"alpha": best_alpha},
        "validation_metrics": best_val_m_ridge,
        "test_metrics": test_m_ridge,
        "tuning_history": val_records_ridge
    }

    # -------------------------------------------------------------------------
    # Baseline 4: Linear SVM (LinearSVC)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 4: Linear Support Vector Machine (LinearSVC)")
    print("=" * 60)
    c_grid_svm = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1.0]
    best_c_svm = None
    best_val_score_svm = -1.0
    best_model_svm = None
    val_records_svm = []
    
    for c in c_grid_svm:
        clf = LinearSVC(C=c, max_iter=3000, random_state=0)
        clf.fit(X_train_scaled, y_train)
        val_preds = clf.predict(X_val_scaled)
        val_decision = clf.decision_function(X_val_scaled)
        val_probs = 1.0 / (1.0 + np.exp(-val_decision))
        m = calc_all_metrics(y_val, val_preds, val_probs)
        score = m["balanced_accuracy"] + 0.01 * m["auc"]
        val_records_svm.append({"C": c, **m})
        print(f"  C={c:<7} | Val Acc: {m['accuracy']*100:.2f}% | Val BAcc: {m['balanced_accuracy']*100:.2f}% | Val AUC: {m['auc']*100:.2f}%")
        if score > best_val_score_svm:
            best_val_score_svm = score
            best_c_svm = c
            best_model_svm = clf

    test_preds_svm = best_model_svm.predict(X_test_scaled)
    test_decision_svm = best_model_svm.decision_function(X_test_scaled)
    test_probs_svm = 1.0 / (1.0 + np.exp(-test_decision_svm))
    val_dec_svm_best = best_model_svm.decision_function(X_val_scaled)
    best_val_m_svm = calc_all_metrics(y_val, best_model_svm.predict(X_val_scaled), 1.0 / (1.0 + np.exp(-val_dec_svm_best)))
    test_m_svm = calc_all_metrics(y_test, test_preds_svm, test_probs_svm)
    print(f"-> Selected C={best_c_svm}: Val Acc={best_val_m_svm['accuracy']*100:.2f}%, Test Acc={test_m_svm['accuracy']*100:.2f}%, Test AUC={test_m_svm['auc']*100:.2f}%")
    
    results["LinearSVC"] = {
        "best_hyperparameter": {"C": best_c_svm},
        "validation_metrics": best_val_m_svm,
        "test_metrics": test_m_svm,
        "tuning_history": val_records_svm
    }

    # Save all results
    out_json = "result/phase6_baselines_and_regularization/classical_baselines_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved classical baseline results to {out_json}")

if __name__ == "__main__":
    run_baselines()
