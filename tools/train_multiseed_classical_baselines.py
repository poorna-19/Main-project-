"""
Train and evaluate classical linear baselines (Logistic L2, Ridge, Linear SVM)
across all 5 pre-audited seeds (Seeds 0 to 4).
Protocol:
- Scaler strictly fit on train_idx
- Hyperparameters selected strictly on val_idx
- Evaluated strictly once on test_idx
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from tools.train_classical_baselines import extract_upper_tri, calc_all_metrics

def main():
    splits_dir = "result/phase6_multiseed/splits"
    dataset_path = "E:/abide_data/abide_284.npy"
    out_file = "result/phase6_multiseed/classical_baselines_multiseed.json"
    
    data = np.load(dataset_path, allow_pickle=True).item()
    X_raw = extract_upper_tri(data["corr"])
    labels_raw = data["label"]
    y_raw = np.argmax(labels_raw, axis=-1) if labels_raw.ndim > 1 else labels_raw.astype(int)
    
    all_results = {}
    c_grid_l2 = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0, 10.0]
    alpha_grid = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    c_grid_svm = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for seed in range(5):
        split_file = os.path.join(splits_dir, f"exact_splits_seed{seed}.json")
        with open(split_file, "r") as f:
            split_data = json.load(f)
            
        train_idx = np.array(split_data["train"]["indices"])
        val_idx = np.array(split_data["val"]["indices"])
        test_idx = np.array(split_data["test"]["indices"])
        
        X_train, y_train = X_raw[train_idx], y_raw[train_idx]
        X_val, y_val = X_raw[val_idx], y_raw[val_idx]
        X_test, y_test = X_raw[test_idx], y_raw[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        # 1. Logistic Regression (L2)
        best_c, best_score, best_lr = None, -1.0, None
        for c in c_grid_l2:
            clf = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=2000, random_state=seed)
            clf.fit(X_train_s, y_train)
            v_preds = clf.predict(X_val_s)
            v_probs = clf.predict_proba(X_val_s)[:, 1]
            m = calc_all_metrics(y_val, v_preds, v_probs)
            score = m["balanced_accuracy"] + 0.01 * m["auc"]
            if score > best_score:
                best_score = score
                best_c = c
                best_lr = clf
        t_preds = best_lr.predict(X_test_s)
        t_probs = best_lr.predict_proba(X_test_s)[:, 1]
        m_lr_val = calc_all_metrics(y_val, best_lr.predict(X_val_s), best_lr.predict_proba(X_val_s)[:, 1])
        m_lr_test = calc_all_metrics(y_test, t_preds, t_probs)
        
        # 2. Ridge Classifier
        best_a, best_score_r, best_ridge = None, -1.0, None
        for a in alpha_grid:
            rclf = RidgeClassifier(alpha=a, random_state=seed)
            rclf.fit(X_train_s, y_train)
            v_preds = rclf.predict(X_val_s)
            v_dec = rclf.decision_function(X_val_s)
            v_probs = 1.0 / (1.0 + np.exp(-v_dec))
            m = calc_all_metrics(y_val, v_preds, v_probs)
            score = m["balanced_accuracy"] + 0.01 * m["auc"]
            if score > best_score_r:
                best_score_r = score
                best_a = a
                best_ridge = rclf
        t_preds = best_ridge.predict(X_test_s)
        t_dec = best_ridge.decision_function(X_test_s)
        t_probs = 1.0 / (1.0 + np.exp(-t_dec))
        m_ridge_val = calc_all_metrics(y_val, best_ridge.predict(X_val_s), 1.0 / (1.0 + np.exp(-best_ridge.decision_function(X_val_s))))
        m_ridge_test = calc_all_metrics(y_test, t_preds, t_probs)
        
        # 3. Linear SVM
        best_c_svm, best_score_s, best_svm = None, -1.0, None
        for c in c_grid_svm:
            sclf = LinearSVC(C=c, penalty='l2', max_iter=3000, random_state=seed)
            sclf.fit(X_train_s, y_train)
            v_preds = sclf.predict(X_val_s)
            v_dec = sclf.decision_function(X_val_s)
            v_probs = 1.0 / (1.0 + np.exp(-v_dec))
            m = calc_all_metrics(y_val, v_preds, v_probs)
            score = m["balanced_accuracy"] + 0.01 * m["auc"]
            if score > best_score_s:
                best_score_s = score
                best_c_svm = c
                best_svm = sclf
        t_preds = best_svm.predict(X_test_s)
        t_dec = best_svm.decision_function(X_test_s)
        t_probs = 1.0 / (1.0 + np.exp(-t_dec))
        m_svm_val = calc_all_metrics(y_val, best_svm.predict(X_val_s), 1.0 / (1.0 + np.exp(-best_svm.decision_function(X_val_s))))
        m_svm_test = calc_all_metrics(y_test, t_preds, t_probs)
        
        all_results[f"seed_{seed}"] = {
            "LogisticRegression_L2": {"best_C": best_c, "val": m_lr_val, "test": m_lr_test},
            "RidgeClassifier": {"best_alpha": best_a, "val": m_ridge_val, "test": m_ridge_test},
            "LinearSVC": {"best_C": best_c_svm, "val": m_svm_val, "test": m_svm_test}
        }
        print(f"Seed {seed} Classical Baselines:")
        print(f"  Logistic L2 (C={best_c}): Test Acc={m_lr_test['accuracy']*100:.2f}%, BAcc={m_lr_test['balanced_accuracy']*100:.2f}%, SEN={m_lr_test['sensitivity']*100:.2f}%, SPE={m_lr_test['specificity']*100:.2f}%, AUC={m_lr_test['auc']*100:.2f}%")
        print(f"  Ridge (a={best_a}):       Test Acc={m_ridge_test['accuracy']*100:.2f}%, BAcc={m_ridge_test['balanced_accuracy']*100:.2f}%, SEN={m_ridge_test['sensitivity']*100:.2f}%, SPE={m_ridge_test['specificity']*100:.2f}%, AUC={m_ridge_test['auc']*100:.2f}%")
        print(f"  Linear SVM (C={best_c_svm}):  Test Acc={m_svm_test['accuracy']*100:.2f}%, BAcc={m_svm_test['balanced_accuracy']*100:.2f}%, SEN={m_svm_test['sensitivity']*100:.2f}%, SPE={m_svm_test['specificity']*100:.2f}%, AUC={m_svm_test['auc']*100:.2f}%")
        
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll multi-seed classical baselines saved to: {out_file}")

if __name__ == "__main__":
    main()
