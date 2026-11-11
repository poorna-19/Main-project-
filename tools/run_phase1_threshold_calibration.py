"""
Phase 1: Validation-Only Post-Hoc Threshold Calibration
Implementation and evaluation for CDCo-MambaNet (full_seed0.pt).

Rules strictly followed:
- Threshold is optimized strictly on the validation set only.
- Test set is evaluated once with the frozen threshold.
- No retraining, no architecture modification, no checkpoint alteration.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import dataset_factory
from models.CDCo_MambaNet import CDCo_MambaNet

OUT_DIR = ROOT / 'reports' / 'Phase1_Threshold_Calibration'
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def load_model(device='cuda'):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        cfg = compose('config', overrides=[
            'dataset=ABIDE',
            'dataset.path=E:/abide_data/abide_284.npy',
            'dataset.expand=false',
            'model=bnt',
            'model.cls_num=2',
            'model.reliability_fusion=true',
            'model.dynamic_dim=true'
        ])
    with open_dict(cfg):
        cfg.dataset.node_sz = 116
        cfg.dataset.node_feature_sz = 116
        cfg.dataset.timeseries_sz = 115

    model = CDCo_MambaNet(cfg).to(device)
    w = ROOT / 'demo' / 'full_seed0.pt'
    state = torch.load(w, map_location=device, weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.eval()
    return model, cfg


@torch.no_grad()
def run_inference(model, loader, device='cuda'):
    model.eval()
    logits0_list, logits1_list, y_list = [], [], []
    torch.manual_seed(0)
    for f, c, y, s in loader:
        f, c, s = f.to(device), c.to(device), s.to(device)
        p0, p1 = model(f, c, s)
        logits0_list.append(p0.cpu())
        logits1_list.append(p1.cpu())
        y_list.append(y.cpu())

    logits1 = torch.cat(logits1_list)
    y_tensor = torch.cat(y_list)
    y_true = y_tensor.argmax(1).numpy() if y_tensor.ndim > 1 else y_tensor.numpy()
    probs = F.softmax(logits1, dim=-1)[:, 1].numpy()
    return probs, y_true


def compute_metrics(y_true, probs, threshold=0.50):
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, pred)
    bacc = balanced_accuracy_score(y_true, pred)
    sen = recall_score(y_true, pred, pos_label=1, zero_division=0)
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec_asd = precision_score(y_true, pred, pos_label=1, zero_division=0)
    prec_ctrl = precision_score(y_true, pred, pos_label=0, zero_division=0)
    f1_macro = f1_score(y_true, pred, average='macro', zero_division=0)
    f1_asd = f1_score(y_true, pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = 0.5
    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'balanced_accuracy': float(bacc),
        'sensitivity': float(sen),
        'specificity': float(spe),
        'precision_asd': float(prec_asd),
        'precision_control': float(prec_ctrl),
        'f1_macro': float(f1_macro),
        'f1_asd': float(f1_asd),
        'auc': float(auc),
        'cm': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def find_optimal_validation_threshold(val_y, val_probs):
    fpr, tpr, thresholds = roc_curve(val_y, val_probs)
    val_auc = float(roc_auc_score(val_y, val_probs))

    # 1. Youden's J statistic
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    th_youden = float(thresholds[best_j_idx])
    max_j = float(j_scores[best_j_idx])

    # 2. Balanced Accuracy Sweep across all unique probability cutoffs
    cand_thresh = np.unique(val_probs)
    bacc_list, acc_list, f1_list, sens_list, spec_list = [], [], [], [], []
    for t in cand_thresh:
        m = compute_metrics(val_y, val_probs, threshold=t)
        bacc_list.append((m['balanced_accuracy'], t))
        acc_list.append((m['accuracy'], t))
        f1_list.append((m['f1_macro'], t))

    bacc_list.sort(reverse=True)
    th_bacc = float(bacc_list[0][1])
    best_bacc = float(bacc_list[0][0])

    f1_list.sort(reverse=True)
    th_f1 = float(f1_list[0][1])
    best_f1 = float(f1_list[0][0])

    # Selected frozen threshold
    selected_threshold = th_youden

    # Threshold curve data for visualization
    sweep_thresholds = np.linspace(val_probs.min() - 0.05, val_probs.max() + 0.05, 200)
    sweep_data = []
    for t in sweep_thresholds:
        m = compute_metrics(val_y, val_probs, threshold=t)
        sweep_data.append({
            'threshold': float(t),
            'bacc': m['balanced_accuracy'],
            'acc': m['accuracy'],
            'sen': m['sensitivity'],
            'spe': m['specificity'],
            'f1': m['f1_macro'],
            'youden': m['sensitivity'] + m['specificity'] - 1.0
        })

    calibration_info = {
        'methodology': "Validation-Only Post-Hoc Threshold Calibration via Youden's J and Balanced Accuracy",
        'validation_cohort_size': len(val_y),
        'validation_controls': int((val_y == 0).sum()),
        'validation_autism': int((val_y == 1).sum()),
        'validation_auc': val_auc,
        'youden_optimal_threshold': th_youden,
        'youden_max_j': max_j,
        'bacc_optimal_threshold': th_bacc,
        'bacc_max_val': best_bacc,
        'f1_optimal_threshold': th_f1,
        'f1_max_val': best_f1,
        'selected_frozen_threshold': selected_threshold,
        'val_metrics_default': compute_metrics(val_y, val_probs, threshold=0.50),
        'val_metrics_calibrated': compute_metrics(val_y, val_probs, threshold=selected_threshold)
    }

    return selected_threshold, calibration_info, sweep_data, (fpr, tpr, thresholds)


def plot_threshold_selection(sweep_data, selected_th, out_path):
    th = [d['threshold'] for d in sweep_data]
    bacc = [d['bacc'] * 100 for d in sweep_data]
    acc = [d['acc'] * 100 for d in sweep_data]
    sen = [d['sen'] * 100 for d in sweep_data]
    spe = [d['spe'] * 100 for d in sweep_data]
    f1 = [d['f1'] * 100 for d in sweep_data]
    j = [d['youden'] * 100 for d in sweep_data]

    plt.figure(figsize=(9, 5.5))
    plt.plot(th, bacc, label='Balanced Accuracy (%)', color='#2ca02c', lw=2.5)
    plt.plot(th, acc, label='Accuracy (%)', color='#1f77b4', lw=2.0, linestyle='--')
    plt.plot(th, f1, label='Macro F1 (%)', color='#ff7f0e', lw=2.0)
    plt.plot(th, sen, label='Sensitivity (Recall ASD) (%)', color='#d62728', lw=1.5, alpha=0.8)
    plt.plot(th, spe, label='Specificity (Recall Ctrl) (%)', color='#9467bd', lw=1.5, alpha=0.8)
    plt.plot(th, j, label="Youden's J (%)", color='#8c564b', lw=1.8, linestyle=':')

    plt.axvline(0.50, color='gray', linestyle='--', lw=1.5, label='Default Threshold (0.50)')
    plt.axvline(selected_th, color='#d62728', linestyle='-', lw=2.5,
                label=f'Selected Frozen Threshold ($\\theta^* = {selected_th:.4f}$)')

    plt.xlabel('Classification Threshold ($\\theta$)')
    plt.ylabel('Validation Metric Score (%)')
    plt.title("Validation-Only Threshold Selection Criteria\n(Optimization strictly on Validation Cohort, N=29)",
              fontweight='bold', fontsize=12)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([min(th), max(th)])
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_roc_comparison(val_roc, test_roc, val_cal, test_cal, out_path):
    fpr_v, tpr_v, _ = val_roc
    fpr_t, tpr_t, _ = test_roc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Validation Set ROC
    ax1 = axes[0]
    ax1.plot(fpr_v, tpr_v, color='#1f77b4', lw=2.5,
             label=f'Validation ROC (AUC = {val_cal["validation_auc"]*100:.1f}%)')
    # Default point
    m_v_def = val_cal['val_metrics_default']
    fpr_v_def = m_v_def['fp'] / (m_v_def['tn'] + m_v_def['fp'])
    tpr_v_def = m_v_def['tp'] / (m_v_def['fn'] + m_v_def['tp'])
    ax1.plot(fpr_v_def, tpr_v_def, 's', color='#1f77b4', markersize=10,
             label=f'Val Default $\\theta=0.50$ (Acc={m_v_def["accuracy"]*100:.1f}%)')
    # Calibrated point
    m_v_cal = val_cal['val_metrics_calibrated']
    fpr_v_cal = m_v_cal['fp'] / (m_v_cal['tn'] + m_v_cal['fp'])
    tpr_v_cal = m_v_cal['tp'] / (m_v_cal['fn'] + m_v_cal['tp'])
    ax1.plot(fpr_v_cal, tpr_v_cal, '*', color='#d62728', markersize=14,
             label=f'Val Calibrated $\\theta^*={val_cal["selected_frozen_threshold"]:.4f}$ (Acc={m_v_cal["accuracy"]*100:.1f}%)')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('Validation Dataset Calibration ROC\n(Criteria Selection Set, N=29)', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Independent Test Set ROC
    ax2 = axes[1]
    ax2.plot(fpr_t, tpr_t, color='#2ca02c', lw=2.5,
             label=f'Test ROC (AUC = {test_cal["auc"]*100:.1f}%)')
    m_t_def = test_cal['test_metrics_default']
    fpr_t_def = m_t_def['fp'] / (m_t_def['tn'] + m_t_def['fp'])
    tpr_t_def = m_t_def['tp'] / (m_t_def['fn'] + m_t_def['tp'])
    ax2.plot(fpr_t_def, tpr_t_def, 's', color='#1f77b4', markersize=10,
             label=f'Test Default $\\theta=0.50$ (Acc={m_t_def["accuracy"]*100:.1f}%)')
    m_t_cal = test_cal['test_metrics_calibrated']
    fpr_t_cal = m_t_cal['fp'] / (m_t_cal['tn'] + m_t_cal['fp'])
    tpr_t_cal = m_t_cal['tp'] / (m_t_cal['fn'] + m_t_cal['tp'])
    ax2.plot(fpr_t_cal, tpr_t_cal, '*', color='#d62728', markersize=14,
             label=f'Test Frozen $\\theta^*={val_cal["selected_frozen_threshold"]:.4f}$ (Acc={m_t_cal["accuracy"]*100:.1f}%)')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate (1 - Specificity)')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    ax2.set_title('Independent Held-Out Test Set ROC\n(Frozen Threshold Evaluated Once, N=28)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("CDCo-MambaNet: Validation vs Independent Test ROC & Calibrated Operating Points",
                 fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_confusion_matrices(val_cal, test_cal, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top-Left: Val Default
    cm_v_def = np.array(val_cal['val_metrics_default']['cm'])
    axes[0, 0].imshow(cm_v_def, cmap=plt.cm.Blues, vmin=0, vmax=16)
    axes[0, 0].set_title(f"Validation: Default $\\theta=0.50$\nAcc={val_cal['val_metrics_default']['accuracy']*100:.1f}%, Sen={val_cal['val_metrics_default']['sensitivity']*100:.1f}%, Spe={val_cal['val_metrics_default']['specificity']*100:.1f}%", fontweight='bold', fontsize=10)

    # Top-Right: Val Calibrated
    cm_v_cal = np.array(val_cal['val_metrics_calibrated']['cm'])
    axes[0, 1].imshow(cm_v_cal, cmap=plt.cm.Greens, vmin=0, vmax=16)
    axes[0, 1].set_title(f"Validation: Calibrated $\\theta^*={val_cal['selected_frozen_threshold']:.4f}$\nAcc={val_cal['val_metrics_calibrated']['accuracy']*100:.1f}%, Sen={val_cal['val_metrics_calibrated']['sensitivity']*100:.1f}%, Spe={val_cal['val_metrics_calibrated']['specificity']*100:.1f}%", fontweight='bold', fontsize=10)

    # Bottom-Left: Test Default
    cm_t_def = np.array(test_cal['test_metrics_default']['cm'])
    axes[1, 0].imshow(cm_t_def, cmap=plt.cm.Blues, vmin=0, vmax=16)
    axes[1, 0].set_title(f"Test: Default $\\theta=0.50$\nAcc={test_cal['test_metrics_default']['accuracy']*100:.1f}%, Sen={test_cal['test_metrics_default']['sensitivity']*100:.1f}%, Spe={test_cal['test_metrics_default']['specificity']*100:.1f}%", fontweight='bold', fontsize=10)

    # Bottom-Right: Test Calibrated
    cm_t_cal = np.array(test_cal['test_metrics_calibrated']['cm'])
    axes[1, 1].imshow(cm_t_cal, cmap=plt.cm.Greens, vmin=0, vmax=16)
    axes[1, 1].set_title(f"Test: Frozen Calibrated $\\theta^*={val_cal['selected_frozen_threshold']:.4f}$\nAcc={test_cal['test_metrics_calibrated']['accuracy']*100:.1f}%, Sen={test_cal['test_metrics_calibrated']['sensitivity']*100:.1f}%, Spe={test_cal['test_metrics_calibrated']['specificity']*100:.1f}%", fontweight='bold', fontsize=10)

    for ax, cm in zip(axes.flat, [cm_v_def, cm_v_cal, cm_t_def, cm_t_cal]):
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > 7 else 'black'
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Ctrl', 'Pred ASD'])
        ax.set_yticklabels(['True Ctrl', 'True ASD'])

    plt.suptitle("Confusion Matrix Shift: Default $\\theta=0.50$ vs. Validation-Calibrated $\\theta^*$",
                 fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing Phase 1 Threshold Calibration on {device}...")

    # 1. Load trained full_seed0.pt
    model, cfg = load_model(device=device)

    # 2. Get dataloaders
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    train_loader, val_loader, test_loader = dataset_factory(cfg)

    # 3. Validation-Only Inference
    print("Running inference on Validation dataset (N=29)...")
    val_probs, val_y = run_inference(model, val_loader, device=device)

    # 4. Find optimal threshold on validation set
    selected_th, val_cal, sweep_data, val_roc = find_optimal_validation_threshold(val_y, val_probs)
    print(f"-> Selected Frozen Threshold: {selected_th:.4f}")
    print(f"   Validation Default (0.50): Acc={val_cal['val_metrics_default']['accuracy']*100:.2f}%, BAcc={val_cal['val_metrics_default']['balanced_accuracy']*100:.2f}%, CM={val_cal['val_metrics_default']['cm']}")
    print(f"   Validation Calibrated ({selected_th:.4f}): Acc={val_cal['val_metrics_calibrated']['accuracy']*100:.2f}%, BAcc={val_cal['val_metrics_calibrated']['balanced_accuracy']*100:.2f}%, CM={val_cal['val_metrics_calibrated']['cm']}")

    # 5. Save calibration info
    cal_json_path = OUT_DIR / 'validation_threshold_calibration.json'
    with open(cal_json_path, 'w') as f:
        json.dump(val_cal, f, indent=2)
    print(f"Saved: {cal_json_path}")

    # 6. Evaluate independent test dataset ONCE with the frozen threshold
    print("Evaluating held-out Test dataset (N=28) with frozen threshold...")
    test_probs, test_y = run_inference(model, test_loader, device=device)

    fpr_t, tpr_t, thresh_t = roc_curve(test_y, test_probs)
    test_auc = float(roc_auc_score(test_y, test_probs))
    test_cal = {
        'test_cohort_size': len(test_y),
        'test_controls': int((test_y == 0).sum()),
        'test_autism': int((test_y == 1).sum()),
        'auc': test_auc,
        'test_metrics_default': compute_metrics(test_y, test_probs, threshold=0.50),
        'test_metrics_calibrated': compute_metrics(test_y, test_probs, threshold=selected_th)
    }

    test_json_path = OUT_DIR / 'independent_test_evaluation.json'
    with open(test_json_path, 'w') as f:
        json.dump(test_cal, f, indent=2)
    print(f"Saved: {test_json_path}")

    # 7. Generate visualizations
    plot_threshold_selection(sweep_data, selected_th, OUT_DIR / 'threshold_selection_curves.png')
    plot_roc_comparison(val_roc, (fpr_t, tpr_t, thresh_t), val_cal, test_cal, OUT_DIR / 'roc_curve_calibration.png')
    plot_confusion_matrices(val_cal, test_cal, OUT_DIR / 'confusion_matrices_comparison.png')

    print("\nPhase 1 Calibration Pipeline complete!")


if __name__ == '__main__':
    main()
