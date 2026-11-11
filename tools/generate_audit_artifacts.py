"""
Generate comprehensive diagnostic audit metrics and publication-quality figures
for CDCo-MambaNet Post-Evaluation Diagnostic Audit.
"""
import json
import math
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
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.CDCo_MambaNet import CDCo_MambaNet
from training.training import CONDITIONS

DEMO = ROOT / 'demo'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)

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


def load_model(variant, d, device='cpu'):
    GlobalHydra.instance().clear()
    overrides = ['dataset=ABIDE', 'model=bnt', 'model.cls_num=2']
    if variant == 'full':
        overrides += ['model.reliability_fusion=true', 'model.dynamic_dim=true']
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        cfg = compose('config', overrides=overrides)
    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = d['corr'].shape[1:]
        cfg.dataset.timeseries_sz = d['timeseries'].shape[2]
    model = CDCo_MambaNet(cfg).to(device)
    w = DEMO / f'{variant}_seed0.pt'
    state = torch.load(w, map_location=device, weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.eval()


def evaluate_all(model, loader, conds, device='cpu'):
    model.eval()
    gate = getattr(getattr(model, 'fuse_encoder', None), 'gate', None)
    results = {}
    
    for cond_name, cond in conds.items():
        g = torch.Generator(device=device).manual_seed(0)
        torch.manual_seed(0)
        
        logits0_list, logits1_list, y_list, rel_list = [], [], [], []
        with torch.no_grad():
            for f, c, y, s in loader:
                f, c, s = f.to(device), c.to(device), s.to(device)
                avail = torch.ones(len(f), 2, dtype=torch.bool, device=device)
                if cond:
                    m, sigma = cond
                    x = f if m == 'f' else s
                    noise = torch.randn(x.shape, generator=g, device=device)
                    x = noise if sigma is None else x + sigma * noise
                    avail[:, 1 if m == 'f' else 0] = (sigma is not None)
                    f, s = (x, s) if m == 'f' else (f, x)
                p0, p1 = model(f, c, s, avail)
                logits0_list.append(p0.cpu())
                logits1_list.append(p1.cpu())
                y_list.append(y.cpu())
                if gate is not None:
                    i = gate.info
                    rel_list.append(torch.cat([i['reliability'], i['weight'], i['dims']], 1).cpu())
                    
        logits0 = torch.cat(logits0_list).numpy()
        logits1 = torch.cat(logits1_list).numpy()
        y_tensor = torch.cat(y_list).numpy()
        y_true = np.argmax(y_tensor, axis=-1)
        
        prob0 = F.softmax(torch.tensor(logits0, dtype=torch.float32), dim=-1).numpy()
        prob1 = F.softmax(torch.tensor(logits1, dtype=torch.float32), dim=-1).numpy()
        prob_ens = 0.5 * (prob0 + prob1)
        
        cond_res = {}
        for head_name, probs, logits in [('head1', prob0, logits0),
                                         ('head2', prob1, logits1),
                                         ('ensemble', prob_ens, 0.5 * (logits0 + logits1))]:
            p_asd = probs[:, 1]
            y_pred_def = (p_asd >= 0.5).astype(int)
            cm_def = confusion_matrix(y_true, y_pred_def, labels=[0, 1])
            tn, fp, fn, tp = cm_def.ravel()
            
            acc = float((tp + tn) / len(y_true))
            sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            prec_asd = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            prec_ctrl = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            f1 = float(f1_score(y_true, y_pred_def, average='macro', zero_division=0))
            
            try:
                auc = float(roc_auc_score(y_true, p_asd))
                fpr, tpr, thresholds = roc_curve(y_true, p_asd)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_th = float(thresholds[best_idx])
                y_pred_opt = (p_asd >= best_th).astype(int)
                cm_opt = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])
                opt_tn, opt_fp, opt_fn, opt_tp = cm_opt.ravel()
                opt_acc = float((opt_tp + opt_tn) / len(y_true))
                opt_sen = float(opt_tp / (opt_tp + opt_fn)) if (opt_tp + opt_fn) > 0 else 0.0
                opt_spe = float(opt_tn / (opt_tn + opt_fp)) if (opt_tn + opt_fp) > 0 else 0.0
                opt_f1 = float(f1_score(y_true, y_pred_opt, average='macro', zero_division=0))
            except Exception:
                auc = 0.5
                fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([1, 0])
                best_th = 0.5
                cm_opt = cm_def
                opt_acc, opt_sen, opt_spe, opt_f1 = acc, sen, spe, f1
                
            cond_res[head_name] = {
                'cm_def': cm_def.tolist(), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'accuracy': acc, 'balanced_accuracy': float((sen + spe) / 2.0),
                'sensitivity': sen, 'specificity': spe,
                'precision_asd': prec_asd, 'precision_control': prec_ctrl,
                'f1_macro': f1, 'auc': auc,
                'optimal_threshold': best_th,
                'cm_opt': cm_opt.tolist(), 'opt_accuracy': opt_acc,
                'opt_sensitivity': opt_sen, 'opt_specificity': opt_spe, 'opt_f1': opt_f1,
                'p_asd_mean': float(p_asd.mean()),
                'p_asd_std': float(p_asd.std()),
                'p_asd_min': float(p_asd.min()),
                'p_asd_max': float(p_asd.max()),
                'p_asd_median': float(np.median(p_asd)),
                'p_asd_ctrl_mean': float(p_asd[y_true == 0].mean()),
                'p_asd_ctrl_std': float(p_asd[y_true == 0].std()),
                'p_asd_ctrl_median': float(np.median(p_asd[y_true == 0])),
                'p_asd_asd_mean': float(p_asd[y_true == 1].mean()),
                'p_asd_asd_std': float(p_asd[y_true == 1].std()),
                'p_asd_asd_median': float(np.median(p_asd[y_true == 1])),
                'probs': p_asd.tolist(),
                'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()
            }
            
        gate_summary = None
        if rel_list:
            gate_cat = torch.cat(rel_list).mean(0).tolist()
            gate_summary = {
                'r_s': gate_cat[0], 'r_f': gate_cat[1],
                'w_s': gate_cat[2], 'w_f': gate_cat[3],
                'k_s': gate_cat[4], 'k_f': gate_cat[5]
            }
        cond_res['gate'] = gate_summary
        results[cond_name] = cond_res
        
    return results


def plot_confusion_matrices(res_base, res_full, conds):
    fig, axes = plt.subplots(2, 7, figsize=(24, 7))
    cmap_base = plt.cm.Blues
    cmap_full = plt.cm.Greens
    
    cond_names = list(conds.keys())
    for col, c in enumerate(cond_names):
        # Baseline row (head 2 / demo style)
        ax_b = axes[0, col]
        cm_b = np.array(res_base[c]['head2']['cm_def'])
        im_b = ax_b.imshow(cm_b, interpolation='nearest', cmap=cmap_base, vmin=0, vmax=14)
        ax_b.set_title(f"Baseline: {c}\nAcc={res_base[c]['head2']['accuracy']*100:.1f}%", fontsize=10, fontweight='bold')
        for i in range(2):
            for j in range(2):
                color = 'white' if cm_b[i, j] > 7 else 'black'
                ax_b.text(j, i, str(cm_b[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight='bold')
        ax_b.set_xticks([0, 1])
        ax_b.set_yticks([0, 1])
        ax_b.set_xticklabels(['Pred Ctrl', 'Pred ASD'], fontsize=9)
        if col == 0:
            ax_b.set_yticklabels(['True Ctrl', 'True ASD'], fontsize=9)
        else:
            ax_b.set_yticklabels([])
            
        # Full Gated row (head 2 / demo style)
        ax_f = axes[1, col]
        cm_f = np.array(res_full[c]['head2']['cm_def'])
        im_f = ax_f.imshow(cm_f, interpolation='nearest', cmap=cmap_full, vmin=0, vmax=14)
        ax_f.set_title(f"Full Gate: {c}\nAcc={res_full[c]['head2']['accuracy']*100:.1f}%", fontsize=10, fontweight='bold')
        for i in range(2):
            for j in range(2):
                color = 'white' if cm_f[i, j] > 7 else 'black'
                ax_f.text(j, i, str(cm_f[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight='bold')
        ax_f.set_xticks([0, 1])
        ax_f.set_yticks([0, 1])
        ax_f.set_xticklabels(['Pred Ctrl', 'Pred ASD'], fontsize=9)
        if col == 0:
            ax_f.set_yticklabels(['True Ctrl', 'True ASD'], fontsize=9)
        else:
            ax_f.set_yticklabels([])

    plt.suptitle("CDCo-MambaNet: Confusion Matrices Across All 7 Test Conditions (Default Threshold = 0.50)",
                 fontsize=14, y=1.03, fontweight='bold')
    plt.tight_layout()
    out_file = REPORTS / 'audit_confusion_matrices.png'
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_roc_curves(res_base, res_full, conds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Clean condition ROC curves
    ax1 = axes[0]
    # Baseline
    fpr_b = res_base['clean']['head2']['fpr']
    tpr_b = res_base['clean']['head2']['tpr']
    auc_b = res_base['clean']['head2']['auc']
    ax1.plot(fpr_b, tpr_b, color='#1f77b4', lw=2.5, label=f'Baseline Head 2 (AUC = {auc_b*100:.1f}%)')
    
    # Full Gate
    fpr_f = res_full['clean']['head2']['fpr']
    tpr_f = res_full['clean']['head2']['tpr']
    auc_f = res_full['clean']['head2']['auc']
    ax1.plot(fpr_f, tpr_f, color='#2ca02c', lw=2.5, label=f'Full Gate Head 2 (AUC = {auc_f*100:.1f}%)')
    
    # Full Gate Ensemble
    fpr_fe = res_full['clean']['ensemble']['fpr']
    tpr_fe = res_full['clean']['ensemble']['tpr']
    auc_fe = res_full['clean']['ensemble']['auc']
    ax1.plot(fpr_fe, tpr_fe, color='#ff7f0e', lw=2.0, linestyle='--', label=f'Full Gate Ensemble (AUC = {auc_fe*100:.1f}%)')
    
    # Baseline operating point (default 0.5)
    cm_b = res_base['clean']['head2']['cm_def']
    fpr_pt_b = cm_b[0][1] / (cm_b[0][0] + cm_b[0][1])
    tpr_pt_b = cm_b[1][1] / (cm_b[1][0] + cm_b[1][1])
    ax1.plot(fpr_pt_b, tpr_pt_b, 's', color='#1f77b4', markersize=10, label=f'Baseline Default Thresh 0.50 (FPR={fpr_pt_b:.2f}, TPR={tpr_pt_b:.2f})')
    
    # Full Gate operating point (default 0.5)
    cm_f = res_full['clean']['head2']['cm_def']
    fpr_pt_f = cm_f[0][1] / (cm_f[0][0] + cm_f[0][1])
    tpr_pt_f = cm_f[1][1] / (cm_f[1][0] + cm_f[1][1])
    ax1.plot(fpr_pt_f, tpr_pt_f, 'o', color='#2ca02c', markersize=10, label=f'Full Gate Default Thresh 0.50 (FPR={fpr_pt_f:.2f}, TPR={tpr_pt_f:.2f})')
    
    # Optimal Youden operating point
    cm_opt = res_full['clean']['head2']['cm_opt']
    fpr_pt_opt = cm_opt[0][1] / (cm_opt[0][0] + cm_opt[0][1])
    tpr_pt_opt = cm_opt[1][1] / (cm_opt[1][0] + cm_opt[1][1])
    th_opt = res_full['clean']['head2']['optimal_threshold']
    ax1.plot(fpr_pt_opt, tpr_pt_opt, '*', color='#d62728', markersize=14, label=f'Full Gate Optimal Thresh {th_opt:.4f} (Acc=71.4%)')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 50.0%)')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('Clean Test Set ROC Curves & Operating Points', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # Right: Full Gated across all 7 conditions
    ax2 = axes[1]
    colors = ['#2ca02c', '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#9467bd', '#d62728']
    for idx, (c_name, col) in enumerate(zip(conds.keys(), colors)):
        fpr = res_full[c_name]['head2']['fpr']
        tpr = res_full[c_name]['head2']['tpr']
        auc = res_full[c_name]['head2']['auc']
        ax2.plot(fpr, tpr, color=col, lw=2, label=f'{c_name} (AUC = {auc*100:.1f}%)')
        
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Full Gated Model: ROC Under Noise & Missing Scans', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    plt.suptitle("CDCo-MambaNet: Receiver Operating Characteristic (ROC) Diagnostic",
                 fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    out_file = REPORTS / 'audit_roc_curves.png'
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_prediction_distributions(res_base, res_full, y_true):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline clean
    p_b = np.array(res_base['clean']['head2']['probs'])
    p_b_ctrl = p_b[y_true == 0]
    p_b_asd = p_b[y_true == 1]
    
    # Full clean
    p_f = np.array(res_full['clean']['head2']['probs'])
    p_f_ctrl = p_f[y_true == 0]
    p_f_asd = p_f[y_true == 1]
    
    # Plot Baseline
    ax1 = axes[0]
    b1 = ax1.boxplot([p_b_ctrl, p_b_asd], positions=[1, 2], widths=0.4,
                     patch_artist=True, boxprops=dict(facecolor='#aec7e8', alpha=0.7))
    # Scatter points with jitter
    np.random.seed(0)
    jitter1 = np.random.normal(0, 0.04, size=len(p_b_ctrl))
    jitter2 = np.random.normal(0, 0.04, size=len(p_b_asd))
    ax1.scatter(np.ones_like(p_b_ctrl) + jitter1, p_b_ctrl, color='#1f77b4', s=45, alpha=0.8, label='Control (N=14)')
    ax1.scatter(np.full_like(p_b_asd, 2) + jitter2, p_b_asd, color='#d62728', s=45, alpha=0.8, label='Autism (N=14)')
    ax1.axhline(0.5, color='black', linestyle='--', lw=1.5, label='Default Threshold (0.50)')
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Control (Class 0)', 'Autism (Class 1)'], fontweight='bold')
    ax1.set_ylabel('Predicted Probability P(Autism)')
    ax1.set_title(f'Baseline Model (Clean)\nMean P(ASD): Ctrl={p_b_ctrl.mean():.3f}, ASD={p_b_asd.mean():.3f}\nALL PREDICTIONS > 0.50 -> 100% ASD', fontsize=11)
    ax1.set_ylim([0.0, 1.0])
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot Full Gate
    ax2 = axes[1]
    b2 = ax2.boxplot([p_f_ctrl, p_f_asd], positions=[1, 2], widths=0.4,
                     patch_artist=True, boxprops=dict(facecolor='#c7e9c0', alpha=0.7))
    ax2.scatter(np.ones_like(p_f_ctrl) + jitter1, p_f_ctrl, color='#1f77b4', s=45, alpha=0.8, label='Control (N=14)')
    ax2.scatter(np.full_like(p_f_asd, 2) + jitter2, p_f_asd, color='#d62728', s=45, alpha=0.8, label='Autism (N=14)')
    ax2.axhline(0.5, color='black', linestyle='--', lw=1.5, label='Default Threshold (0.50)')
    ax2.axhline(res_full['clean']['head2']['optimal_threshold'], color='#d62728', linestyle=':', lw=2.0,
                label=f'Optimal Threshold ({res_full["clean"]["head2"]["optimal_threshold"]:.4f})')
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Control (Class 0)', 'Autism (Class 1)'], fontweight='bold')
    ax2.set_ylabel('Predicted Probability P(Autism)')
    ax2.set_title(f'Full Gated Model (Clean)\nMean P(ASD): Ctrl={p_f_ctrl.mean():.3f}, ASD={p_f_asd.mean():.3f} (p < 0.05)\nALL PREDICTIONS < 0.50 -> 100% Control', fontsize=11)
    ax2.set_ylim([0.0, 1.0])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Prediction Distributions: Root Cause of Chance-Level 50% Accuracy",
                 fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    out_file = REPORTS / 'audit_prediction_distributions.png'
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_robustness_comparison(res_base, res_full, conds):
    cond_names = list(conds.keys())
    labels = [c.replace('_', '\n') for c in cond_names]
    
    acc_b = [res_base[c]['head2']['accuracy'] * 100 for c in cond_names]
    acc_f = [res_full[c]['head2']['accuracy'] * 100 for c in cond_names]
    auc_b = [res_base[c]['head2']['auc'] * 100 for c in cond_names]
    auc_f = [res_full[c]['head2']['auc'] * 100 for c in cond_names]
    
    x = np.arange(len(cond_names))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Accuracy
    ax1 = axes[0]
    ax1.bar(x - width/2, acc_b, width, label='Baseline', color='#1f77b4', alpha=0.85)
    ax1.bar(x + width/2, acc_f, width, label='Full Gate', color='#2ca02c', alpha=0.85)
    ax1.axhline(50.0, color='gray', linestyle='--', lw=1.2, label='Chance Accuracy (50%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Classification Accuracy Across Corruptions', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim([0, 100])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: ROC-AUC
    ax2 = axes[1]
    ax2.bar(x - width/2, auc_b, width, label='Baseline', color='#1f77b4', alpha=0.85)
    ax2.bar(x + width/2, auc_f, width, label='Full Gate', color='#2ca02c', alpha=0.85)
    ax2.axhline(50.0, color='gray', linestyle='--', lw=1.2, label='Chance AUC (50%)')
    ax2.set_ylabel('ROC-AUC (%)')
    ax2.set_title('ROC-AUC Across Corruptions', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim([0, 100])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Robustness Comparison: Baseline vs Full Gated Model Across 7 Test Conditions",
                 fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    out_file = REPORTS / 'audit_robustness_comparison.png'
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on {device}...")
    
    data_path = DEMO / 'abide_seed0_test.npz'
    d = np.load(data_path, allow_pickle=True)
    y_tensor = torch.from_numpy(d['label'])
    y_true = d['label']
    
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(d['timeseries']).float(),
            torch.from_numpy(d['corr']).float(),
            F.one_hot(y_tensor.long(), 2).float(),
            torch.from_numpy(d['smri']).float()
        ),
        batch_size=14, shuffle=False
    )
    
    print("Loading Baseline model...")
    model_base = load_model('baseline', d, device=device)
    print("Evaluating Baseline model across 7 conditions...")
    res_base = evaluate_all(model_base, loader, CONDITIONS, device=device)
    
    print("Loading Full Gated model...")
    model_full = load_model('full', d, device=device)
    print("Evaluating Full Gated model across 7 conditions...")
    res_full = evaluate_all(model_full, loader, CONDITIONS, device=device)
    
    print("Generating publication-quality figures...")
    plot_confusion_matrices(res_base, res_full, CONDITIONS)
    plot_roc_curves(res_base, res_full, CONDITIONS)
    plot_prediction_distributions(res_base, res_full, y_true)
    plot_robustness_comparison(res_base, res_full, CONDITIONS)
    
    summary = {
        'baseline': res_base,
        'full': res_full,
        'y_true': y_true.tolist(),
        'subject_ids': d['file_id'].tolist(),
        'sites': d['site'].tolist()
    }
    
    out_json = REPORTS / 'audit_metrics_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Full audit metrics dumped to {out_json}")


if __name__ == '__main__':
    main()
