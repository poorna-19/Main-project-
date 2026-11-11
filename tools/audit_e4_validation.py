"""
Validation-Only Architecture Audit Tool for CDCo-MambaNet E4.
Evaluates Head 1, Head 2, and Ensemble dynamics on Validation Data Only (N=29).
Strictly prevents any held-out test data leakage or inspection.
"""
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["MPLCONFIGDIR"] = "result"
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, brier_score_loss, f1_score, recall_score, precision_score

ROOT = Path(".")
sys.path.append(str(ROOT))

from dataset import dataset_factory
from models import model_factory

conf = OmegaConf.load("conf/config.yaml")
conf.model = OmegaConf.load("conf/model/cdcomambanet_v3_compact.yaml")
conf.dataset = OmegaConf.load("conf/dataset/ABIDE.yaml")
conf.training = OmegaConf.load("conf/training/exp2_novelty_training.yaml")
conf.datasz = OmegaConf.load("conf/datasz/100p.yaml")
conf.preprocess = OmegaConf.load("conf/preprocess/non_mixup.yaml")
conf.dataset.path = "E:/abide_data/abide_284.npy"
conf.dataset.exact_splits_path = "result/phase6_multiseed/splits/exact_splits_seed0.json"
conf.dataset.aug_cache = "E:/abide_data/train_aug_smri.npy"
conf.dataset.batch_size = 2
conf.dataset.val_batch_size = 2
conf.dataset.test_batch_size = 2

dataloaders = dataset_factory(conf)
val_loader = dataloaders[1] # Validation loader strictly

ckpt = torch.load("result/experiment3_rightsized/seed_0/best_model.pt", map_location="cpu")
state_dict = ckpt["model"] if "model" in ckpt else ckpt
model = model_factory(conf)
model.load_state_dict(state_dict)
model.eval().cuda()

h1_logits, h2_logits, targets = [], [], []

with torch.no_grad():
    for fmri, corr, label, smri in val_loader:
        fmri = fmri.cuda()
        corr = corr.cuda()
        smri = smri.cuda()
        out1, out2 = model(fmri, corr, smri)
        h1_logits.append(out1.cpu())
        h2_logits.append(out2.cpu())
        y = torch.argmax(label, dim=-1).cpu() if label.dim() > 1 else label.cpu()
        targets.append(y)

h1_logits = torch.cat(h1_logits, dim=0) # (29, 2)
h2_logits = torch.cat(h2_logits, dim=0) # (29, 2)
targets = torch.cat(targets, dim=0)     # (29,)

p1 = F.softmax(h1_logits, dim=-1).numpy()
p2 = F.softmax(h2_logits, dim=-1).numpy()
p_ens = 0.5 * (p1 + p2)
y_true = targets.numpy()

def calc_ece(probs, labels, n_bins=5):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

print("=" * 80)
print("VALIDATION-ONLY ARCHITECTURE AUDIT (N=29 Validation Participants)")
print("Model Checkpoint: E3 Seed 0 (Epoch 2 best_model.pt)")
print("=" * 80)

# Metrics calculation
def get_metrics(probs):
    pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="macro")
    sen = recall_score(y_true, pred, pos_label=1, zero_division=0)
    spe = recall_score(y_true, pred, pos_label=0, zero_division=0)
    auc = roc_auc_score(y_true, probs[:, 1])
    brier = brier_score_loss(y_true, probs[:, 1])
    ece = calc_ece(probs, y_true)
    cm = confusion_matrix(y_true, pred)
    return acc, f1, sen, spe, auc, brier, ece, cm

acc1, f1_1, sen1, spe1, auc1, brier1, ece1, cm1 = get_metrics(p1)
acc2, f1_2, sen2, spe2, auc2, brier2, ece2, cm2 = get_metrics(p2)
acc_e, f1_e, sen_e, spe_e, auc_e, brier_e, ece_e, cm_e = get_metrics(p_ens)

print(f"Head 1 (Class Token):  Acc={acc1*100:.2f}%, F1={f1_1*100:.2f}%, Sen={sen1*100:.2f}%, Spe={spe1*100:.2f}%, AUC={auc1*100:.2f}%, ECE={ece1:.4f}, Brier={brier1:.4f}, CM={cm1.tolist()}")
print(f"Head 2 (Spatial MLP):  Acc={acc2*100:.2f}%, F1={f1_2*100:.2f}%, Sen={sen2*100:.2f}%, Spe={spe2*100:.2f}%, AUC={auc2*100:.2f}%, ECE={ece2:.4f}, Brier={brier2:.4f}, CM={cm2.tolist()}")
print(f"Ensemble (Average):    Acc={acc_e*100:.2f}%, F1={f1_e*100:.2f}%, Sen={sen_e*100:.2f}%, Spe={spe_e*100:.2f}%, AUC={auc_e*100:.2f}%, ECE={ece_e:.4f}, Brier={brier_e:.4f}, CM={cm_e.tolist()}")

# Confidence analysis
conf1 = np.max(p1, axis=1)
conf2 = np.max(p2, axis=1)
print(f"\nConfidence Diagnostics on Validation:")
print(f"Head 1 Mean Confidence: {np.mean(conf1):.4f} (Min: {np.min(conf1):.4f}, Max: {np.max(conf1):.4f}, Std: {np.std(conf1):.4f})")
print(f"Head 2 Mean Confidence: {np.mean(conf2):.4f} (Min: {np.min(conf2):.4f}, Max: {np.max(conf2):.4f}, Std: {np.std(conf2):.4f})")

# Inter-head discordance
pred1 = np.argmax(p1, axis=1)
pred2 = np.argmax(p2, axis=1)
agree_idx = np.where(pred1 == pred2)[0]
disagree_idx = np.where(pred1 != pred2)[0]
print(f"\nInter-Head Agreement: {len(agree_idx)}/29 ({len(agree_idx)/29*100:.2f}%)")
print(f"Inter-Head Disagreement: {len(disagree_idx)}/29 ({len(disagree_idx)/29*100:.2f}%)")

# Accuracy when heads agree vs disagree
if len(agree_idx) > 0:
    acc_agree = accuracy_score(y_true[agree_idx], pred1[agree_idx])
    print(f"Validation Accuracy when Heads Agree: {acc_agree*100:.2f}% ({int(acc_agree*len(agree_idx))}/{len(agree_idx)})")
if len(disagree_idx) > 0:
    print(f"When Heads Disagree: Head 1 correct: {np.sum(pred1[disagree_idx] == y_true[disagree_idx])}/{len(disagree_idx)} | Head 2 correct: {np.sum(pred2[disagree_idx] == y_true[disagree_idx])}/{len(disagree_idx)}")

# Temperature Scaling on Validation Logits
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.temperature

# Head 1 Temperature Scaling
ts1 = TemperatureScaler()
opt1 = torch.optim.LBFGS(ts1.parameters(), lr=0.01, max_iter=50)
def eval1():
    opt1.zero_grad()
    loss = F.cross_entropy(ts1(h1_logits), targets)
    loss.backward()
    return loss
opt1.step(eval1)
T1 = ts1.temperature.item()
p1_cal = F.softmax(h1_logits / T1, dim=-1).numpy()
acc1_c, f1_1_c, sen1_c, spe1_c, auc1_c, brier1_c, ece1_c, _ = get_metrics(p1_cal)

# Head 2 Temperature Scaling
ts2 = TemperatureScaler()
opt2 = torch.optim.LBFGS(ts2.parameters(), lr=0.01, max_iter=50)
def eval2():
    opt2.zero_grad()
    loss = F.cross_entropy(ts2(h2_logits), targets)
    loss.backward()
    return loss
opt2.step(eval2)
T2 = ts2.temperature.item()
p2_cal = F.softmax(h2_logits / T2, dim=-1).numpy()
acc2_c, f1_2_c, sen2_c, spe2_c, auc2_c, brier2_c, ece2_c, _ = get_metrics(p2_cal)

print(f"\nTemperature Scaling Calibration (Validation-Only Optimization):")
print(f"Head 1: Optimal T = {T1:.4f} | ECE: {ece1:.4f} -> {ece1_c:.4f} | Brier: {brier1:.4f} -> {brier1_c:.4f}")
print(f"Head 2: Optimal T = {T2:.4f} | ECE: {ece2:.4f} -> {ece2_c:.4f} | Brier: {brier2:.4f} -> {brier2_c:.4f}")
