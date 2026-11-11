# CDCo-MambaNet V4 — Phase 2 Experimental Design Specification

**Document Status**: Official Experimental Design Specification  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Reference Baseline**: Experiment E4 Seed 0 (Epoch 1 Best Checkpoint: Acc 64.29%, F1 64.10%, Sens 71.43%, AUC 62.24%)  
**Dataset**: ABIDE I Cohort ($N=284$ subjects), 17 Scanner Sites  
**Split**: Frozen Seed-0 Split (`exact_splits_seed0.json`: 227 Train, 29 Val, 28 Held-Out Test)  
**Execution Guardrail**: **Design preparation only.** Zero training, zero background jobs, zero file overwriting.

---

## 1. Pre-Finalization Verification & Methodological Audit

Before specifying the five controlled experiments, five critical implementation mechanisms were forensically verified:

### 1.1 Scheduler Step Frequency Verification
- **Code Location**: [`training/training.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/training/training.py) line 160:
  ```python
  self.current_step += 1
  lr_scheduler.update(optimizer=optimizer, step=self.current_step)
  ```
- **Finding**: `lr_scheduler.update()` is called **per mini-batch**, NOT per epoch.
- **Root Cause of Baseline Flaw**: In [`conf/optimizer/adamw_wd05.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/optimizer/adamw_wd05.yaml), `mode: cos_lr` computed:
  $$\text{period} = \max(1, \text{total\_steps} // 100) = 13,620 // 100 = 136\text{ steps}$$
  Because each epoch has $454$ mini-batches, the learning rate cycled up and down **$3.338$ times per epoch**, injecting aggressive gradient shocks and destabilizing AdamW second-order moments.
- **Verification of Remedy**: [`components/lr_scheduler.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/components/lr_scheduler.py) already contains a native, standard cosine implementation (`mode: cos`):
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T_{\max} - t_{\text{warmup}}}\pi\right)\right)$$
  Switching configuration to `mode: cos` with `warm_up_steps: 1362` ($3\text{ epochs} \times 454\text{ steps}$) yields a monotonic, leak-free macro-schedule.

### 1.2 Gradient Accumulation & Clipping Verification
- **Code Location**: [`training/training.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/training/training.py) lines 183–195:
  ```python
  loss_scaled = loss / accum_steps
  loss_scaled.backward()
  if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
      optimizer.step()
      optimizer.zero_grad()
  ```
- **Finding**:
  - Micro-batch size is $2$; `accum_steps = 8`; effective batch size is $16$.
  - With $454$ total batches per epoch, $454 \pmod 8 = 6$. The boundary condition `(batch_idx + 1) == num_batches` correctly triggers the optimizer on the final 6 accumulated batches, ensuring zero gradient loss.
  - **Absence of Clipping**: Currently, `torch.nn.utils.clip_grad_norm_` does not exist in the codebase.
- **Verification of Remedy**: Gradient norm clipping must be inserted immediately prior to `optimizer.step()`:
  ```python
  if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
      if self.config.training.get('clip_grad_norm', None):
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_grad_norm)
      optimizer.step()
      optimizer.zero_grad()
  ```
  This correctly clips the accumulated gradient norm across the effective batch of 16.

### 1.3 Normalization & Augmentation Leakage Check
- **Code Location**: [`dataset/abide.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/dataset/abide.py) lines 20–23:
  - Subject fMRI timeseries are z-score standardized per-subject and per-ROI:
    $$x_{s, r}(t) \leftarrow \frac{x_{s, r}(t) - \mu_{s, r}}{\sigma_{s, r}}$$
  - Pearson correlation matrices are computed strictly on each individual's timeseries.
  - sMRI volumes are mapped to MNI-152 space independently.
  - **Verdict**: Zero cross-subject information leakage in raw data normalization.
- **Augmentation Isolation**:
  - `data_aug` in `dataset/dataloader.py` is invoked only on training tensors (`final_timeseires_train, smri_train`).
  - Validation and test loaders draw exclusively from unaugmented arrays.
  - **Verdict**: Zero augmentation leakage into validation or test partitions.

### 1.4 Scanner-Site Small-Sample Limitations
- **Empirical Site Distribution**:
  - Training ($N=227$): 17 sites (NYU has 50; UM_1 has 30; UCLA_1 has 20; 14 sites have $\le 17$).
  - Validation ($N=29$): 11 sites (NYU: 7, UM_1: 4, UCLA_1: 4, UM_2: 4, LEUVEN_2: 3, KKI: 2, SDSU: 1, PITT: 1, YALE: 1, UCLA_2: 1, MAX_MUN: 1).
- **Small-Sample Warning**:
  - 6 out of 11 validation sites have only **1 single subject**. Disaggregating validation accuracy by individual site would produce binary values ($0\%$ or $100\%$) with extreme statistical noise.
- **Methodological Remedy**: Scanner-site analysis must group participants by scanner manufacturer/field strength (e.g. Siemens Allegra 3T vs GE Signa 3T vs Philips Achieva 3T) or compare High-Volume ($N \ge 4$: NYU, UM_1, UCLA_1, UM_2) vs Low-Volume ($N \le 2$), rather than reporting unrepresentative single-subject cells.

### 1.5 Validation-Only Model Selection Protocol
- Checkpoint selection is strictly governed by **minimum validation loss (`val_loss`)**:
  $$\theta^* = \arg\min_{\theta \in \{\theta_1, \dots, \theta_{30}\}} \mathcal{L}_{\text{val}}(\theta)$$
- Early stopping patience is fixed at **5 consecutive non-improving epochs**.
- Secondary tie-breaker (only if validation loss matches to 4 decimal places): Validation balanced accuracy $+ 0.01 \times \text{Validation AUC}$.
- **Test Quarantine**: The 28 held-out test participants are strictly quarantined and will never be evaluated during Phase 2 model development.

---

## 2. Controlled Experiment Specifications

In strict compliance with the single-factor requirement, each experiment alters **exactly one variable** relative to the verified E4 Seed-0 baseline.

---

### Experiment E1: Cosine Learning-Rate Scheduler Correction

#### 1. Scientific Hypothesis
Replacing the high-frequency pulsing `cos_lr` scheduler (3.33 complete cycles per epoch) with a smooth, monotonic Cosine Annealing schedule (`mode: cos`) with an epoch-aware linear warmup (3 epochs = 1,362 steps) will stabilize AdamW optimization, eliminate periodic gradient shocks, and prevent the optimizer from converging prematurely into sharp, non-generalizing training minima at Epoch 1.

#### 2. Exact Code Changes Required
No architectural code changes. In `components/lr_scheduler.py`, the existing implementation of `mode: cos` is utilized directly.

#### 3. Configuration Changes
Create [`conf/optimizer/adamw_e1_cosine.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/optimizer/adamw_e1_cosine.yaml):
```yaml
- name: AdamW
  lr: 1.0e-4
  match_rule: None
  except_rule: None
  no_weight_decay: false
  weight_decay: 0.05
  lr_scheduler:
    mode: cos                   # Replaces 'cos_lr' (eliminates 136-step pulsing)
    base_lr: 1.0e-4
    target_lr: 1.0e-6
    warm_up_from: 1.0e-6
    warm_up_steps: 1362         # 3 full epochs (3 * 454 steps) linear warmup
```

#### 4. Variables Held Constant
- Model: `CDCo_MambaNet_V4` (3,226,317 parameters).
- Split: Seed 0 (`exact_splits_seed0.json`).
- Weight decay: $0.05$.
- Spatial attention dropout: $0.10$.
- Head dropout: $0.50$.
- Gradient clipping: None.
- Augmentation: Static disk-cached (Baseline).
- Checkpoint metric: Strict minimum `val_loss`.

#### 5. Validation Metrics
Validation loss trajectory, validation accuracy, balanced accuracy, macro F1, sensitivity, specificity, and epoch of minimum validation loss.

#### 6. Expected Computational Cost
- Peak VRAM: $\sim 180.0\text{ MB}$.
- Wall time: $\sim 20$–$25$ minutes (depending on early stopping epoch).

#### 7. Output Directory
- Checkpoints & Logs: `result/phase2_experiments/E1_lr_cosine/seed_0/`
- Report: `reports/Phase2/E1_lr_cosine/`

#### 8. Reproducibility Requirements
Seed fixed to 0; deterministic CuDNN flags enabled; split verified by SHA-256 hash.

#### 9. Potential Scientific Limitations
Monotonic cosine annealing slows convergence during early epochs; if the effective learning rate remains too low during the warmup phase, representation learning in the heavy 3D ResNet-18 branch could be retarded.

---

### Experiment E2: Gradient Norm Clipping

#### 1. Scientific Hypothesis
Enforcing gradient norm clipping (`max_norm = 1.0`) across accumulated batches will bound the maximum gradient vector magnitude, preventing noisy 3D sMRI outlier scans or motion-corrupted fMRI frames from executing destructive parameter updates in the convolutional and Mamba state-space layers.

#### 2. Exact Code Changes Required
In `training/training.py`, add gradient clipping before `optimizer.step()`:
```python
if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
    clip_norm = self.config.training.get('clip_grad_norm', None)
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_norm))
    optimizer.step()
    optimizer.zero_grad()
```

#### 3. Configuration Changes
Create [`conf/training/e2_clip_training.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/training/e2_clip_training.yaml):
```yaml
name: Train
epochs: 30
accumulate_grad_batches: 8
clip_grad_norm: 1.0             # Enforces gradient norm threshold
early_stopping_patience: 5
checkpoint_metric: val_loss
```

#### 4. Variables Held Constant
- Model: `CDCo_MambaNet_V4` (3,226,317 parameters).
- Split: Seed 0 (`exact_splits_seed0.json`).
- Optimizer & Scheduler: Baseline (`adamw_wd05.yaml`).
- Weight decay: $0.05$.
- Spatial attention dropout: $0.10$.
- Head dropout: $0.50$.
- Augmentation: Static disk-cached (Baseline).
- Checkpoint metric: Strict minimum `val_loss`.

#### 5. Validation Metrics
Validation loss trajectory, gradient norm statistics, validation accuracy, balanced accuracy, macro F1, and early stopping epoch.

#### 6. Expected Computational Cost
- Peak VRAM: $\sim 180.2\text{ MB}$ (minimal overhead for norm computation).
- Wall time: $\sim 22$–$26$ minutes.

#### 7. Output Directory
- Checkpoints & Logs: `result/phase2_experiments/E2_grad_clip/seed_0/`
- Report: `reports/Phase2/E2_grad_clip/`

#### 8. Reproducibility Requirements
Exact gradient accumulation step frequency verified ($8$ steps, flush on batch $454$); exact norm $1.0$.

#### 9. Potential Scientific Limitations
If true gradient norms are already small (e.g. $< 1.0$), clipping will be inactive; if clipping is too aggressive, it could artificially scale down informative updates for under-represented classes.

---

### Experiment E3: Systematic Weight Decay Evaluation ($wd \in \{0.01, 0.05, 0.10\}$)

#### 1. Scientific Hypothesis
Because $64.49\%$ of the model parameters reside in the 3D ResNet-18 encoder, standard weight decay ($0.05$) is either too permissive (allowing excessive kernel specialization and memorization) or overly restrictive. Evaluating $wd \in \{0.01, 0.05, 0.10\}$ will identify the optimal $L_2$ regularization penalty on weight norms to maximize validation generalization.

#### 2. Exact Code Changes Required
No code changes. Driven entirely by Hydra configuration overrides.

#### 3. Configuration Changes
Three dedicated configurations:
- [`conf/optimizer/e3_adamw_wd01.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/optimizer/e3_adamw_wd01.yaml): `weight_decay: 0.01`
- [`conf/optimizer/e3_adamw_wd05.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/optimizer/e3_adamw_wd05.yaml): `weight_decay: 0.05` (Baseline reference)
- [`conf/optimizer/e3_adamw_wd10.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/optimizer/e3_adamw_wd10.yaml): `weight_decay: 0.10`

#### 4. Variables Held Constant
- Model: `CDCo_MambaNet_V4` (3,226,317 parameters).
- Split: Seed 0 (`exact_splits_seed0.json`).
- Learning rate: $1.0\times 10^{-4}$.
- Scheduler: Baseline (`cos_lr`).
- Spatial attention dropout: $0.10$.
- Head dropout: $0.50$.
- Gradient clipping: None.
- Augmentation: Static disk-cached (Baseline).
- Checkpoint metric: Strict minimum `val_loss`.

#### 5. Validation Metrics
Epoch-wise training loss vs. validation loss gap, validation accuracy, validation balanced accuracy, macro F1, and final parameter weight norms.

#### 6. Expected Computational Cost
- 3 independent controlled runs.
- Peak VRAM per run: $\sim 180.0\text{ MB}$.
- Total wall time: $\sim 60$–$75$ minutes ($3 \times \sim 22\text{ min}$).

#### 7. Output Directory
- Checkpoints & Logs:
  - `result/phase2_experiments/E3_weight_decay/wd_01/seed_0/`
  - `result/phase2_experiments/E3_weight_decay/wd_05/seed_0/`
  - `result/phase2_experiments/E3_weight_decay/wd_10/seed_0/`
- Report: `reports/Phase2/E3_weight_decay/`

#### 8. Reproducibility Requirements
Identical weight initialization across all three runs (Seed 0 initialization verified before optimizer instantiation).

#### 9. Potential Scientific Limitations
High weight decay ($0.10$) might indiscriminately penalize small weights in the compact Connectome GNN ($7.3\text{K}$ params), suppressing functional connectivity features while attempting to regularize 3D convolution kernels.

---

### Experiment E4: Spatial Attention Pooling Dropout Evaluation ($p \in \{0.10, 0.25\}$)

#### 1. Scientific Hypothesis
In the unified architecture, the learnable query vector $\mathbf{q} \in \mathbb{R}^{48}$ aggregates 384 spatial tokens into a single 48-dimensional vector. With low dropout ($p=0.10$), the query vector quickly overfits by attending to a tiny subset of high-contrast anatomical boundary tokens. Increasing attention map dropout to $p=0.25$ will force the query to distribute attention across diffuse, whole-brain spatial networks, improving validation generalization.

#### 2. Exact Code Changes Required
Ensure `spatial_attn_dropout` is dynamically mapped from Hydra configuration into `SpatialAttentionPooling` in `mamba_fuse_v4.py` (already supported via `config.model.get('spatial_attn_dropout', 0.1)`).

#### 3. Configuration Changes
Two model configurations:
- [`conf/model/e4_spatial_drop10.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/model/e4_spatial_drop10.yaml): `spatial_attn_dropout: 0.10` (Baseline reference)
- [`conf/model/e4_spatial_drop25.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/model/e4_spatial_drop25.yaml): `spatial_attn_dropout: 0.25`

#### 4. Variables Held Constant
- Model: `CDCo_MambaNet_V4` (3,226,317 parameters).
- Split: Seed 0 (`exact_splits_seed0.json`).
- Optimizer: Baseline (`adamw_wd05.yaml`, $wd=0.05$).
- Head dropout: $0.50$.
- Gradient clipping: None.
- Augmentation: Static disk-cached (Baseline).
- Checkpoint metric: Strict minimum `val_loss`.

#### 5. Validation Metrics
Validation loss trajectory, attention weight entropy ($H = -\sum w_i \log w_i$), validation accuracy, balanced accuracy, and macro F1.

#### 6. Expected Computational Cost
- 2 controlled runs.
- Peak VRAM per run: $\sim 180.0\text{ MB}$.
- Total wall time: $\sim 45$–$50$ minutes ($2 \times \sim 22\text{ min}$).

#### 7. Output Directory
- Checkpoints & Logs:
  - `result/phase2_experiments/E4_spatial_dropout/drop_10/seed_0/`
  - `result/phase2_experiments/E4_spatial_dropout/drop_25/seed_0/`
- Report: `reports/Phase2/E4_spatial_dropout/`

#### 8. Reproducibility Requirements
Identical random dropout generator seeds across runs; verify dropout is inactive during evaluation (`model.eval()`).

#### 9. Potential Scientific Limitations
Excessive attention dropout ($0.25$) could introduce noise during spatial token aggregation, potentially diluting localized neuroimaging focal markers (e.g. in amygdala or prefrontal cortex).

---

### Experiment E5: Training-Only Stochastic On-the-Fly Augmentation

#### 1. Scientific Hypothesis
The baseline 4x sMRI augmentation is pre-cached statically on disk, presenting the exact same 908 static volumes every epoch and allowing complete training memorization by Epoch 4 ($98.02\%$). Replacing static caching with **training-only on-the-fly randomized 3D rotations ($\pm 5^\circ$) and subtle continuous Gaussian noise injection ($\sigma=0.01$)** will provide true epoch-to-epoch variability, preventing memorization and maintaining a healthy training loss gradient.

#### 2. Exact Code Changes Required
In `dataset/dataloader.py`, create a dedicated `OnTheFlyAugmentedDataset` used exclusively by `train_dataloader`:
- During `__getitem__`:
  - 3D sMRI: Apply randomized Euler rotation $\theta_x, \theta_y, \theta_z \sim \mathcal{U}(-5^\circ, +5^\circ)$ on the fly.
  - Functional Connectivity: Apply random ROI edge dropout ($p=0.05$) to the Pearson matrix.
- `val_dataloader` and `test_dataloader` remain 100% deterministic, unaugmented, and clean.

#### 3. Configuration Changes
Create [`conf/dataset/e5_stochastic_aug.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/dataset/e5_stochastic_aug.yaml):
```yaml
name: ABIDE
stochastic_aug: true
aug_rotation_range: [-5.0, 5.0]
aug_fc_edge_dropout: 0.05
aug_smri_noise_sigma: 0.01
expand: false                   # Eliminates static disk-cached 4x pre-expansion
batch_size: 2
val_batch_size: 2
test_batch_size: 2
```

#### 4. Variables Held Constant
- Model: `CDCo_MambaNet_V4` (3,226,317 parameters).
- Split: Seed 0 (`exact_splits_seed0.json`).
- Optimizer: Baseline (`adamw_wd05.yaml`).
- Spatial attention dropout: $0.10$.
- Head dropout: $0.50$.
- Gradient clipping: None.
- Checkpoint metric: Strict minimum `val_loss`.

#### 5. Validation Metrics
Training accuracy trajectory (verifying that training accuracy does not artificially explode to $98\%$), validation loss trajectory, validation accuracy, balanced accuracy, and macro F1.

#### 6. Expected Computational Cost
- CPU rotation calculation on 3D volumes increases per-batch data preparation time slightly ($\sim 15\%$ increase in epoch wall time).
- Peak VRAM: $\sim 180.0\text{ MB}$.
- Total wall time: $\sim 28$–$32$ minutes.

#### 7. Output Directory
- Checkpoints & Logs: `result/phase2_experiments/E5_stochastic_aug/seed_0/`
- Report: `reports/Phase2/E5_stochastic_aug/`

#### 8. Reproducibility Requirements
Augmentation random generator seeded per worker with `worker_init_fn`; rotation range constrained to $\pm 5^\circ$ to prevent anatomical inversion.

#### 9. Potential Scientific Limitations
3D spline rotation on CPU can introduce slight interpolation blur into fine subcortical structures; rotation angles beyond $\pm 5^\circ$ could disrupt stereotaxic alignment with MNI-152 space.

---

## 3. Comparative Summary of Phase 2 Experiments

| Exp ID | Target Research Factor | Changed Parameter | Baseline Value | Experimental Value(s) | Primary Validation Goal |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **E1** | Learning-Rate Schedule | Scheduler Mode & Warmup | `cos_lr`, 5 steps | `cos`, 1,362 steps (3 eps) | Eliminate 3.33-cycles/epoch pulsing; smooth decay |
| **E2** | Gradient Stability | Gradient Norm Clipping | None (`null`) | `max_norm: 1.0` | Bound outlier gradient spikes from noisy scans |
| **E3** | Weight Regularization | AdamW Weight Decay | $0.05$ | $0.01$, $0.05$, $0.10$ | Determine optimal $L_2$ shrinkage on 3D convs |
| **E4** | Attention Pooling | Spatial Attention Dropout | $0.10$ | $0.10$, $0.25$ | Prevent query vector locking to fixed landmarks |
| **E5** | Data Regularization | Augmentation Mechanism | Static disk cache (4x) | On-the-fly stochastic ($\pm 5^\circ$) | Provide true epoch variability; halt memorization |

---

## 4. Methodological Compliance Declaration

1. **One Factor at a Time**: Each experiment E1 through E5 varies exactly one conceptual and configuration dimension relative to the verified baseline.
2. **Zero Test Data Access**: The held-out test cohort remains strictly quarantined. No test evaluation will be executed during Phase 2.
3. **Predeclared Selection**: All checkpoints will be selected exclusively by minimum validation loss (`val_loss`).
4. **Permanent E4 Protection**: No E4 checkpoint, report, or configuration file will be altered or overwritten.

---

*Phase 2 experimental design is complete and fully specified. Awaiting explicit user approval before modifying code or launching runs.*
