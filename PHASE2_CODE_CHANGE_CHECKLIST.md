# CDCo-MambaNet V4 — Phase 2 Code Change Checklist & Implementation Blueprints

**Document Status**: Official Pre-Implementation Blueprint  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Execution Guardrail**: **Blueprint only.** Zero source code modified until user authorization.

---

## 1. Overview of Code Modifications Required for Phase 2

Phase 2 requires minimal, surgical modifications to only two existing source files:
1. `training/training.py`: To support gradient norm clipping (`clip_grad_norm`).
2. `dataset/dataloader.py`: To support on-the-fly stochastic data augmentation (for Experiment E5).

All other experiments (E1, E3, E4) require **zero Python code modifications** and are executed entirely through isolated Hydra configuration files.

---

## 2. Surgical Code Change Blueprints

### Change Item 1: Gradient Norm Clipping Support in `training/training.py`
- **Target File**: [`training/training.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/training/training.py)
- **Target Function**: `train_per_epoch(self, optimizer, lr_scheduler)`
- **Target Region**: Lines 193–196 (Immediately before `optimizer.step()`)
- **Affects Experiment**: **E2 (Gradient Clipping)**

#### Exact Code Diff Blueprint:
```python
<<<< PRE-CHANGE STATE (training/training.py, Lines 193-196):
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad()

==== POST-CHANGE STATE (training/training.py):
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                clip_norm = self.config.training.get('clip_grad_norm', None)
                if clip_norm is not None and float(clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_norm))
                optimizer.step()
                optimizer.zero_grad()
>>>>
```

#### Verification & Safety Test:
- When `clip_grad_norm: null` (Experiments E1, E3, E4, Baseline), the conditional evaluates to False; behavior is $100\%$ identical to baseline.
- When `clip_grad_norm: 1.0` (Experiment E2), gradient norms are capped at $1.0$ across the effective accumulated batch of 16.

---

### Change Item 2: Scheduler Stepping Alignment in `training/training.py`
- **Target File**: [`training/training.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/training/training.py)
- **Target Region**: Line 160 vs. Line 195
- **Affects Experiment**: **E1 (Cosine Scheduler)**

#### Analysis of Stepping Cadence:
- **Baseline State**: `lr_scheduler.update(optimizer=optimizer, step=self.current_step)` is called at every mini-batch ($454\text{ times per epoch}$).
- **Option A (Zero-Code Modification, Configuration Alignment)**:
  - Keep line 160 as-is.
  - Set `warm_up_steps: 1362` ($3\text{ epochs} \times 454\text{ mini-batches}$).
  - In `components/lr_scheduler.py`, `step` increments continuously from $1$ to $13,620$.
  - In `mode: cos`, cosine decay occurs smoothly across mini-batches with zero within-epoch cycling.
  - **Verdict**: **Recommended.** Requires zero code edits; eliminates implementation risk.

---

### Change Item 3: Stochastic On-the-Fly Augmentation in `dataset/dataloader.py`
- **Target File**: [`dataset/dataloader.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/dataset/dataloader.py)
- **Target Class**: Add `StochasticAugmentedDataset(utils.Dataset)` wrapper
- **Affects Experiment**: **E5 (Stochastic Augmentation)**

#### Exact Code Blueprint:
```python
class StochasticAugmentedDataset(utils.Dataset):
    """Training-only stochastic augmentation wrapper for 3D sMRI and Connectome FC."""
    def __init__(self, timeseries, pearson, labels, smri, rot_range=(-5.0, 5.0), edge_drop=0.05, noise_sigma=0.01):
        self.timeseries = timeseries
        self.pearson = pearson
        self.labels = labels
        self.smri = smri
        self.rot_range = rot_range
        self.edge_drop = edge_drop
        self.noise_sigma = noise_sigma

    def __len__(self):
        # 4x dynamic epoch length to match baseline gradient step budget (908 batches/epoch)
        return len(self.labels) * 4

    def __getitem__(self, idx):
        real_idx = idx % len(self.labels)
        t = self.timeseries[real_idx].clone()
        p = self.pearson[real_idx].clone()
        y = self.labels[real_idx].clone()
        s = self.smri[real_idx].clone()

        # Apply random Euler rotation (±5 degrees) on 3D volume
        if self.rot_range and random.random() > 0.3:
            s = random_rotate_3d(s, angles_range=self.rot_range)

        # Apply subtle Gaussian noise
        if self.noise_sigma > 0 and random.random() > 0.5:
            s = s + self.noise_sigma * torch.randn_like(s)

        # Apply mild connectome edge dropout (p=0.05)
        if self.edge_drop > 0 and random.random() > 0.5:
            mask = (torch.rand_like(p) > self.edge_drop).float()
            p = p * mask

        return t, p, y, s
```

#### Isolation Guarantee:
- Used exclusively by `train_dataloader`.
- `val_dataloader` and `test_dataloader` remain standard, deterministic `TensorDataset` objects.

---

## 3. Pre-Execution Verification Checklist

Before applying any code changes:

- [ ] Verify clean git working tree state: `git status --porcelain`.
- [ ] Create automated pre-change backup: `Copy-Item training/training.py training/training.py.bak`.
- [ ] Verify syntax validity using `python -m py_compile` prior to execution.
- [ ] Run dry-run unit test on single batch forward/backward.
- [ ] Verify that `result/experiment4_unified/seed_0/best_model.pt` checksum is unmodified.

---

*Phase 2 Code Change Checklist complete. No modifications will be applied until explicit approval is granted.*
