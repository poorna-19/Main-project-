# CDCo-MambaNet V4 — Phase 2 Critical Scheduler Cadence & Gradient Accumulation Audit

**Document Status**: Official Forensic Audit & Mathematical Resolution  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Subject**: Discrepancy between Mini-Batch Scheduler Steps and Optimizer Update Steps under Gradient Accumulation  
**Target Code**: `training/training.py` lines 151–196 and `components/lr_scheduler.py` lines 17–58  
**Execution Guardrail**: **Forensic audit only.** Zero training source code modified.

---

## 1. Forensic Code Trace of the Training Loop

The exact training loop in [`training/training.py`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/training/training.py) (lines 151–196) was inspected line by line:

```python
# training/training.py:
151: accum_steps = max(1, self.config.training.get('accumulate_grad_batches', 1)) # = 8
152: optimizer.zero_grad()
153: max_b = self.config.training.get('max_batches', None)
154: num_batches = len(self.train_dataloader) if not max_b else min(max_b, len(self.train_dataloader)) # = 454
155: 
156: for batch_idx, (fMRI_Feature, fMRI_Con, label, sMRI_Feature) in enumerate(self.train_dataloader):
157:     if max_b is not None and batch_idx >= max_b:
158:         break
159:     self.current_step += 1
160:     lr_scheduler.update(optimizer=optimizer, step=self.current_step)
         ...
183:     loss_scaled = loss / accum_steps
184:     loss_scaled.backward()
         ...
193:     if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
194:         optimizer.step()
195:         optimizer.zero_grad()
```

### Forensic Location of Critical Optimization Primitives:
1. `optimizer.zero_grad()`:
   - Line 152: Initialized once before the epoch loop begins.
   - Line 195: Called inside the accumulation condition immediately after `optimizer.step()`.
2. `loss_scaled.backward()`:
   - Line 184: Called on **every single mini-batch** ($454\text{ times per epoch}$), accumulating $\frac{1}{8}\nabla \mathcal{L}$ into `p.grad`.
3. `optimizer.step()`:
   - Line 194: Called **only when the accumulation condition is met** ($57\text{ times per epoch}$).
4. `lr_scheduler.update()`:
   - Line 160: Called on **every single mini-batch** ($454\text{ times per epoch}$), before the forward and backward passes.

---

## 2. Gradient Accumulation & Flushing Mathematics

### 2.1 Batch and Sample Counts
- Training cohort: $N_{\text{train}} = 227\text{ subjects}$.
- 4x static augmentation: $227 \times 4 = \mathbf{908\text{ training samples}}$.
- Micro-batch size: $b = 2$.
- Mini-batches per epoch: $N_{\text{batches}} = 908 \div 2 = \mathbf{454\text{ mini-batches}}$.
- Gradient accumulation factor: $A = 8$.
- Effective batch size: $B_{\text{eff}} = 2 \times 8 = \mathbf{16\text{ samples}}$.

### 2.2 Optimizer Update Step Count
- Full accumulation cycles per epoch:
  $$N_{\text{full}} = 454 // 8 = \mathbf{56\text{ cycles}} \quad (\text{covering batches } 1 \text{ to } 448)$$
- Remainder mini-batches:
  $$N_{\text{rem}} = 454 \pmod 8 = \mathbf{6\text{ mini-batches}} \quad (\text{covering batches } 449 \text{ to } 454)$$
- Flushing mechanism:
  At `batch_idx = 453` (the 454th mini-batch), `(batch_idx + 1) == num_batches` evaluates to `True`.
  The 57th `optimizer.step()` fires on the accumulated gradients of the final 6 batches (scaled by $1/8$, so effective weight is $6/8$).
- **Exact Optimizer Updates per Epoch**: Exactly **$57\text{ optimizer update steps}$**.
- **Exact Optimizer Updates across 30 Epochs**:
  $$N_{\text{opt\_updates}} = 57\text{ updates/epoch} \times 30\text{ epochs} = \mathbf{1,710\text{ actual optimizer updates}}$$

---

## 3. Discrepancy Analysis: Learning Rate Application Dynamics

Because `lr_scheduler.update()` is placed at Line 160 (inside the mini-batch loop) rather than at Line 194 (inside the optimizer step block), a critical question arises:

> **Which learning rate is applied to each optimizer update, and what happens to the intermediate learning rate values?**

### Micro-Batch Execution Trace (Example: Accumulation Cycle 1, Batches 0 to 7):

| Batch Index (`batch_idx`) | Micro-Batch Count | `self.current_step` | Scheduler Sets `lr` to: | Action Taken | Optimizer Called? |
| :---: | :---: | :---: | :---: | :--- | :---: |
| `0` | Batch 1 | $1$ | $\eta_1$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_1$ | **NO** |
| `1` | Batch 2 | $2$ | $\eta_2$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_2$ | **NO** |
| `2` | Batch 3 | $3$ | $\eta_3$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_3$ | **NO** |
| `3` | Batch 4 | $4$ | $\eta_4$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_4$ | **NO** |
| `4` | Batch 5 | $5$ | $\eta_5$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_5$ | **NO** |
| `5` | Batch 6 | $6$ | $\eta_6$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_6$ | **NO** |
| `6` | Batch 7 | $7$ | $\eta_7$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_7$ | **NO** |
| `7` | Batch 8 | $8$ | $\eta_8$ | `loss.backward()` accumulates $\frac{1}{8}\nabla \mathcal{L}_8$ | **YES (`optimizer.step()`)** |

### Mathematical Finding:
1. `backward()` computes *only* the Jacobian-loss product $\nabla_\theta \mathcal{L}_i$. PyTorch's `backward()` **does not read or use `param_group['lr']`**.
2. The optimizer step $\theta \leftarrow \theta - \eta \cdot \text{AdamW}(\sum \frac{1}{8}\nabla \mathcal{L}_i)$ reads `param_group['lr']` **only when `optimizer.step()` is executed**.
3. Therefore, the parameter update for the entire 16-sample effective batch uses **$\eta_8$** (the learning rate assigned on the final micro-batch of the accumulation window).
4. The values $\eta_1, \eta_2, \dots, \eta_7$ assigned during micro-batches $0$ through $6$ are **phantom values**: they overwrite `param_group['lr']` but are never read or applied to parameter weights.
5. In every accumulation cycle, **7 out of 8 scheduler updates ($87.5\%$) are completely unconsumed**.

---

## 4. Evaluation of the Two Decision Approaches

### Approach A: Optimizer-Update-Based Scheduler (Recommended)

- **Mechanics**:
  - `optimizer_step` counter increments *only* when `optimizer.step()` executes ($57\text{ times per epoch}$).
  - `lr_scheduler.update(optimizer, step=optimizer_step)` is called **immediately prior to `optimizer.step()`** (at Line 194).
  - Total scheduler steps across 30 epochs:
    $$\text{total\_steps} = 57\text{ updates/epoch} \times 30\text{ epochs} = \mathbf{1,710\text{ steps}}$$
  - Intended 3-epoch warmup:
    $$\text{warm\_up\_steps} = 3\text{ epochs} \times 57\text{ updates/epoch} = \mathbf{171\text{ optimizer update steps}}$$
- **Mathematical Properties**:
  - Every single calculated learning rate is directly applied to a parameter update.
  - Zero phantom scheduler steps ($100\%$ efficiency).
  - Step size $\eta_k$ is mathematically synchronous with the actual AdamW moment updates.
  - Invariant to changes in `accumulate_grad_batches` (if accumulation changes, the schedule remains defined over effective updates).

---

### Approach B: Mini-Batch-Based Scheduler (Baseline Convention)

- **Mechanics**:
  - `self.current_step` increments on *every micro-batch* ($454\text{ times per epoch}$).
  - `lr_scheduler.update(optimizer, step=self.current_step)` is called at Line 160.
  - Total scheduler steps across 30 epochs:
    $$\text{total\_steps} = 454\text{ batches/epoch} \times 30\text{ epochs} = \mathbf{13,620\text{ steps}}$$
  - Intended 3-epoch warmup:
    $$\text{warm\_up\_steps} = 3\text{ epochs} \times 454\text{ batches/epoch} = \mathbf{1,362\text{ mini-batch steps}}$$
- **Mathematical Properties**:
  - Out of 13,620 scheduler updates, **only 1,710 are read by `optimizer.step()`**; the other 11,910 updates are discarded.
  - The effective learning rate sequence sampled by the optimizer advances in discrete strides of 8:
    $$\{\eta_8, \eta_{16}, \eta_{24}, \dots, \eta_{448}, \eta_{454}, \eta_{462}, \dots\}$$
  - On the final batch of the epoch, the stride is 6 ($\eta_{454}$), creating a slight step irregularity.

---

## 5. Comparative Warmup Trajectory (First 3 Epochs)

Let $\eta_t$ be the learning rate during the 3-epoch warmup from $\eta_{\min} = 1.0\times 10^{-6}$ to $\eta_{\max} = 1.0\times 10^{-4}$:

$$\Delta \eta = 1.0\times 10^{-4} - 1.0\times 10^{-6} = 9.9\times 10^{-5}$$

| Parameter / Milestone | Approach A (Optimizer-Update Based) | Approach B (Mini-Batch Based) | Equivalence |
| :--- | :---: | :---: | :---: |
| **Total Warmup Duration** | Exactly 3.0 Epochs | Exactly 3.0 Epochs | **Identical physical duration** |
| **Warmup Parameter in Config** | `warm_up_steps: 171` | `warm_up_steps: 1362` | Scaled by accumulation factor $A \approx 8$ |
| **Effective Parameter Updates in Warmup** | 171 updates | 171 updates | **Identical update count** |
| **Learning Rate Increment per Update** | $\frac{\Delta \eta}{171} = 5.79\times 10^{-7}$ per update | $\frac{8 \times \Delta \eta}{1362} = 5.81\times 10^{-7}$ per update | **Equivalent within $0.3\%$** |
| **LR at End of Epoch 1** (Update 57 / Batch 454) | $\eta = 3.40\times 10^{-5}$ | $\eta = 3.40\times 10^{-5}$ | **Exact match ($34.0\%$ of max)** |
| **LR at End of Epoch 2** (Update 114 / Batch 908) | $\eta = 6.70\times 10^{-5}$ | $\eta = 6.70\times 10^{-5}$ | **Exact match ($67.0\%$ of max)** |
| **LR at End of Epoch 3** (Update 171 / Batch 1362)| $\eta = 1.00\times 10^{-4}$ | $\eta = 1.00\times 10^{-4}$ | **Exact match ($100.0\%$ of max)** |
| **Phantom Updates Computed** | **0 phantom steps** | **1,191 phantom steps** | Approach A is strictly cleaner |

---

## 6. Official Senior Engineering Recommendation

### Recommendation: **ADOPT APPROACH A (Optimizer-Update-Based Scheduler)**

#### Scientific Justification:
1. **Mathematical Consistency**: In optimization theory, step sizes $\eta_t$ parameterize parameter state transitions $\theta_{t+1} \leftarrow \theta_t - \eta_t \mathbf{u}_t$. Computing step sizes on micro-batches that do not produce state transitions violates the mathematical definition of a learning rate schedule.
2. **Elimination of Phantom Steps**: Approach A completely eliminates $11,910$ phantom scheduler calls across 30 epochs.
3. **Decoupling from Hardware Accumulation**: If hardware constraints require changing `accumulate_grad_batches` (e.g. from 8 to 4 on an 8 GB GPU or from 8 to 16 on a 1 GB GPU), Approach A preserves the exact effective learning rate schedule over $N_{\text{updates}}$, whereas Approach B artificially changes the schedule's speed by a factor of 2.
4. **Implementation Simplicity**: Moving `lr_scheduler.update()` from Line 160 to Line 194 in `training/training.py` requires changing exactly 2 lines of code.

---

## 7. Implementation Blueprint for Approach A

When authorized, the exact implementation in `training/training.py` will be:

```python
# In training/training.py, train_per_epoch():
# Remove line 160: lr_scheduler.update(optimizer=optimizer, step=self.current_step)

# Inside the accumulation step trigger (lines 193-196):
if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
    self.optimizer_step += 1
    lr_scheduler.update(optimizer=optimizer, step=self.optimizer_step)
    
    clip_norm = self.config.training.get('clip_grad_norm', None)
    if clip_norm is not None and float(clip_norm) > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_norm))
        
    optimizer.step()
    optimizer.zero_grad()
```

And in [`conf/optimizer/adamw_e1_cosine.yaml`](file:///c:/Users/Dell/Desktop/CDCO-Mamba-Research-main/conf/optimizer/adamw_e1_cosine.yaml):
```yaml
lr_scheduler:
  mode: cos
  base_lr: 1.0e-4
  target_lr: 1.0e-6
  warm_up_from: 1.0e-6
  warm_up_steps: 171            # Exactly 3 full epochs (3 * 57 optimizer updates)
```

And in `dataset/dataloader.py`:
```yaml
cfg.total_steps = 57 * cfg.training.epochs # = 1,710 total optimizer update steps
```

---

*Phase 2 Scheduler Cadence Audit complete. This specification provides full mathematical clarity and eliminates all ambiguity.*
