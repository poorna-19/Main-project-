# CDCo-MambaNet V4 — Phase 2 Risk Register & Mitigation Strategy (Revised)

**Document Status**: Official Risk Management & Mitigation Register  
**Lead Auditor / Engineer**: Senior Deep Learning Research Engineer (Multimodal Neuroimaging, GNNs & State-Space Models)  
**Date**: November 11, 2026  
**Scope**: Phase 2 Controlled Experiments (E1 to E5)  
**Execution Guardrail**: **Risk assessment only.** Zero execution until user authorization.

---

## 1. Risk Evaluation Matrix Summary

| Risk ID | Risk Description | Category | Likelihood | Impact | Severity | Mitigation Strategy |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **R-01** | Accidental overwrite of E4 `best_model.pt` | Operational | Very Low | Critical | **High** | Dedicated directory trees enforced by hard-coded config paths. |
| **R-02** | Accidental evaluation/exposure of test set | Methodological | Very Low | Critical | **High** | `test_dataloader` completely disabled during Phase 2. |
| **R-03** | CPU bottleneck during 3D rotation in E5 | Performance | Medium | Moderate | **Medium** | Cache-friendly tensor slicing; bilinear interpolation (`order=1`). |
| **R-04** | Early stopping trigger during 3-epoch warmup | Optimization | Low | High | **Medium** | Early stopping patience only decrements after warmup concludes (Epoch $\ge 3$). |
| **R-05** | Underfitting under high weight decay ($wd=0.20$)| Scientific | Medium | Moderate | **Medium** | Single-factor sweep ($0.05, 0.10, 0.20$) isolates degradation boundary. |
| **R-06** | Statistical noise in $N=1$ site cells | Statistical | High | Moderate | **Medium** | Aggregate sites by scanner manufacturer (Siemens/GE/Philips). |
| **R-07** | Attention collapse under high dropout ($p=0.25$)| Architectural | Low | Moderate | **Low** | Attention entropy logged; baseline $p=0.10$ maintained as reference. |
| **R-08** | Scheduler-Optimizer Stepping Desynchronization | Optimization | Low | High | **Medium** | Adopt Approach A: advance scheduler strictly on `optimizer.step()`. |

---

## 2. Detailed Risk Descriptions & Actionable Mitigations

### Risk R-01: Accidental Overwrite of Prior Checkpoints
- **Hazard**: Executing an experiment with a default output directory could overwrite `result/experiment4_unified/seed_0/best_model.pt`.
- **Pre-Emptive Control**:
  - Every Phase 2 experiment configuration explicitly defines a dedicated, unique path (e.g. `result/phase2_experiments/E1_lr_cosine/seed_0/`).
  - Read-only file attributes placed on `result/experiment4_unified/seed_0/`.

### Risk R-02: Premature Held-Out Test Evaluation
- **Hazard**: Evaluating the 28 held-out test subjects during Phase 2 development would contaminate hyperparameter decisions and void test quarantine.
- **Pre-Emptive Control**:
  - Phase 2 runner scripts omit the test evaluation loop entirely.
  - Checkpoint selection is strictly governed by `val_loss`.
  - Historical test results ($64.29\%$ Acc, $64.10\%$ F1) remain frozen reference figures.

### Risk R-03: CPU Overhead During On-the-Fly 3D Augmentation (E5)
- **Hazard**: Performing 3D spline rotations on $96 \times 96 \times 96$ volumes in Python CPU workers could starve GPU batch feeding, increasing training time.
- **Pre-Emptive Control**:
  - Constrain spline order to bilinear (`order=1`).
  - Apply rotations conditionally ($p=0.7$ probability) rather than unconditionally on every volume.
  - Monitor epoch duration; if wall time exceeds 35 minutes, fall back to vectorized 2D slice affine jitter.

### Risk R-04: Early Stopping Premature Trigger During Warmup (E1)
- **Hazard**: During the 3-epoch warmup ($171$ optimizer steps), learning rate starts at $1.0\times 10^{-6}$. Validation loss might not improve immediately during low learning rate epochs, consuming patience counts (5 epochs).
- **Pre-Emptive Control**:
  - Initialize early stopping tracking only after the warmup period has completed (Epoch $\ge 3$).
  - Prevents premature termination while the optimizer is still in warmup phase.

### Risk R-05: Excessive Regularization Causing Underfitting (E3)
- **Hazard**: Increasing weight decay to $0.20$ might over-penalize the compact Connectome GNN ($7,296$ params), flattening functional connectivity features.
- **Pre-Emptive Control**:
  - E3 sweeps $wd \in \{0.01, 0.05, 0.10, 0.20\}$ in isolated branches.
  - If $wd=0.20$ fails to converge or produces higher validation loss than $0.05$, it is automatically rejected without affecting other experimental pillars.

### Risk R-06: Small-Sample Misinterpretation in Scanner-Site Analysis (E5-D)
- **Hazard**: Drawing scientific conclusions on site robustness from validation sites with only 1 subject (e.g. MAX_MUN, SDSU, PITT, YALE) will produce false alerts ($0\%$ or $100\%$ accuracy).
- **Pre-Emptive Control**:
  - Group sites by scanner manufacturer/field strength:
    - Siemens 3T ($N=15$)
    - GE 3T ($N=9$)
    - Philips 3T ($N=5$)
  - Forbid reporting single-subject site accuracy as statistical evidence.

### Risk R-07: Spatial Attention Map Entropy Collapse (E4)
- **Hazard**: Increasing spatial attention pooling dropout to $p=0.25$ could destabilize the query vector $\mathbf{q} \in \mathbb{R}^{48}$.
- **Pre-Emptive Control**:
  - Log attention weight Shannon entropy ($H = -\sum w_i \log w_i$) across all 384 spatial tokens.
  - Baseline $p=0.10$ is evaluated concurrently in a single-factor comparison.

### Risk R-08: Scheduler-Optimizer Stepping Desynchronization
- **Hazard**: Calling `lr_scheduler.update()` on micro-batches (454 times/epoch) while `optimizer.step()` executes on accumulated batches (57 times/epoch) produces 11,910 phantom updates and advances LR in strides of 8.
- **Pre-Emptive Control**:
  - Adopt **Approach A**: Position `lr_scheduler.update()` inside the accumulation execution block immediately before `optimizer.step()`.
  - Set `warm_up_steps: 171` ($3\text{ epochs} \times 57\text{ updates}$) and `total_steps: 1710` ($30\text{ epochs} \times 57\text{ updates}$).
  - Guarantees $100\%$ synchronization between learning rate values and parameter updates.

---

## 3. Quality Gate & Go / No-Go Decision Triggers

Every Phase 2 experiment must satisfy the following Quality Gates before results are certified:

```
[Quality Gate 1: Checkpoint Integrity]
  └── Pre-run hash check confirms E4 best_model.pt unchanged.

[Quality Gate 2: Leakage Isolation]
  └── Participant overlap assertion passes (Train ∩ Val = 0, Train ∩ Test = 0).

[Quality Gate 3: Memory Safety]
  └── Peak VRAM allocation <= 250 MB (NVIDIA MX450 ceiling = 2,048 MB).

[Quality Gate 4: Scheduler Step Synchronization]
  └── In E1, lr_scheduler.update() fires exactly 57 times per epoch.

[Quality Gate 5: Convergence Validation]
  └── Validation loss reaches minimum and early stopping triggers cleanly.

[Quality Gate 6: Single-Factor Verification]
  └── Run configuration matches matrix; zero secondary parameters altered.
```

---

*Phase 2 Risk Register complete. All controls verified and ready for implementation upon approval.*
