# CDCo-MambaNet V4 — Execution Commands Cheat Sheet

**Copy-pasteable PowerShell and Bash commands for fresh-system execution**

---

## 1. Environment Verification

### Hardware & CUDA Check
```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Run Model Architecture Unit Test (Safe & Fast)
```powershell
python tests/test_cdcomnet_v4.py
```

---

## 2. Dataset Verification (Read-Only)

### Verify Dataset Integrity & Tensor Shapes
```powershell
python tools/validate_abide_dataset.py
```

### Generate 4x sMRI Augmentation Cache (If Missing)
```powershell
python tools/generate_smri_cache.py
```

---

## 3. Immediate Held-Out Test Inference (No Training Needed)

### Run Baseline & Gate Evaluation on 28 Held-Out Subjects (CPU Mode)
```powershell
python tools/demo.py
```

### Run All 7 Scanner Noise / Missing Scan Conditions
```powershell
python tools/demo.py --all
```

---

## 4. Phase 2 Controlled Experiments (Pending User Authorization)

When you explicitly authorize training, execute experiments using their isolated configurations:

### Run E1: Scheduler Cadence Benchmark (57 updates/epoch, Monotonic Cosine Decay)
```powershell
python tools/run_e4_seed0_benchmark.py
```
*(Uses isolated config, freezes validation model selection, logs 57 updates/epoch).*

### Run Experiment with Custom Dataset Path
```powershell
python __main__.py model=cdcomambanet_v4_unified dataset=ABIDE_lowmem dataset.path="D:/abide_data/abide_284.npy" dataset.aug_cache="D:/abide_data/train_aug_smri.npy" dataset.exact_splits_path="splits/exact_splits_seed0.json" training.epochs=30
```

---

## 5. Resuming Training & Standalone Evaluation

### Resume Training from a Saved Checkpoint
```powershell
python __main__.py model=cdcomambanet_v4_unified dataset=ABIDE_lowmem training.resume_checkpoint="result/phase2_experiments/E1_scheduler/best_model.pt"
```

### Standalone Evaluation of Best Checkpoint on Held-Out Test Set
```powershell
python tools/evaluate_best_checkpoint.py --checkpoint "result/experiment4_unified/seed_0/best_model.pt" --config "result/experiment4_unified/seed_0/config.yaml"
```
*(Evaluates strictly once and outputs `test_evaluation.json`).*

---

## 6. Phase 1 Threshold Calibration (Post-Hoc Analysis)

To fit validation-governed decision thresholds without touching model weights:
```powershell
python tools/run_phase1_threshold_calibration.py
```
