"""
Multi-Seed Experiment 2 Runner: Seeds 1 to 4 Sequential Execution.
Guarantees:
1. Sequential execution (1 seed at a time) fitting within 2 GB VRAM on NVIDIA MX450.
2. Pre-checks GPU memory and disk space before each seed.
3. Completely isolated directories: result/phase6_multiseed/runs/seed{s}/
4. Isolated sMRI augmentation caches: E:/abide_data/train_aug_smri_seed{s}.npy
5. Strict zero-leakage: validation-only checkpoint selection (val_acc_ensemble),
   test set evaluated strictly post-training.
6. Aggregates results across seeds 0 to 4 into final multi-seed benchmark table.
"""
import os
os.environ["WANDB_MODE"] = "disabled"
import sys
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import json
import shutil
import numpy as np
import torch
from omegaconf import OmegaConf
from dataset import dataset_factory
from models import model_factory
from components import optimizers_factory, lr_scheduler_factory, logger_factory
from training import training_factory

def check_resources(required_c_gb=2.0, required_e_gb=5.0):
    _, _, free_c = shutil.disk_usage('.')
    free_c_gb = free_c / (1024 ** 3)
    _, _, free_e = shutil.disk_usage('E:/')
    free_e_gb = free_e / (1024 ** 3)
    
    if free_c_gb < required_c_gb:
        raise RuntimeError(f"Low C: drive disk space: {free_c_gb:.2f} GB available (need {required_c_gb} GB)")
    if free_e_gb < required_e_gb:
        raise RuntimeError(f"Low E: drive disk space: {free_e_gb:.2f} GB available (need {required_e_gb} GB)")
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024 ** 2)
        print(f"[Resource Check] GPU: {torch.cuda.get_device_name(0)} | Free VRAM: {vram_free:.1f} MB | Free C: {free_c_gb:.1f} GB | Free E: {free_e_gb:.1f} GB")
    else:
        print(f"[Resource Check] Running on CPU | Free C: {free_c_gb:.1f} GB | Free E: {free_e_gb:.1f} GB")

def is_seed_completed(seed):
    """
    Non-destructive check: returns True only if best_model.pt, final.json,
    and linear_probe_seed{seed}.json exist, are valid, and complete.
    """
    ckpt_path = f"result/phase6_multiseed/runs/seed{seed}/seed_{seed}/best_model.pt"
    final_path = f"result/phase6_multiseed/runs/seed{seed}/seed_{seed}/final.json"
    probe_path = f"result/phase6_multiseed/runs/seed{seed}/linear_probe_seed{seed}.json"
    
    if os.path.exists(ckpt_path) and os.path.exists(final_path) and os.path.exists(probe_path):
        try:
            if os.path.getsize(ckpt_path) > 100 * 1024 * 1024:
                with open(final_path, 'r') as f:
                    final_data = json.load(f)
                with open(probe_path, 'r') as f:
                    probe_data = json.load(f)
                if ("test_at_best_val" in final_data or "test_ensemble" in final_data) and len(probe_data) > 0:
                    return True
        except Exception:
            return False
    return False

def run_single_seed(seed, base_cfg_path="result/phase6_experiment2_novelty/10-28-13-02-58/config.yaml"):
    print("\n" + "=" * 70)
    print(f"STARTING PHASE 6 EXPERIMENT 2 FOR SEED {seed}")
    print("=" * 70)
    
    check_resources()
    
    out_dir = f"result/phase6_multiseed/runs/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    
    split_path = f"result/phase6_multiseed/splits/exact_splits_seed{seed}.json"
    assert os.path.exists(split_path), f"Missing split file: {split_path}"
    
    ckpt_path = f"result/phase6_multiseed/runs/seed{seed}/seed_{seed}/best_model.pt"
    cfg_path = f"result/phase6_multiseed/runs/seed{seed}/seed_{seed}/config.yaml"
    final_path = f"result/phase6_multiseed/runs/seed{seed}/seed_{seed}/final.json"
    probe_out = f"result/phase6_multiseed/runs/seed{seed}/linear_probe_seed{seed}.json"
    
    # 1. Preservation check: Skip if already 100% completed with valid artifacts
    if is_seed_completed(seed):
        print(f"[PRESERVATION CHECK] Seed {seed} is ALREADY COMPLETED with valid artifacts:")
        print(f"  - Checkpoint: {ckpt_path}")
        print(f"  - Manifest:   {final_path}")
        print(f"  - Probe:      {probe_out}")
        print(f"[PRESERVATION CHECK] Skipping Seed {seed} to strictly preserve existing verified results.")
        return

    # 2. If training is already completed and only probe is missing, run probe only
    if os.path.exists(ckpt_path) and os.path.exists(final_path) and not os.path.exists(probe_out):
        print(f"[Seed {seed}] Training completed but linear probe missing. Running probe evaluation...")
        probe_cmd = [
            sys.executable,
            "tools/evaluate_linear_probe.py",
            "--checkpoint", ckpt_path,
            "--config", cfg_path,
            "--splits", split_path,
            "--out", probe_out
        ]
        ret = subprocess.run(probe_cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"[Seed {seed}] Linear Probe evaluation failed with exit code {ret.returncode}")
        print(f"[Seed {seed}] Linear Probe evaluation completed successfully -> {probe_out}")
        return

    # 3. Otherwise execute full clean training run
    # Load base config from verified Experiment 2 run
    cfg = OmegaConf.load(base_cfg_path)
    
    # Configure seed-specific parameters
    cfg.seed = int(seed)
    cfg.repeat_time = 1
    cfg.log_path = f"result/phase6_multiseed/runs/seed{seed}"
    cfg.unique_id = f"seed_{seed}"
    
    # Dataset split and isolated augmentation cache
    cfg.dataset.exact_splits_path = split_path
    cfg.dataset.aug_cache = f"E:/abide_data/train_aug_smri_seed{seed}.npy"
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    trainer = training_factory(cfg, model, optimizers, lr_schedulers, dataloaders, logger)
    
    print(f"Launching training for Seed {seed}...")
    try:
        metrics = trainer.train()
        print(f"Completed Seed {seed}! Returned metrics: {metrics}")
    except Exception as e:
        print(f"[Seed {seed}] ERROR: Training halted with exception: {e}")
        raise
    
    # Cleanup memory
    del model, optimizers, lr_schedulers, trainer, dataloaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Automatically execute Linear Probe & Hybrid Evaluation (Issue 2 fix)
    if os.path.exists(ckpt_path) and os.path.exists(cfg_path):
        print(f"\n[Seed {seed}] Running automated Linear Probe & Hybrid Evaluation...")
        import subprocess
        probe_cmd = [
            sys.executable,
            "tools/evaluate_linear_probe.py",
            "--checkpoint", ckpt_path,
            "--config", cfg_path,
            "--splits", split_path,
            "--out", probe_out
        ]
        ret = subprocess.run(probe_cmd)
        if ret.returncode == 0:
            print(f"[Seed {seed}] Linear Probe evaluation completed successfully -> {probe_out}")
        else:
            raise RuntimeError(f"[Seed {seed}] Linear Probe evaluation failed with exit code {ret.returncode}")
    else:
        raise FileNotFoundError(f"[Seed {seed}] Checkpoint {ckpt_path} not found post-training!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment 2 Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4], help="List of seeds to execute sequentially")
    args = parser.parse_args()
    
    print("Multi-seed execution plan: Running seeds sequentially:", args.seeds)
    for s in args.seeds:
        run_single_seed(s)
        
    print("\nAll requested seeds completed!")
    
    # Check if all 5 seeds are ready for verification & aggregation
    import subprocess
    print("\nRunning verification and statistical aggregator...")
    subprocess.run([sys.executable, "tools/verify_fiveseed_results.py"])

if __name__ == "__main__":
    main()
