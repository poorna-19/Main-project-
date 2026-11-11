"""
Multi-Seed Controlled Baseline Runner: Seeds 1 to 4 Sequential Execution.

Guarantees:
1. Exact historical baseline configuration:
   - model.reliability_fusion = False
   - model.dynamic_dim = False
   - model.nested_dropout = 0.0
   - model.edl_weight = 0.0 (Standard Cross-Entropy only)
   - model.head1_regularization = False
   - optimizer[0].weight_decay = 0.01
   - training.checkpoint_metric = 'val_acc_head1' (or 'val_accuracy')
   - training.early_stopping_patience = 12
   - training.epochs = 30
2. Completely isolated output directory:
   - result/phase6_controlled_baseline_multiseed/runs/seed{s}/
3. Preserves all Seed 0 and Seed 1 artifacts in result/controlled_run_resumed/
   and result/phase6_multiseed/.
4. Strictly evaluates matching exact split files:
   - result/phase6_multiseed/splits/exact_splits_seed{s}.json
5. Sequential execution fitting within 2 GB VRAM on NVIDIA GeForce MX450.
"""
import os
import sys
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

def run_single_baseline_seed(seed, base_cfg_path="result/controlled_run/10-26-14-51-19/config.yaml"):
    print("\n" + "=" * 75)
    print(f"STARTING PHASE 6 CONTROLLED BASELINE FOR SEED {seed}")
    print("=" * 75)
    
    check_resources()
    
    out_dir = f"result/phase6_controlled_baseline_multiseed/runs/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    
    split_path = f"result/phase6_multiseed/splits/exact_splits_seed{seed}.json"
    assert os.path.exists(split_path), f"Missing split file: {split_path}"
    
    # Load historical controlled baseline configuration
    cfg = OmegaConf.load(base_cfg_path)
    
    # Configure seed-specific parameters
    cfg.seed = int(seed)
    cfg.repeat_time = 1
    cfg.log_path = f"result/phase6_controlled_baseline_multiseed/runs/seed{seed}"
    cfg.unique_id = f"baseline_seed_{seed}"
    
    # Enforce strict baseline configuration flags
    cfg.model.reliability_fusion = False
    cfg.model.dynamic_dim = False
    cfg.model.nested_dropout = 0.0
    cfg.model.edl_weight = 0.0
    cfg.model.head1_regularization = False
    if 'head1_dropout' in cfg.model:
        cfg.model.head1_dropout = 0.0
    cfg.optimizer[0].weight_decay = 0.01
    cfg.training.checkpoint_metric = "val_acc_head1"
    cfg.training.early_stopping_patience = 12
    cfg.training.epochs = 30
    cfg.training.resume_checkpoint = None
    
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
    
    print(f"Launching controlled baseline training for Seed {seed}...")
    metrics = trainer.train()
    print(f"Completed Controlled Baseline Seed {seed}! Returned metrics: {metrics}")
    
    # Cleanup memory
    del model, optimizers, lr_schedulers, trainer, dataloaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Seed Controlled Baseline Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4], help="List of baseline seeds to execute sequentially")
    args = parser.parse_args()
    
    print("Controlled Baseline Multi-Seed Plan: Running seeds sequentially:", args.seeds)
    for s in args.seeds:
        run_single_baseline_seed(s)
        
    print("\nAll requested controlled baseline seeds completed successfully!")

if __name__ == "__main__":
    main()
