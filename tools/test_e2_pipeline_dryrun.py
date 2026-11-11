"""
Quick Dry-Run Smoke Test for CDCo-MambaNet V2 Training Loop.
Runs 1 epoch with 2 batches on CUDA to guarantee zero runtime bugs.
"""
import os
os.environ["WANDB_MODE"] = "disabled"
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import dataset_factory
from models import model_factory
from components import optimizers_factory, lr_scheduler_factory, logger_factory
from training import training_factory

def dry_run():
    print("Running CDCo-MambaNet V2 Pipeline Dry-Run (2 batches)...")
    conf = OmegaConf.load(ROOT / "conf" / "config.yaml")
    conf.model = OmegaConf.load(ROOT / "conf" / "model" / "cdcomambanet_novelty.yaml")
    conf.dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "ABIDE.yaml")
    conf.training = OmegaConf.load(ROOT / "conf" / "training" / "exp2_novelty_training.yaml")
    conf.optimizer = OmegaConf.load(ROOT / "conf" / "optimizer" / "adamw_wd05.yaml")
    conf.datasz = OmegaConf.load(ROOT / "conf" / "datasz" / "100p.yaml")
    conf.preprocess = OmegaConf.load(ROOT / "conf" / "preprocess" / "non_mixup.yaml")

    with open_dict(conf):
        conf.seed = 0
        conf.repeat_time = 1
        conf.log_path = "result/smoke_test_v2"
        conf.unique_id = "test"
        conf.model.name = "CDCo_MambaNet_V2"
        conf.dataset.path = "E:/abide_data/abide_284.npy"
        conf.dataset.exact_splits_path = "result/phase6_multiseed/splits/exact_splits_seed0.json"
        conf.dataset.aug_cache = "E:/abide_data/train_aug_smri.npy"
        conf.dataset.batch_size = 2
        conf.dataset.val_batch_size = 2
        conf.dataset.test_batch_size = 2
        conf.training.epochs = 1
        conf.training.accumulate_grad_batches = 1
        conf.training.max_batches = 2
        conf.training.checkpoint_metric = "val_loss"

    logger = logger_factory(conf)
    dataloaders = dataset_factory(conf)
    model = model_factory(conf)
    optimizers = optimizers_factory(model=model, optimizer_configs=conf.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=conf.optimizer, cfg=conf)

    trainer = training_factory(conf, model, optimizers, lr_schedulers, dataloaders, logger)
    trainer.train()
    print("[SUCCESS] CDCo-MambaNet V2 Training Pipeline Dry-Run Passed with ZERO ERRORS!")

if __name__ == "__main__":
    dry_run()
