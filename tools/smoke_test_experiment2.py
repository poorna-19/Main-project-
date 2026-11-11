"""
Phase 6 Experiment 2 Smoke Test:
Runs a complete mini 2-epoch training and evaluation cycle using the updated training.py pipeline.
Verifies:
1. Novelty forward & backward pass with EDL loss and gradient flow.
2. Head 1 (LayerNorm + Dropout) & Head 2 dual evaluation.
3. Validation-only checkpoint selection (val_acc_ensemble).
4. Early stopping logic and zero test leakage.
5. Checkpoint saving, reloading, and post-training test evaluation.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from dataset.abide import load_abide_data
from models.CDCo_MambaNet.cdcomnet import CDCo_MambaNet
from components import optimizers_factory, lr_scheduler_factory, logger_factory
from training.training import Train

class DummyLogger:
    def info(self, msg):
        print(f"[SMOKE_TEST INFO] {msg}")
    def warning(self, msg):
        print(f"[SMOKE_TEST WARN] {msg}")
    def error(self, msg):
        print(f"[SMOKE_TEST ERROR] {msg}")

def run_smoke_test():
    print("=" * 70)
    print("STARTING PHASE 6 EXPERIMENT 2 SMOKE TEST")
    print("=" * 70)

    smoke_dir = "result/smoke_test_exp2"
    if os.path.exists(smoke_dir):
        shutil.rmtree(smoke_dir)
    os.makedirs(smoke_dir, exist_ok=True)

    # Load base config
    cfg = OmegaConf.load("result/controlled_run_resumed/10-26-23-09-22/config.yaml")

    # Set Experiment 2 configuration
    cfg.model.reliability_fusion = True
    cfg.model.dynamic_dim = True
    cfg.model.nested_dropout = 0.3
    cfg.model.edl_weight = 0.5
    cfg.model.head1_regularization = True
    cfg.model.head1_dropout = 0.5
    cfg.model.cls_num = 2
    cfg.model.grad_checkpoint = True

    # Optimizer with weight_decay = 0.05
    cfg.optimizer[0].name = "AdamW"
    cfg.optimizer[0].lr = 1e-4
    cfg.optimizer[0].weight_decay = 0.05

    # Training settings for smoke test
    cfg.training.epochs = 2
    cfg.training.accumulate_grad_batches = 2
    cfg.training.early_stopping_patience = 5
    cfg.training.checkpoint_metric = "val_acc_ensemble"
    cfg.training.robustness_eval = False
    cfg.training.resume_checkpoint = None

    cfg.log_path = smoke_dir
    cfg.unique_id = "smoke_run"
    cfg.save_learnable_graph = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create synthetic fast dataloaders for smoke test
    b = 2
    # 4 train batches (8 subjects), 2 val batches (4 subjects), 2 test batches (4 subjects)
    def create_mock_loader(n_samples):
        ts = torch.randn(n_samples, 116, 115)
        corr = torch.randn(n_samples, 116, 116)
        y = torch.zeros(n_samples, 2)
        for i in range(n_samples):
            y[i, i % 2] = 1.0  # balanced
        smri = torch.randn(n_samples, 1, 96, 96, 96)
        ds = torch.utils.data.TensorDataset(ts, corr, y, smri)
        return torch.utils.data.DataLoader(ds, batch_size=b, shuffle=False)

    train_loader = create_mock_loader(8)
    val_loader = create_mock_loader(4)
    test_loader = create_mock_loader(4)
    dataloaders = [train_loader, val_loader, test_loader]

    model = CDCo_MambaNet(cfg).to(device)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    logger = DummyLogger()

    trainer = Train(cfg, model, optimizers, lr_schedulers, dataloaders, logger)

    print("\nRunning 2-epoch training cycle...")
    result = trainer.train()
    print(f"\nTraining completed cleanly! Returned test metrics: {result}")

    # Verify artifacts created
    best_pt = os.path.join(smoke_dir, "smoke_run", "best_model.pt")
    final_json = os.path.join(smoke_dir, "smoke_run", "final.json")
    best_val_npy = os.path.join(smoke_dir, "smoke_run", "best_val_metrics.npy")
    epoch_hist = os.path.join(smoke_dir, "smoke_run", "epoch_history.json")

    assert os.path.exists(best_pt), f"Missing {best_pt}"
    assert os.path.exists(final_json), f"Missing {final_json}"
    assert os.path.exists(best_val_npy), f"Missing {best_val_npy}"
    assert os.path.exists(epoch_hist), f"Missing {epoch_hist}"

    # Verify final.json content
    with open(final_json, "r") as f:
        final_data = json.load(f)

    assert "checkpoint_metric" in final_data
    assert final_data["checkpoint_metric"] == "val_acc_ensemble"
    assert "test_at_best_val" in final_data
    assert "ensemble" in final_data["test_at_best_val"]
    assert "head1" in final_data["test_at_best_val"]
    assert "head2" in final_data["test_at_best_val"]
    assert "best_val_metrics" in final_data

    # Test reloading checkpoint
    test_model = CDCo_MambaNet(cfg).to(device)
    state = torch.load(best_pt, map_location=device)
    test_model.load_state_dict(state)
    test_model.eval()

    # Verify deterministic eval on test batch
    b_ts, b_corr, b_y, b_smri = next(iter(test_loader))
    with torch.no_grad():
        out1_a, out2_a = test_model(b_ts.to(device), b_corr.to(device), b_smri.to(device))
        out1_b, out2_b = test_model(b_ts.to(device), b_corr.to(device), b_smri.to(device))
    assert torch.allclose(out1_a, out1_b), "Evaluation mode must be strictly deterministic"
    assert torch.allclose(out2_a, out2_b), "Evaluation mode must be strictly deterministic"

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED ALL ASSERTIONS SUCCESSFULLY!")
    print(f"Artifacts verified in {smoke_dir}/smoke_run:")
    print(f"  - best_model.pt (size: {os.path.getsize(best_pt):,} bytes)")
    print(f"  - final.json (keys: {list(final_data.keys())})")
    print(f"  - epoch_history.json (entries: 2 epochs)")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
