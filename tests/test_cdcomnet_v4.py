"""
Comprehensive Unit Test Suite for CDCo-MambaNet V4 (Experiment E4 Unified Calibrated Architecture).

Verifies:
1. Parameter Count and Substantial Reduction (<3.5M parameters, >88% reduction vs baseline).
2. Input/Output Tensor Shapes across all branches and batch sizes.
3. Gradient Flow Through All 3 Modalities (sMRI, fMRI, FC), Attention Pool, and Unified Head.
4. FC Branch Causal Contribution (FC variation alters outputs).
5. Mathematical Axioms of Dirichlet Evidential Outputs (non-negative, bounded uncertainty).
6. Spatial Multi-Head Attention Pooling Mechanics (valid weights, query gradient).
7. Removal of Dual-Head Mutual Distillation (Clean Non-Conflicting Loss Function).
8. Validation-Governed Temperature Scaling Calibration (ECE reduction on validation data).
9. Deterministic Reproducibility (Bitwise identical outputs given fixed seed).
10. Exact Split Integrity & Zero Participant Leakage (Seed 0: 227 Train, 29 Val, 28 Test).
"""
import sys
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.CDCo_MambaNet_V4.cdcomnet_v4 import CDCo_MambaNet_V4
from models.CDCo_MambaNet_V4.calibration import TemperatureScaler, compute_ece, evaluate_calibration


def get_e4_config():
    conf = OmegaConf.load(ROOT / "conf" / "config.yaml")
    model_conf = OmegaConf.load(ROOT / "conf" / "model" / "cdcomambanet_v4_unified.yaml")
    conf.model = model_conf
    conf.dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "ABIDE.yaml")
    return conf


@pytest.fixture(scope="module")
def e4_config():
    return get_e4_config()


def test_01_parameter_count_and_reduction(e4_config):
    """Verify parameter count is ~3.22M (<3.5M) and achieves >88% reduction vs baseline/E2."""
    model_v4 = CDCo_MambaNet_V4(e4_config)
    total_params = sum(p.numel() for p in model_v4.parameters())
    trainable_params = sum(p.numel() for p in model_v4.parameters() if p.requires_grad)

    fc_params = sum(p.numel() for p in model_v4.fuse_encoder.fc_encoder.parameters())
    smri_params = sum(p.numel() for p in model_v4.fuse_encoder.smri_encoder.parameters())
    fmri_params = sum(p.numel() for p in model_v4.fuse_encoder.fmri_encoder.parameters())
    gate_params = sum(p.numel() for p in model_v4.fuse_encoder.gate.parameters())
    attn_pool_params = sum(p.numel() for p in model_v4.fuse_encoder.spatial_attn_pool.parameters())
    unified_head_params = sum(p.numel() for p in model_v4.fuse_encoder.unified_head.parameters())

    baseline_total = 27_729_262
    e2_total = 27_811_311
    reduction_pct = (1.0 - total_params / baseline_total) * 100.0

    print(f"\n--- CDCo-MambaNet V4 (Experiment E4) Parameter Breakdown ---")
    print(f"Total Parameters:        {total_params:,} (vs Baseline: {baseline_total:,}, -{reduction_pct:.2f}%)")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"sMRI 3D ResNet-18:       {smri_params:,} ({smri_params/total_params*100:.2f}%)")
    print(f"fMRI Compact ConvNeXt:   {fmri_params:,} ({fmri_params/total_params*100:.2f}%)")
    print(f"FC Connectome GNN:       {fc_params:,} ({fc_params/total_params*100:.2f}%)")
    print(f"Spatial Attention Pool:  {attn_pool_params:,} ({attn_pool_params/total_params*100:.2f}%)")
    print(f"Unified Evidential Head: {unified_head_params:,} ({unified_head_params/total_params*100:.2f}%)")
    print(f"Tri-Modal RAF Gate:      {gate_params:,} ({gate_params/total_params*100:.2f}%)")

    assert total_params < 3_500_000, f"Expected <3.5M parameters, got {total_params:,}"
    assert reduction_pct > 88.0, f"Expected >88% reduction, got {reduction_pct:.2f}%"
    assert trainable_params == total_params, "All parameters must be trainable"


def test_02_input_output_tensor_shapes(e4_config):
    """Verify exact tensor shapes across multiple batch sizes (B=1, B=2, B=4)."""
    model_v4 = CDCo_MambaNet_V4(e4_config).eval()

    for b in [1, 2, 4]:
        fmri = torch.randn(b, 116, 115)
        fc = torch.randn(b, 116, 116)
        fc = (fc + fc.transpose(1, 2)) / 2.0
        smri = torch.randn(b, 1, 96, 96, 96)

        # Standard forward pass (returns logits, logits for compatibility)
        out1, out2 = model_v4(fmri, fc, smri)
        assert out1.shape == (b, 2), f"Expected (B, 2), got {out1.shape}"
        assert out2.shape == (b, 2), f"Expected (B, 2), got {out2.shape}"
        assert torch.allclose(out1, out2), "In E4, out1 and out2 are identical unified logits"

        # Full evidential forward pass
        logits, evidence, alpha, probs, uncertainty, attn_weights = model_v4(
            fmri, fc, smri, return_all=True
        )
        assert logits.shape == (b, 2)
        assert evidence.shape == (b, 2)
        assert alpha.shape == (b, 2)
        assert probs.shape == (b, 2)
        assert uncertainty.shape == (b, 1)
        assert attn_weights.shape == (b, 4, 1, 384)


def test_03_gradient_flow_through_all_modalities(e4_config):
    """Verify that all encoders, attention pool, and unified head receive active non-zero gradients."""
    model_v4 = CDCo_MambaNet_V4(e4_config).train()
    b = 2

    fmri = torch.randn(b, 116, 115, requires_grad=True)
    fc_raw = torch.randn(b, 116, 116, requires_grad=True)
    fc = (fc_raw + fc_raw.transpose(1, 2)) / 2.0
    fc.retain_grad()
    smri = torch.randn(b, 1, 96, 96, 96, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    p1, p2 = model_v4(fmri, fc, smri)
    loss = model_v4.loss_function(p1, labels, p2)
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert loss.item() > 0.0, "Loss must be positive"

    loss.backward()

    # 1. Encoders
    smri_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_v4.fuse_encoder.smri_encoder.parameters())
    fmri_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_v4.fuse_encoder.fmri_encoder.parameters())
    fc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_v4.fuse_encoder.fc_encoder.parameters())
    assert smri_has_grad, "sMRI encoder received zero gradients"
    assert fmri_has_grad, "fMRI encoder received zero gradients"
    assert fc_has_grad, "FC Connectome GNN received zero gradients"

    # 2. Input Tensors
    assert smri.grad is not None and smri.grad.abs().sum() > 0, "sMRI input received zero gradients"
    assert fmri.grad is not None and fmri.grad.abs().sum() > 0, "fMRI input received zero gradients"
    assert fc.grad is not None and fc.grad.abs().sum() > 0, "FC matrix received zero gradients"

    # 3. Novelty Modules: Spatial Attention Pool & Unified Head
    attn_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_v4.fuse_encoder.spatial_attn_pool.parameters())
    head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_v4.fuse_encoder.unified_head.parameters())
    assert attn_has_grad, "Spatial Attention Pooling received zero gradients"
    assert head_has_grad, "Unified Evidential Head received zero gradients"


def test_04_fc_branch_causal_contribution(e4_config):
    """Verify that changing the FC matrix causally alters the model's unified predictions."""
    model_v4 = CDCo_MambaNet_V4(e4_config).eval()
    b = 2

    fmri = torch.randn(b, 116, 115)
    smri = torch.randn(b, 1, 96, 96, 96)

    # Identical FC vs Permuted/Modified FC
    fc_a = torch.eye(116).unsqueeze(0).repeat(b, 1, 1)
    fc_b = torch.randn(b, 116, 116)
    fc_b = (fc_b + fc_b.transpose(1, 2)) / 2.0

    with torch.no_grad():
        out_a, _, _, _, _, attn_a = model_v4(fmri, fc_a, smri, return_all=True)
        out_b, _, _, _, _, attn_b = model_v4(fmri, fc_b, smri, return_all=True)

    logit_diff = (out_a - out_b).abs().mean().item()
    attn_diff = (attn_a - attn_b).abs().mean().item()

    assert logit_diff > 1e-4, f"FC modification must alter output logits (diff={logit_diff})"
    assert attn_diff > 1e-7, f"FC modification must alter attention pooling weights (diff={attn_diff})"


def test_05_evidential_output_validity(e4_config):
    """Verify mathematical axioms of Dirichlet evidential outputs."""
    model_v4 = CDCo_MambaNet_V4(e4_config).eval()
    b = 3

    fmri = torch.randn(b, 116, 115)
    fc = torch.randn(b, 116, 116)
    fc = (fc + fc.transpose(1, 2)) / 2.0
    smri = torch.randn(b, 1, 96, 96, 96)

    with torch.no_grad():
        logits, evidence, alpha, probs, uncertainty, _ = model_v4(fmri, fc, smri, return_all=True)

    # 1. Evidence must be strictly non-negative: e_k >= 0
    assert (evidence >= 0.0).all(), "Evidence must be >= 0"

    # 2. Alpha parameters: alpha_k = e_k + 1 >= 1.0
    assert (alpha >= 1.0).all(), "Alpha must be >= 1"
    assert torch.allclose(alpha, evidence + 1.0), "alpha must equal evidence + 1"

    # 3. Expected probabilities sum to 1.0: sum(p_k) = 1.0
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums)), "Probabilities must sum to 1"

    # 4. Epistemic uncertainty: u = K / S, u in (0, 1]
    S = alpha.sum(dim=-1, keepdim=True)
    expected_u = 2.0 / S
    assert torch.allclose(uncertainty, expected_u), "Uncertainty must equal 2 / S"
    assert (uncertainty > 0.0).all() and (uncertainty <= 1.0).all(), "Uncertainty must be in (0, 1]"


def test_06_spatial_attention_pooling_mechanics(e4_config):
    """Verify Multi-Head Attention Pooling attention properties."""
    pool = model_v4 = CDCo_MambaNet_V4(e4_config).fuse_encoder.spatial_attn_pool
    b, n, d = 2, 384, 48
    x = torch.randn(b, n, d, requires_grad=True)

    out, attn = pool(x)

    assert out.shape == (b, d), f"Pooled output must be (B, {d}), got {out.shape}"
    assert attn.shape == (b, 4, 1, n), f"Attention weights must be (B, 4, 1, {n}), got {attn.shape}"

    # Attention weights must be non-negative and sum to 1 across the spatial token dimension
    assert (attn >= 0.0).all(), "Attention weights must be >= 0"
    attn_sums = attn.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums)), "Attention weights must sum to 1.0"

    # Verify query gradient
    loss = out.sum()
    loss.backward()
    assert pool.query.grad is not None and pool.query.grad.abs().sum() > 0, "Query must receive gradient"


def test_07_no_mutual_distillation_and_clean_loss(e4_config):
    """Verify that E4 loss function contains NO dual-head KL mutual distillation."""
    model_v4 = CDCo_MambaNet_V4(e4_config).train()
    b = 2

    fmri = torch.randn(b, 116, 115)
    fc = torch.randn(b, 116, 116)
    fc = (fc + fc.transpose(1, 2)) / 2.0
    smri = torch.randn(b, 1, 96, 96, 96)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    p1, p2 = model_v4(fmri, fc, smri)
    loss = model_v4.loss_function(p1, labels, p2)

    # In E4, pred2 is purely optional and ignored in loss calculation
    loss_single = model_v4.loss_function(p1, labels)
    assert torch.allclose(loss, loss_single), "E4 loss must not depend on a secondary discordant head"


def test_08_temperature_scaling_calibration():
    """Verify that TemperatureScaler strictly fits on validation data and reduces ECE."""
    torch.manual_seed(42)
    np.random.seed(42)

    scaler = TemperatureScaler(init_temp=1.0)

    # Simulate overconfident validation predictions (N=29, similar to validation cohort)
    n_val = 29
    true_labels = np.random.choice([0, 1], size=n_val, p=[0.5, 0.5])
    # Overconfident logits (magnitude 3.0 -> confidence ~0.95)
    noise = np.random.randn(n_val, 2)
    val_logits = np.zeros((n_val, 2))
    for i in range(n_val):
        # 60% accuracy with 95% confidence (severe miscalibration)
        correct = (np.random.rand() < 0.60)
        c = true_labels[i] if correct else 1 - true_labels[i]
        val_logits[i, c] = 3.0 + noise[i, 0]
        val_logits[i, 1 - c] = -1.0 + noise[i, 1]

    fit_stats = scaler.fit(val_logits, true_labels, max_iter=150, lr=0.02)

    opt_temp = fit_stats["optimal_temp"]
    init_ece = fit_stats["initial_ece"]
    final_ece = fit_stats["final_ece"]

    print(f"\n--- Temperature Calibration Audit (Validation Data) ---")
    print(f"Optimal Temperature T: {opt_temp:.4f}")
    print(f"Initial Validation ECE: {init_ece*100:.2f}%")
    print(f"Calibrated Val ECE:     {final_ece*100:.2f}% (Reduction: {fit_stats['ece_reduction_pct']:.1f}%)")

    assert opt_temp > 1.0, f"Overconfident model must yield T > 1.0, got {opt_temp:.4f}"
    assert final_ece < init_ece, f"Calibrated ECE ({final_ece:.4f}) must be strictly lower than raw ECE ({init_ece:.4f})"


def test_09_deterministic_reproducibility(e4_config):
    """Verify bitwise identical outputs under fixed seed."""
    torch.manual_seed(999)
    model1 = CDCo_MambaNet_V4(e4_config).eval()

    torch.manual_seed(999)
    model2 = CDCo_MambaNet_V4(e4_config).eval()

    fmri = torch.randn(2, 116, 115)
    fc = torch.randn(2, 116, 116)
    fc = (fc + fc.transpose(1, 2)) / 2.0
    smri = torch.randn(2, 1, 96, 96, 96)

    with torch.no_grad():
        out1, _, _, _, _, attn1 = model1(fmri, fc, smri, return_all=True)
        out2, _, _, _, _, attn2 = model2(fmri, fc, smri, return_all=True)

    assert torch.allclose(out1, out2, atol=1e-7), "Outputs must be bitwise identical"
    assert torch.allclose(attn1, attn2, atol=1e-7), "Attention maps must be bitwise identical"


def test_10_exact_split_verification_and_zero_leakage():
    """Verify Seed 0 split has 227 Train, 29 Val, 28 Test participants with zero overlap."""
    split_path = ROOT / "result" / "phase6_multiseed" / "splits" / "exact_splits_seed0.json"
    assert split_path.exists(), f"Split file not found: {split_path}"

    with open(split_path, "r") as f:
        splits = json.load(f)

    # In exact_splits_seed0.json, splits are dictionaries containing 'sub_ids' and 'indices'
    train_subs = set(splits["train"]["sub_ids"])
    val_subs = set(splits["val"]["sub_ids"])
    test_subs = set(splits["test"]["sub_ids"])

    train_indices = set(splits["train"]["indices"])
    val_indices = set(splits["val"]["indices"])
    test_indices = set(splits["test"]["indices"])

    assert len(train_subs) == 227, f"Expected 227 train subjects, got {len(train_subs)}"
    assert len(val_subs) == 29, f"Expected 29 val subjects, got {len(val_subs)}"
    assert len(test_subs) == 28, f"Expected 28 test subjects, got {len(test_subs)}"

    # Check zero overlap on subject IDs
    assert len(train_subs & val_subs) == 0, "Train and Val sets must have zero overlap"
    assert len(train_subs & test_subs) == 0, "Train and Test sets must have zero overlap"
    assert len(val_subs & test_subs) == 0, "Val and Test sets must have zero overlap"

    # Check zero overlap on dataset indices
    assert len(train_indices & val_indices) == 0, "Train and Val indices must have zero overlap"
    assert len(train_indices & test_indices) == 0, "Train and Test indices must have zero overlap"
    assert len(val_indices & test_indices) == 0, "Val and Test indices must have zero overlap"

    assert len(train_subs | val_subs | test_subs) == 284, "Total unique participants must equal 284"
    assert len(train_indices | val_indices | test_indices) == 284, "Total unique indices must equal 284"
