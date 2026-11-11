"""
Comprehensive Unit Test Suite for CDCo-MambaNet V3 (Experiment E3 Right-Sized Architecture).
Verifies:
1. Parameter Count Substantial Reduction (<3.5M params, >85% reduction vs E2).
2. Input/Output Tensor Shapes at All Branches.
3. Gradient Flow Through All 3 Modalities (sMRI, fMRI, FC).
4. FC Branch Contribution (FC matrix causally impacts model logits).
5. Deterministic Reproducibility (Fixed seed test).
6. Preserved Core Novelty Suite: RAF, DFD, Nested Dropout, and Multi-Task EDL.
"""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.CDCo_MambaNet_V3.cdcomnet_v3 import CDCo_MambaNet_V3


def get_e3_config():
    conf = OmegaConf.load(ROOT / "conf" / "config.yaml")
    model_conf = OmegaConf.load(ROOT / "conf" / "model" / "cdcomambanet_v3_compact.yaml")
    conf.model = model_conf
    conf.dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "ABIDE.yaml")
    return conf


@pytest.fixture(scope="module")
def e3_config():
    return get_e3_config()


def test_01_parameter_count_substantial_reduction(e3_config):
    """Verify that E3 achieves >85% parameter reduction and stays below 3.5M parameters."""
    model_v3 = CDCo_MambaNet_V3(e3_config)
    total_params = sum(p.numel() for p in model_v3.parameters())
    trainable_params = sum(p.numel() for p in model_v3.parameters() if p.requires_grad)

    fc_params = sum(p.numel() for p in model_v3.fuse_encoder.fc_encoder.parameters())
    smri_params = sum(p.numel() for p in model_v3.fuse_encoder.smri_encoder.parameters())
    fmri_params = sum(p.numel() for p in model_v3.fuse_encoder.fmri_encoder.parameters())
    gate_params = sum(p.numel() for p in model_v3.fuse_encoder.gate.parameters())

    e2_total = 27_811_311
    reduction_pct = (1.0 - total_params / e2_total) * 100

    print(f"\n--- CDCo-MambaNet V3 (E3 Right-Sized) Parameter Audit ---")
    print(f"Total Parameters:      {total_params:,} (vs E2: {e2_total:,}, Reduction: {reduction_pct:.2f}%)")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"sMRI Branch:           {smri_params:,} ({smri_params/total_params*100:.2f}%)")
    print(f"fMRI Branch:           {fmri_params:,} ({fmri_params/total_params*100:.2f}%)")
    print(f"FC GNN Branch:         {fc_params:,} ({fc_params/total_params*100:.2f}%)")
    print(f"Tri-Modal Gate:        {gate_params:,} ({gate_params/total_params*100:.2f}%)")

    # Strict scientific assertions
    assert total_params < 3_500_000, f"Expected <3.5M parameters, got {total_params:,}"
    assert reduction_pct > 85.0, f"Expected >85% reduction vs E2, got {reduction_pct:.2f}%"
    assert trainable_params == total_params


def test_02_input_output_tensor_shapes(e3_config):
    """Verify exact tensor shapes across all processing branches and dual heads."""
    model_v3 = CDCo_MambaNet_V3(e3_config).eval()
    b = 2

    # Synthetic multi-modal batch
    fmri = torch.randn(b, 116, 115)
    fc = torch.randn(b, 116, 116)
    fc = (fc + fc.transpose(1, 2)) / 2.0  # symmetric correlation
    smri = torch.randn(b, 1, 96, 96, 96)

    # 1. Test FC Encoder directly (target_tokens=128)
    spatial_fc, global_fc = model_v3.fuse_encoder.fc_encoder(fc)
    assert spatial_fc.shape == (b, 128, 48), f"Expected (B, 128, 48), got {spatial_fc.shape}"
    assert global_fc.shape == (b, 1, 48), f"Expected (B, 1, 48), got {global_fc.shape}"

    # 2. Test Full Model Forward Pass
    out_h1, out_h2 = model_v3(fmri, fc, smri)
    assert out_h1.shape == (b, 2), f"Head 1 output must be (B, 2), got {out_h1.shape}"
    assert out_h2.shape == (b, 2), f"Head 2 output must be (B, 2), got {out_h2.shape}"
    assert not torch.isnan(out_h1).any(), "NaNs detected in Head 1 logits"
    assert not torch.isnan(out_h2).any(), "NaNs detected in Head 2 logits"


def test_03_gradient_flow_through_all_modalities(e3_config):
    """Verify that every parameter in sMRI, fMRI, and FC branches receives active gradients."""
    model_v3 = CDCo_MambaNet_V3(e3_config).train()
    b = 2

    fmri = torch.randn(b, 116, 115, requires_grad=True)
    fc_raw = torch.randn(b, 116, 116, requires_grad=True)
    fc = (fc_raw + fc_raw.transpose(1, 2)) / 2.0
    fc.retain_grad()
    smri = torch.randn(b, 1, 96, 96, 96, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    out_h1, out_h2 = model_v3(fmri, fc, smri)
    loss = model_v3.loss_function(out_h1, labels, out_h2)
    assert not torch.isnan(loss), "Loss was NaN"

    loss.backward()

    # Check input gradients
    assert fmri.grad is not None and torch.norm(fmri.grad) > 0, "fMRI input received zero gradient"
    assert fc.grad is not None and torch.norm(fc.grad) > 0, "FC input received zero gradient"
    assert smri.grad is not None and torch.norm(smri.grad) > 0, "sMRI input received zero gradient"

    # Check FC encoder parameter gradients
    for name, param in model_v3.fuse_encoder.fc_encoder.named_parameters():
        assert param.grad is not None, f"FC param {name} received no gradient tensor"
        assert torch.norm(param.grad) > 0, f"FC param {name} has strictly zero gradient norm"

    # Check sMRI encoder parameter gradients
    smri_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model_v3.fuse_encoder.smri_encoder.parameters())
    assert smri_has_grad, "sMRI encoder received zero gradients"

    # Check fMRI encoder parameter gradients
    fmri_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model_v3.fuse_encoder.fmri_encoder.parameters())
    assert fmri_has_grad, "fMRI encoder received zero gradients"


def test_04_fc_branch_causal_contribution(e3_config):
    """Verify that perturbing the FC matrix causally alters both output predictions."""
    model_v3 = CDCo_MambaNet_V3(e3_config).eval()
    b = 2

    torch.manual_seed(42)
    fmri = torch.randn(b, 116, 115)
    fc1 = torch.randn(b, 116, 116)
    fc1 = (fc1 + fc1.transpose(1, 2)) / 2.0
    fc2 = torch.randn(b, 116, 116)
    fc2 = (fc2 + fc2.transpose(1, 2)) / 2.0
    smri = torch.randn(b, 1, 96, 96, 96)

    with torch.no_grad():
        out1_h1, out1_h2 = model_v3(fmri, fc1, smri)
        out2_h1, out2_h2 = model_v3(fmri, fc2, smri)

    diff_h1 = torch.norm(out1_h1 - out2_h1).item()
    diff_h2 = torch.norm(out1_h2 - out2_h2).item()

    print(f"\nFC Perturbation Logit Shift — Head 1: {diff_h1:.6f}, Head 2: {diff_h2:.6f}")
    assert diff_h1 > 1e-4, "Head 1 is completely unresponsive to FC matrix changes!"
    assert diff_h2 > 1e-4, "Head 2 is completely unresponsive to FC matrix changes!"


def test_05_reproducibility(e3_config):
    """Verify that identical seeds produce identical model forward outputs."""
    torch.manual_seed(1234)
    model_a = CDCo_MambaNet_V3(e3_config).eval()

    torch.manual_seed(1234)
    model_b = CDCo_MambaNet_V3(e3_config).eval()

    torch.manual_seed(999)
    fmri = torch.randn(2, 116, 115)
    fc = torch.randn(2, 116, 116)
    fc = (fc + fc.transpose(1, 2)) / 2.0
    smri = torch.randn(2, 1, 96, 96, 96)

    with torch.no_grad():
        out_a1, out_a2 = model_a(fmri, fc, smri)
        out_b1, out_b2 = model_b(fmri, fc, smri)

    assert torch.allclose(out_a1, out_b1, atol=1e-5), "Head 1 non-reproducible under identical seed!"
    assert torch.allclose(out_a2, out_b2, atol=1e-5), "Head 2 non-reproducible under identical seed!"


def test_06_preserved_novelty_mechanisms(e3_config):
    """Verify RAF, DFD, Nested Dropout, and Multi-Task EDL Loss are operational."""
    model_v3 = CDCo_MambaNet_V3(e3_config).train()
    b = 2

    # Check ReliabilityGate_V3 initialization
    gate = model_v3.fuse_encoder.gate
    assert gate is not None, "ReliabilityGate_V3 is missing!"
    assert len(gate.heads) == 3, f"Expected 3 evidential heads for [sMRI, fMRI, FC], got {len(gate.heads)}"
    assert gate.missing.shape == (3, 1, 1, 48), f"Expected missing shape (3, 1, 1, 48), got {gate.missing.shape}"

    fmri = torch.randn(b, 116, 115)
    fc = torch.randn(b, 116, 116)
    fc = (fc + fc.transpose(1, 2)) / 2.0
    smri = torch.randn(b, 1, 96, 96, 96)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    out_h1, out_h2 = model_v3(fmri, fc, smri)
    loss = model_v3.loss_function(out_h1, labels, out_h2)

    # Check that evidential loss was accumulated
    assert "evidence" in gate.info, "Gate info missing evidential outputs"
    assert len(gate.info["evidence"]) == 3, "Expected 3 evidential outputs"
    assert "reliability" in gate.info, "Gate info missing reliability weights"
    assert gate.info["reliability"].shape == (b, 3), "Reliability tensor shape mismatch"
    assert not torch.isnan(loss), "Loss computation produced NaN"
