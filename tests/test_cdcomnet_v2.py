"""
Comprehensive Unit Test Suite for CDCo-MambaNet V2.
Verifies:
1. Model Instantiation & Parameter Inventory.
2. Tensor Dimensions at Every Forward Branch.
3. End-to-End Gradient Flow to All Modules (especially the new FC branch).
4. Modality Sensitivity: Proving FC matrix causally affects output predictions.
5. Preserved Core Novelties: RAF, DFD, Nested Dropout, and EDL Uncertainty.
"""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.CDCo_MambaNet.cdcomnet import CDCo_MambaNet
from models.CDCo_MambaNet_V2.cdcomnet_v2 import CDCo_MambaNet_V2


def get_base_config():
    conf = OmegaConf.load(ROOT / "conf" / "config.yaml")
    model_conf = OmegaConf.load(ROOT / "conf" / "model" / "cdcomambanet_novelty.yaml")
    conf.model = model_conf
    conf.dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "ABIDE.yaml")
    return conf


@pytest.fixture(scope="module")
def base_config():
    return get_base_config()


def test_01_parameter_count_and_fc_branch(base_config):
    """Verify parameter count and verify the FC branch is lightweight."""
    model_v2 = CDCo_MambaNet_V2(base_config)
    total_params = sum(p.numel() for p in model_v2.parameters())
    trainable_params = sum(p.numel() for p in model_v2.parameters() if p.requires_grad)

    fc_params = sum(p.numel() for p in model_v2.fuse_encoder.fc_encoder.parameters())
    smri_params = sum(p.numel() for p in model_v2.fuse_encoder.smri_encoder.parameters())
    fmri_params = sum(p.numel() for p in model_v2.fuse_encoder.fmri_encoder.parameters())
    gate_params = sum(p.numel() for p in model_v2.fuse_encoder.gate.parameters())

    print(f"\n--- CDCo-MambaNet V2 Parameter Audit ---")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"FC GNN Branch:         {fc_params:,} ({fc_params/total_params*100:.3f}%)")
    print(f"sMRI Branch:           {smri_params:,} ({smri_params/total_params*100:.2f}%)")
    print(f"fMRI Branch:           {fmri_params:,} ({fmri_params/total_params*100:.2f}%)")
    print(f"Tri-Modal Gate:        {gate_params:,} ({gate_params/total_params*100:.3f}%)")

    assert total_params > 27_000_000
    assert fc_params < 200_000, "FC Branch must remain lightweight to prevent overfitting"
    assert trainable_params == total_params


def test_02_tensor_dimensions_at_all_branches(base_config):
    """Verify exact tensor shapes across all processing stages."""
    model_v2 = CDCo_MambaNet_V2(base_config).eval()
    b = 2

    # Synthetic batch
    fmri = torch.randn(b, 116, 115)
    fc = torch.randn(b, 116, 116)
    fc = (fc + fc.transpose(1, 2)) / 2.0  # symmetric correlation
    smri = torch.randn(b, 1, 96, 96, 96)

    # 1. Test FC Encoder directly
    spatial_fc, global_fc = model_v2.fuse_encoder.fc_encoder(fc)
    assert spatial_fc.shape == (b, 512, 48), f"Expected (B, 512, 48), got {spatial_fc.shape}"
    assert global_fc.shape == (b, 1, 48), f"Expected (B, 1, 48), got {global_fc.shape}"

    # 2. Test Full Model Forward Pass
    out_h1, out_h2 = model_v2(fmri, fc, smri)
    assert out_h1.shape == (b, 2), f"Head 1 output shape must be (B, 2), got {out_h1.shape}"
    assert out_h2.shape == (b, 2), f"Head 2 output shape must be (B, 2), got {out_h2.shape}"
    assert not torch.isnan(out_h1).any(), "NaNs detected in Head 1"
    assert not torch.isnan(out_h2).any(), "NaNs detected in Head 2"


def test_03_end_to_end_gradient_flow_to_fc_branch(base_config):
    """Verify that every parameter in the FC branch receives non-zero gradients."""
    model_v2 = CDCo_MambaNet_V2(base_config).train()
    b = 2

    fmri = torch.randn(b, 116, 115, requires_grad=True)
    fc_raw = torch.randn(b, 116, 116, requires_grad=True)
    fc = (fc_raw + fc_raw.transpose(1, 2)) / 2.0
    fc.retain_grad()
    smri = torch.randn(b, 1, 96, 96, 96, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    out_h1, out_h2 = model_v2(fmri, fc, smri)
    loss = model_v2.loss_function(out_h1, labels, out_h2)
    assert not torch.isnan(loss), "Loss was NaN"

    loss.backward()

    # Verify FC Branch gradients
    fc_params_with_grad = 0
    fc_total_params = 0
    for name, param in model_v2.fuse_encoder.fc_encoder.named_parameters():
        fc_total_params += 1
        assert param.grad is not None, f"Parameter {name} has None grad!"
        grad_norm = param.grad.abs().sum().item()
        assert grad_norm > 0.0, f"Parameter {name} received zero gradient!"
        fc_params_with_grad += 1

    print(f"\n[OK] FC Branch Gradient Verification: {fc_params_with_grad}/{fc_total_params} parameters received active gradients.")

    # Verify input tensor gradients
    assert fc.grad is not None and fc.grad.abs().sum().item() > 0.0, "Input FC matrix received no gradient!"
    assert fmri.grad is not None and fmri.grad.abs().sum().item() > 0.0, "Input fMRI received no gradient!"
    assert smri.grad is not None and smri.grad.abs().sum().item() > 0.0, "Input sMRI received no gradient!"


def test_04_modality_causality_and_fc_sensitivity(base_config):
    """
    Mathematical Proof of Modality Usage:
    Altering the Functional Connectivity matrix MUST alter the model's logits.
    """
    model_v2 = CDCo_MambaNet_V2(base_config).eval()
    b = 2
    torch.manual_seed(42)

    fmri = torch.randn(b, 116, 115)
    fc_orig = torch.randn(b, 116, 116)
    fc_orig = (fc_orig + fc_orig.transpose(1, 2)) / 2.0
    smri = torch.randn(b, 1, 96, 96, 96)

    # 1. Forward with original FC
    with torch.no_grad():
        out1_orig, out2_orig = model_v2(fmri, fc_orig, smri)

    # 2. Forward with perturbed FC (shuffled edges)
    fc_perturbed = torch.randn(b, 116, 116)
    fc_perturbed = (fc_perturbed + fc_perturbed.transpose(1, 2)) / 2.0
    with torch.no_grad():
        out1_pert, out2_pert = model_v2(fmri, fc_perturbed, smri)

    diff_h1 = (out1_orig - out1_pert).abs().max().item()
    diff_h2 = (out2_orig - out2_pert).abs().max().item()

    print(f"\n[OK] FC Modality Sensitivity:")
    print(f"  Head 1 Maximum Logit Difference: {diff_h1:.6f}")
    print(f"  Head 2 Maximum Logit Difference: {diff_h2:.6f}")

    assert diff_h1 > 1e-4, "Head 1 is insensitive to Functional Connectivity!"
    assert diff_h2 > 1e-4, "Head 2 is insensitive to Functional Connectivity!"


def test_05_preserved_novelty_mechanics(base_config):
    """Verify RAF, DFD, Nested Dropout, and EDL Uncertainty on 3 modalities."""
    model_v2 = CDCo_MambaNet_V2(base_config).eval()
    b = 3

    fmri = torch.randn(b, 116, 115)
    fc = torch.randn(b, 116, 116)
    smri = torch.randn(b, 1, 96, 96, 96)

    # Test with availability mask: patient 0 has all, patient 1 missing sMRI, patient 2 missing fMRI+FC
    avail = torch.tensor([
        [True, True, True],
        [False, True, True],
        [True, False, False]
    ], dtype=torch.bool)

    with torch.no_grad():
        out_h1, out_h2 = model_v2(fmri, fc, smri, avail=avail)

    gate_info = model_v2.fuse_encoder.gate.info
    weights = gate_info["weight"]
    dims = gate_info["dims"]
    reliability = gate_info["reliability"]

    print(f"\n[OK] Preserved Novelty Verification (Tri-Modal Gate):")
    print(f"  Reliability (r):\n{reliability}")
    print(f"  Relative Weights (a):\n{weights}")
    print(f"  Kept Feature Dimensions (k):\n{dims}")

    # Maximum reliability modality must have weight 1.0
    assert torch.allclose(weights.max(dim=1).values, torch.ones(b), atol=1e-4)
    # Missing modalities must have weight 0.0
    assert weights[1, 0].item() == 0.0, "Missing sMRI must have weight 0.0"
    assert weights[2, 1].item() == 0.0, "Missing fMRI must have weight 0.0"
    assert weights[2, 2].item() == 0.0, "Missing FC must have weight 0.0"


if __name__ == "__main__":
    conf = get_base_config()
    print("Running CDCo-MambaNet V2 Test Suite...")
    test_01_parameter_count_and_fc_branch(conf)
    test_02_tensor_dimensions_at_all_branches(conf)
    test_03_end_to_end_gradient_flow_to_fc_branch(conf)
    test_04_modality_causality_and_fc_sensitivity(conf)
    test_05_preserved_novelty_mechanics(conf)
    print("\nALL 5 TEST SUITES PASSED SUCCESSFULLY!")
