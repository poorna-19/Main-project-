import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pytest
from models.CDCo_CrossAttentionNet import CDCo_CrossAttentionNet

def test_cdco_crossattn_net_forward_cpu():
    # Strict CPU test: 0 GPU VRAM, zero interference with active GPU training task
    device = torch.device('cpu')
    model = CDCo_CrossAttentionNet().to(device)
    model.eval()

    # Parameter count verification
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Verification] Total Trainable Parameters: {num_params:,}")
    assert num_params < 4_500_000, f"Expected < 4.5M parameters, got {num_params:,}"

    # Dummy inputs matching ABIDE I specifications
    b = 2
    fmri_feat = torch.randn(b, 116, 115, device=device)
    fmri_con = torch.randn(b, 116, 116, device=device)
    # Ensure symmetric correlation matrix with 1s on diagonal
    fmri_con = (fmri_con + fmri_con.transpose(1, 2)) / 2.0
    for i in range(b):
        fmri_con[i].fill_diagonal_(1.0)
    smri_feat = torch.randn(b, 1, 96, 96, 96, device=device)

    # Forward pass
    with torch.no_grad():
        out = model(fmri_feat, fmri_con, smri_feat)

    # Assert outputs
    assert 'logits' in out
    assert 'probs' in out
    assert 'uncertainty' in out
    assert 'site_logits' in out
    assert 'latent' in out

    assert out['probs'].shape == (b, 2)
    assert out['uncertainty'].shape == (b,)
    assert out['site_logits'].shape == (b, 17)
    assert out['latent'].shape == (b, 256)

    # Verify probability normalization
    prob_sums = torch.sum(out['probs'], dim=-1)
    assert torch.allclose(prob_sums, torch.ones(b, device=device), atol=1e-5)

    # Verify loss computation
    targets = torch.tensor([1, 0], dtype=torch.long, device=device)
    site_targets = torch.tensor([0, 5], dtype=torch.long, device=device)
    loss = model.compute_loss(out, targets, site_targets, epoch=1)
    assert loss.item() > 0.0
    print("[Verification] Forward pass and loss verified successfully on CPU!")

if __name__ == "__main__":
    test_cdco_crossattn_net_forward_cpu()
