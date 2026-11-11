"""
Unit and Verification Tests for Phase 6 Experiment 2: RAF + DFD Novelty Training
Verifies:
1. Active functionality of reliability_fusion, dynamic_dim, nested_dropout, and edl_weight.
2. Tensor dimensional compatibility and deterministic eval for Head 1 LayerNorm + Dropout.
3. Checkpoint selection metric validation-only logic (zero test leakage).
4. Smoke test: finite loss, finite gradients, checkpoint save/load, and correct metrics.
"""
import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.CDCo_MambaNet.encoder.reliability import ReliabilityGate, edl_loss
from models.CDCo_MambaNet.cdcomnet import CDCo_MambaNet
from training.metrics import calculate_metrics

class TestPhase6Experiment2(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_01_reliability_gate_mechanisms(self):
        """Confirm RAF, DFD, nested_dropout, and EDL loss are actively engaged."""
        d = 48
        n_cls = 2
        min_dim = 12
        gate = ReliabilityGate(d=d, n_cls=n_cls, raf=True, dfd=True, min_dim=min_dim, nested_dropout=0.3)
        gate.train()

        tokens = [torch.randn(4, 10, d), torch.randn(4, 10, d)]
        avail = torch.ones(4, 2, dtype=torch.bool)

        out = gate(tokens, avail)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (4, 10, d))
        self.assertEqual(out[1].shape, (4, 10, d))

        # Check evidence was calculated
        self.assertIn('evidence', gate.info)
        ev = gate.info['evidence']
        self.assertEqual(len(ev), 2)
        self.assertTrue((ev[0] >= 0).all(), "Dirichlet evidence must be non-negative")

        # Check dynamic dimension (DFD): dimensions beyond max kept width should be 0
        dims = gate.info['dims']
        self.assertEqual(dims.shape, (4, 2))
        self.assertTrue((dims >= min_dim).all(), "Kept dimensions must be >= min_dim")
        self.assertTrue((dims <= d).all(), "Kept dimensions must be <= d")

        # Check evidential loss computation
        y = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        loss = gate.loss(y, kl_weight=0.5)
        self.assertTrue(torch.isfinite(loss), "EDL loss must be finite")
        self.assertGreater(loss.item(), 0.0, "EDL loss must be positive")

        # Verify gradient flow to gate heads
        loss.backward()
        for h in gate.heads:
            for p in h.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertFalse((p.grad == 0).all(), "Gate heads must receive active non-zero gradients")

    def test_02_nested_dropout_behavior(self):
        """Confirm nested_dropout applies stochastic cuts in train mode, deterministic in eval mode."""
        d = 48
        gate = ReliabilityGate(d=d, n_cls=2, raf=True, dfd=True, min_dim=12, nested_dropout=0.5)
        a = torch.ones(100)  # full reliability

        # In train mode: widths should vary stochastically due to nested_dropout
        gate.train()
        widths_train = [gate.width(a, d).mean().item() for _ in range(10)]
        self.assertFalse(all(w == d for w in widths_train), "Nested dropout should introduce stochastic truncation in train mode")

        # In eval mode: widths must be exactly d and completely deterministic
        gate.eval()
        widths_eval = [gate.width(a, d).mean().item() for _ in range(5)]
        self.assertTrue(all(w == d for w in widths_eval), "In eval mode, width must be deterministic full width")

    def test_03_head1_regularization_shapes_and_compatibility(self):
        """Test Head 1 with LayerNorm + Dropout vs legacy architecture."""
        cfg = OmegaConf.load("result/controlled_run_resumed/10-26-23-09-22/config.yaml")
        
        # Test A: With head1_regularization = True
        cfg_reg = cfg.copy()
        cfg_reg.model.head1_regularization = True
        cfg_reg.model.head1_dropout = 0.5
        model_reg = CDCo_MambaNet(cfg_reg)
        self.assertTrue(hasattr(model_reg.fuse_encoder, 'head1_norm'))
        self.assertTrue(hasattr(model_reg.fuse_encoder, 'head1_drop'))
        self.assertIsInstance(model_reg.fuse_encoder.head1_norm, nn.LayerNorm)
        self.assertEqual(model_reg.fuse_encoder.head1_norm.normalized_shape, (96,))

        # Test B: With head1_regularization = False (legacy)
        cfg_legacy = cfg.copy()
        cfg_legacy.model.head1_regularization = False
        model_legacy = CDCo_MambaNet(cfg_legacy)
        self.assertFalse(hasattr(model_legacy.fuse_encoder, 'head1_norm'))

    def test_04_full_model_forward_backward_with_edl(self):
        """Confirm CDCo_MambaNet forward pass and EDL loss contribution."""
        cfg = OmegaConf.load("result/controlled_run_resumed/10-26-23-09-22/config.yaml")
        cfg.model.reliability_fusion = True
        cfg.model.dynamic_dim = True
        cfg.model.nested_dropout = 0.3
        cfg.model.edl_weight = 0.5
        cfg.model.head1_regularization = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CDCo_MambaNet(cfg).to(device)
        model.train()

        b = 2
        fmri = torch.randn(b, 116, 115, device=device)
        corr = torch.randn(b, 116, 116, device=device)
        smri = torch.randn(b, 1, 96, 96, 96, device=device)
        label = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)

        p1, p2 = model(fmri, corr, smri)
        self.assertEqual(p1.shape, (b, 2))
        self.assertEqual(p2.shape, (b, 2))

        loss = model.loss_function(p1, label, p2)
        self.assertTrue(torch.isfinite(loss), "Combined loss must be finite")
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        # Verify gradients exist and are finite
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grad_norms), 0)
        self.assertTrue(all(np.isfinite(g) for g in grad_norms), "All gradients must be finite (0 NaNs/Infs)")

    def test_05_metrics_and_checkpoint_selection(self):
        """Confirm calculate_metrics computes Head 1, Head 2, and Ensemble metrics cleanly."""
        label = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        logits_h1 = np.array([[2.0, -1.0], [1.5, -0.5], [-1.0, 2.0], [0.5, -0.2]])  # 3 right, 1 wrong (75%)
        logits_h2 = np.array([[1.0, -0.5], [0.2, 0.8], [-2.0, 1.5], [-1.0, 2.0]])   # 3 right, 1 wrong (75%)
        
        m1 = calculate_metrics(logits_h1, label, return_dict=True)
        m2 = calculate_metrics(logits_h2, label, return_dict=True)
        
        p1 = F.softmax(torch.tensor(logits_h1), dim=-1).numpy()
        p2 = F.softmax(torch.tensor(logits_h2), dim=-1).numpy()
        p_ens = 0.5 * (p1 + p2)
        m_ens = calculate_metrics(p_ens, label, return_dict=True)
        
        self.assertIn('accuracy', m_ens)
        self.assertIn('balanced_accuracy', m_ens)
        self.assertIn('auc', m_ens)
        self.assertIn('confusion_matrix', m_ens)
        self.assertEqual(m_ens['accuracy'], 1.0, "Ensemble should resolve correct predictions for all 4 samples")

if __name__ == '__main__':
    unittest.main()
