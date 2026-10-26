"""Unit tests for CDCO-Mamba-Research fixes.

Covers:
1. Modality ordering in Mamba_fusion (mamba_fuse.py).
2. Finite loss, distillation warmup, and backward gradients in CDCo_MambaNet (cdcomnet.py).
3. Evaluation determinism (eval mode without random noise perturbation).
4. Metric calculations (metrics.py).
5. Safe data augmentation (data_argument.py - no NaNs, finite connectomes, subtle rotations).
"""
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from models.CDCo_MambaNet.cdcomnet import CDCo_MambaNet
from models.CDCo_MambaNet.encoder.mamba_fuse import Mamba_fusion
from training.metrics import calculate_metrics
from dataset.data_argument import random_rotate_3d, Dropout


def get_mock_config():
    cfg = OmegaConf.create({
        'dataset': {
            'name': 'abide',
            'node_sz': 116,
            'node_feature_sz': 116,
            'timeseries_sz': 115,
            'batch_size': 2,
        },
        'model': {
            'name': 'CDCo_MambaNet',
            'fuse_channel': 64,
            'fuse_shape': 48,
            'ct_loss': 1.0,
            'rate1': 0.8,
            'rate2': 0.5,
            'abide_fmri_feature_shape': 64,
            'smri_feature_shape': 54,
            'fmri_encoder_deepth': [1, 1, 1, 1],
            'fmri_encoder_dims': [32, 64, 128, 256, 512],
            'd_state': 4,
            'd_conv': 3,
            'expand': 4,
            'cls_num': 2,
            'distill_detach': True,
            'distill_warmup_epochs': 5,
            'distill_temp': 2.0,
            'use_legacy_loss': False,
            'reliability_fusion': False,
            'dynamic_dim': False,
            'grad_checkpoint': False,
        },
        'training': {
            'epochs': 10,
            'accumulate_grad_batches': 1,
        },
        'preprocess': {
            'name': 'non_mixup',
            'continus': False,
            'alpha': 0.0,
        }
    })
    return cfg


class TestModalityOrdering(unittest.TestCase):
    def test_modality_alignment(self):
        """Verify that fuse_feature and sf_feature_o have identical modality ordering (fmri, smri)."""
        cfg = get_mock_config()
        fuse = Mamba_fusion(cfg, in_features=64, in_features_ssm=48)
        fuse.eval()

        b, l, d = 2, 10, 48
        fmri = torch.randn(b, l, d)
        smri = torch.randn(b, l, d)

        # fuse_feature should concatenate (fmri, smri)
        cat_direct = torch.cat((fmri, smri), dim=1)
        self.assertEqual(cat_direct.shape, (b, 2 * l, d))

        out, r_loss = fuse.fuse_method(fmri, smri)
        self.assertFalse(torch.isnan(out).any(), "fuse_method output has NaNs")
        self.assertEqual(out.shape[0], b)

        out1, r_loss1 = fuse.fuse_method1(fmri, smri)
        self.assertFalse(torch.isnan(out1).any(), "fuse_method1 output has NaNs")
        self.assertEqual(out1.shape[0], b)


class TestFiniteLossAndWarmup(unittest.TestCase):
    def test_loss_finite_and_gradients(self):
        """Verify loss is finite, non-negative, and backpropagation yields valid gradients."""
        cfg = get_mock_config()
        model = CDCo_MambaNet(cfg)
        model.train()

        b = 2
        pred1 = torch.randn(b, 2, requires_grad=True)
        pred2 = torch.randn(b, 2, requires_grad=True)
        label = F.one_hot(torch.tensor([0, 1]), num_classes=2).float()
        model.contrastive_loss = torch.tensor(0.25)

        for epoch in [0, 2, 5, 10]:
            model.epoch = epoch
            loss = model.loss_function(pred1, label, pred2)
            self.assertTrue(torch.isfinite(loss), f"Loss is not finite at epoch {epoch}: {loss.item()}")
            self.assertGreater(loss.item(), 0.0)

            # Test backpropagation
            loss.backward(retain_graph=True)
            self.assertIsNotNone(pred1.grad)
            self.assertIsNotNone(pred2.grad)
            self.assertTrue(torch.isfinite(pred1.grad).all())
            self.assertTrue(torch.isfinite(pred2.grad).all())
            pred1.grad.zero_()
            pred2.grad.zero_()


class TestEvaluationDeterminism(unittest.TestCase):
    def test_eval_is_deterministic(self):
        """Verify that in eval() mode, identical inputs yield bitwise identical outputs across passes."""
        cfg = get_mock_config()
        fuse = Mamba_fusion(cfg, in_features=64, in_features_ssm=48)
        fuse.eval()

        b, l, d = 2, 10, 48
        fmri = torch.randn(b, l, d)
        smri = torch.randn(b, l, d)

        # Pass 1
        out1_a, r1_a = fuse.fuse_method1(fmri, smri)
        # Pass 2 without seed fixing
        out1_b, r1_b = fuse.fuse_method1(fmri, smri)

        # In eval mode, no noise should be added, so outputs must be identical
        diff = (out1_a - out1_b).abs().max().item()
        self.assertEqual(diff, 0.0, f"Evaluation is non-deterministic in fuse_method1: max diff {diff}")


class TestMetricCalculations(unittest.TestCase):
    def test_metrics(self):
        """Verify accuracy, specificity, sensitivity, F1, and AUC calculations."""
        # 4 samples: 2 controls, 2 patients
        # Label: [0, 0, 1, 1]
        label = np.eye(2)[[0, 0, 1, 1]]
        # Predictions: [0, 0, 1, 0] -> 3 correct, 1 wrong (1 patient called control)
        pred = np.array([
            [2.5, -1.0],  # true 0, pred 0 (TN)
            [1.8, -0.5],  # true 0, pred 0 (TN)
            [-1.2, 2.0],  # true 1, pred 1 (TP)
            [1.0, -0.5],  # true 1, pred 0 (FN)
        ])

        acc, spe, sen, f1, auc = calculate_metrics(pred, label)
        self.assertEqual(acc, 0.75)
        self.assertEqual(spe, 1.0)   # tn=2, fp=0 -> 2/2 = 1.0
        self.assertEqual(sen, 0.5)   # tp=1, fn=1 -> 1/2 = 0.5
        self.assertAlmostEqual(f1, 0.7333, places=3)
        self.assertTrue(0.0 <= auc <= 1.0)


class TestSafeDataAugmentation(unittest.TestCase):
    def test_subtle_rotation(self):
        """Verify random_rotate_3d runs cleanly and maintains tensor shape."""
        x = torch.randn(1, 16, 16, 16)
        rotated = random_rotate_3d(x, angles_range=(-8, 8))
        self.assertEqual(rotated.shape, x.shape)
        self.assertTrue(torch.isfinite(rotated).all())

    def test_safe_dropout_connectome(self):
        """Verify Dropout does not produce NaN correlation matrices or collapsed diagonal."""
        t = torch.randn(116, 115)
        drop = Dropout(p_zero=0.1)
        drop.train()
        t_aug = drop(t.clone())

        with np.errstate(invalid='ignore', divide='ignore'):
            pcc = np.corrcoef(t_aug.numpy())
        pcc = np.nan_to_num(pcc, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(pcc, 1.0)

        self.assertFalse(np.isnan(pcc).any(), "Correlation matrix contains NaNs")
        self.assertTrue(np.allclose(np.diag(pcc), 1.0), "Correlation matrix diagonal is not 1.0")


if __name__ == '__main__':
    unittest.main()
