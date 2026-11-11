"""
CDCo-MambaNet V2: Cross-Domain Collaborative Mamba Network with Connectome Graph Integration.

Preserves All Core Novelty Components:
1. RAF (Reliability-Aware Adaptive Fusion) across 3 modalities [sMRI, fMRI, FC].
2. DFD (Dynamic Feature Dimension) with Nested Dropout.
3. Multi-task Dirichlet Evidential Loss (EDL Weighting).
4. Head 1 Regularization (LayerNorm + Dropout).
5. Stop-gradient KL mutual distillation with classification warmup.
6. Contrastive perturbation loss.

Pillar 1 Innovation:
- Integrates the 116x116 Pearson Functional Connectivity (FC) matrix via BrainConnectomeGNN.
- Guarantees end-to-end gradient flow to the FC branch.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .encoder.mamba_fuse_v2 import Mamba_fusion_V2


class CDCo_MambaNet_V2(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        fuse_channel = config.model.get('fuse_channel', 512)
        fuse_shape = config.model.get('fuse_shape', 48)

        # Tri-modal Mamba fusion block
        self.fuse_encoder = Mamba_fusion_V2(config, in_features=fuse_channel, in_features_ssm=fuse_shape)
        self.contrastive_loss = None

        # Hyperparameters preserved from baseline
        self.rate1 = config.model.get('rate1', 0.8)
        self.rate2 = config.model.get('rate2', 0.5)
        self.ct_loss = config.model.get('ct_loss', 1.0)
        self.distill_detach = config.model.get('distill_detach', True)
        self.distill_warmup_epochs = config.model.get('distill_warmup_epochs', 5)
        self.distill_temp = config.model.get('distill_temp', 2.0)
        self.use_legacy_loss = config.model.get('use_legacy_loss', False)
        self.edl_weight = config.model.get('edl_weight', 0.5)
        self.edl_anneal_epochs = config.model.get('edl_anneal_epochs', 10)
        self.epoch = 0

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.0))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, fmri_feature, fmri_con, smri_feature, avail=None):
        """
        Args:
            fmri_feature: (B, 116, 115) fMRI BOLD timeseries
            fmri_con: (B, 116, 116) Functional Connectivity Pearson correlation matrix
            smri_feature: (B, 1, 96, 96, 96) sMRI volume
            avail: (B, 2) or (B, 3) bool tensor
        Returns:
            cls: (B, 2) Head 1 logits
            cls1: (B, 2) Head 2 logits
        """
        cls, cls1, contrastive_loss = self.fuse_encoder(smri_feature, fmri_feature, fmri_con, avail)
        self.contrastive_loss = contrastive_loss
        return cls, cls1

    def loss_function(self, pred1, label, pred2):
        label = label.float()

        # 1. Supervised Cross-Entropy on Ground Truth
        l1 = F.cross_entropy(pred1, label)
        l2 = F.cross_entropy(pred2, label)

        # 2. Temperature-scaled KL Mutual Distillation with Stop-Gradient
        t = self.distill_temp
        log_p1 = F.log_softmax(pred1 / t, dim=-1)
        log_p2 = F.log_softmax(pred2 / t, dim=-1)
        p1_soft = F.softmax(pred1 / t, dim=-1)
        p2_soft = F.softmax(pred2 / t, dim=-1)

        if self.distill_detach:
            p1_soft = p1_soft.detach()
            p2_soft = p2_soft.detach()

        l3 = F.kl_div(log_p1, p2_soft, reduction='batchmean') * (t ** 2)
        l4 = F.kl_div(log_p2, p1_soft, reduction='batchmean') * (t ** 2)

        if self.distill_warmup_epochs > 0:
            warmup_factor = min(1.0, float(self.epoch) / max(1, self.distill_warmup_epochs))
        else:
            warmup_factor = 1.0

        distill_w = (1.0 - self.rate1) * warmup_factor
        cls_w = 1.0 - distill_w

        loss = cls_w * (l1 + l2) + distill_w * (l3 + l4) + self.contrastive_loss * self.ct_loss

        # 3. Multi-task Dirichlet Evidential Loss on All Available Modalities
        gate = self.fuse_encoder.gate
        if gate is not None:
            kl_factor = min(1.0, float(self.epoch) / max(1, self.edl_anneal_epochs))
            loss = loss + self.edl_weight * gate.loss(label, kl_factor)

        return loss
