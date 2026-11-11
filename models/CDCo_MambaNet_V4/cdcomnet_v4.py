"""
CDCo-MambaNet V4: Unified Calibrated Evidential Cross-Domain Collaborative Mamba Network.

Core Innovations & Scientific Advancements (Experiment E4):
1. Unified Calibrated Evidential Head:
   Replaces the discordant dual-head architecture with a single unified head combining
   global class tokens and fine-grained spatial-topological tokens.
2. Spatial Multi-Head Attention Pooling:
   Dynamically aggregates 384 spatial tokens into an optimal 48-dimensional biomarker summary.
3. Removal of Dual-Head Mutual Distillation:
   Eliminates the KL divergence conflict noise between contradictory heads (62.07% validation discordance),
   restoring clean monotonic optimization.
4. Preserved Tri-Modal Novelty Suite:
   - 3D ResNet-18 (sMRI) + Compact ConvNeXtV2 (fMRI) + BrainConnectomeGNN (FC).
   - Reliability-Aware Adaptive Fusion (RAF) & Dynamic Feature Dimension (DFD).
   - Nested Dropout for dimension hierarchy.
   - Contrastive perturbation consistency loss.
   - Dirichlet Evidential Deep Learning (EDL) with annealed KL regularization.
5. Independent Validation-Governed Temperature Scaling:
   Post-hoc temperature scaling fitted strictly on validation data to eliminate overconfidence.
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .encoder.mamba_fuse_v4 import Mamba_fusion_V4
from .encoder.reliability_v4 import edl_loss
from .calibration import TemperatureScaler


class CDCo_MambaNet_V4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Architectural dimensions
        fuse_channel = config.model.get('fuse_channel', 128)
        fuse_shape = config.model.get('fuse_shape', 48)

        self.fuse_encoder = Mamba_fusion_V4(config, in_features=fuse_channel, in_features_ssm=fuse_shape)

        # Loss and training hyperparameters
        self.ct_loss = config.model.get('ct_loss', 1.0)
        self.edl_weight = config.model.get('edl_weight', 0.5)
        self.edl_anneal_epochs = config.model.get('edl_anneal_epochs', 10)
        self.epoch = 0

        # Post-hoc Temperature Scaler (Strictly validation-fitted post-training)
        self.temperature_scaler = TemperatureScaler(init_temp=1.5)

        # Internal state cache for evidential loss and attention inspection
        self.contrastive_loss = torch.tensor(0.0)
        self.last_evidence = None
        self.last_alpha = None
        self.last_probs = None
        self.last_uncertainty = None
        self.last_attn_weights = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.0))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, fmri_feature, fmri_con, smri_feature, avail=None, return_all=False):
        """
        Forward pass for CDCo-MambaNet V4.
        
        Args:
            fmri_feature: (B, 116, 115) fMRI BOLD timeseries
            fmri_con: (B, 116, 116) Functional Connectivity Pearson correlation matrix
            smri_feature: (B, 1, 96, 96, 96) sMRI volume
            avail: (B, 2) or (B, 3) bool tensor of modality availability
            return_all: If True, returns full evidential bundle;
                        If False (default), returns (logits, logits) for backward compatibility with trainers.
        """
        logits, evidence, alpha, probs, uncertainty, attn_weights, contrastive_loss = self.fuse_encoder(
            smri_feature, fmri_feature, fmri_con, avail
        )

        self.contrastive_loss = contrastive_loss
        self.last_evidence = evidence
        self.last_alpha = alpha
        self.last_probs = probs
        self.last_uncertainty = uncertainty
        self.last_attn_weights = attn_weights

        if return_all:
            return logits, evidence, alpha, probs, uncertainty, attn_weights
        # Return (logits, logits) for seamless compatibility with trainers expecting (p1, p2)
        return logits, logits

    def loss_function(self, pred1, label, pred2=None):
        """
        E4 Loss Function: Supervised Cross-Entropy + Dirichlet Evidential Loss + Contrastive Consistency.
        Dual-head mutual distillation is REMOVED to eliminate gradient conflict noise.
        """
        label = label.float()

        # 1. Supervised Cross-Entropy on ground truth with the unified head
        l_ce = F.cross_entropy(pred1, label)

        # 2. Dirichlet Evidential Loss on Unified Head Evidence
        kl_factor = min(1.0, float(self.epoch) / max(1, self.edl_anneal_epochs))
        if self.last_evidence is not None and self.edl_weight > 0:
            l_edl_head = edl_loss(self.last_evidence, label, kl_factor)
        else:
            l_edl_head = torch.tensor(0.0, device=pred1.device)

        # 3. Multi-Task Dirichlet Loss from Modality Reliability Gate (RAF)
        gate = self.fuse_encoder.gate
        if gate is not None and self.edl_weight > 0:
            l_edl_gate = gate.loss(label, kl_factor)
        else:
            l_edl_gate = torch.tensor(0.0, device=pred1.device)

        # 4. Contrastive Perturbation Consistency Loss
        l_ct = self.contrastive_loss * self.ct_loss

        # Total non-conflicting objective
        loss = l_ce + self.edl_weight * (l_edl_head + l_edl_gate) + l_ct
        return loss

    def calibrate(self, val_logits, val_labels):
        """
        Fit temperature on validation data only.
        NEVER pass test data to this function.
        """
        return self.temperature_scaler.fit(val_logits, val_labels)

    def predict_calibrated(self, logits):
        """Scale unnormalized logits using fitted validation temperature."""
        return self.temperature_scaler(logits)
