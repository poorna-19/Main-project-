import torch
import torch.nn as nn
from omegaconf import DictConfig

from .smri_encoder import LightweightResNet3D
from .fmri_transformer import TemporalTransformerEncoder
from .gnn_fc import BrainGNN
from .cross_attention import BidirectionalCrossAttention
from .evidential_head import CalibratedEvidentialHead, SiteDiscriminator, edl_loss

class CDCo_CrossAttentionNet(nn.Module):
    """
    CDCo-CrossAttentionNet: Lightweight Multimodal Cross-Attention Network
    for Autism Spectrum Disorder (ASD) Classification on ABIDE I.

    Key Innovations:
    1. Lightweight 3D ResNet-18 (<2.8M params) for sMRI volume.
    2. Temporal Transformer (<0.5M params) for fMRI BOLD dynamics.
    3. Brain Graph Attention Network (BrainGNN, <0.2M params) for Functional Connectivity.
    4. Bidirectional Morpho-Temporal Cross-Modal Attention.
    5. Single Unified Dirichlet Evidential Classifier (Zero Head-Discordance).
    6. Domain-Adversarial Gradient Reversal Layer (GRL) for 17-site invariance.

    Total Model Parameters: ~3.99M (vs 27.7M in CDCo-MambaNet, an 85.6% reduction to combat overfitting).
    Peak VRAM: ~1.31 GB (comfortably fits NVIDIA GeForce MX450 2,048 MB VRAM).
    """
    def __init__(self, config: DictConfig = None):
        super(CDCo_CrossAttentionNet, self).__init__()
        self.config = config

        d_model = 128
        num_rois = 116
        num_classes = 2
        num_sites = 17
        dropout = 0.25

        # 1. Compact Modality Encoders
        self.smri_encoder = LightweightResNet3D(
            in_channels=1, num_rois=num_rois, d_model=d_model, drop_prob=dropout
        )
        self.fmri_encoder = TemporalTransformerEncoder(
            num_rois=num_rois, time_steps=115, d_model=d_model, nhead=4, num_layers=3, dropout=dropout
        )
        self.gnn_encoder = BrainGNN(
            num_rois=num_rois, d_model=d_model, threshold=0.2, dropout=dropout
        )

        # 2. Bidirectional Morpho-Temporal Cross-Attention
        self.cross_attention = BidirectionalCrossAttention(
            d_model=d_model, nhead=4, dropout=dropout
        )

        # 3. Single Unified Evidential Classification Head
        self.classifier = CalibratedEvidentialHead(
            in_features=256, num_classes=num_classes, hidden_dim=128, dropout=dropout
        )

        # 4. Scanner-Site Invariant Adversarial Discriminator
        self.site_discriminator = SiteDiscriminator(
            in_features=256, num_sites=num_sites, hidden_dim=128, dropout=dropout
        )

    def forward(self, fmri_feature, fmri_con, smri_feature, grl_alpha=1.0):
        """
        Inputs:
            fmri_feature: (B, 116, 115) BOLD time series
            fmri_con:     (B, 116, 116) Pearson correlation matrix
            smri_feature: (B, 1, 96, 96, 96) 3D T1-weighted sMRI volume
            grl_alpha:    Float, domain adaptation gradient reversal scaling
        Returns:
            Dictionary containing evidential outputs, site logits, and latent representations.
        """
        # 1. Encode modalities into shared token space (B, 116, 128)
        h_smri = self.smri_encoder(smri_feature)
        h_fmri = self.fmri_encoder(fmri_feature)
        h_gnn  = self.gnn_encoder(fmri_con)

        # 2. Bidirectional Morpho-Temporal Cross-Attention Fusion
        z_subject, z_nodes = self.cross_attention(h_smri, h_fmri, h_gnn) # z_subject: (B, 256)

        # 3. Calibrated Evidential Classification
        cls_out = self.classifier(z_subject)

        # 4. Site Discriminator Output
        site_logits = self.site_discriminator(z_subject, alpha=grl_alpha)

        return {
            'logits': cls_out['logits'],
            'evidence': cls_out['evidence'],
            'alpha': cls_out['alpha'],
            'probs': cls_out['probs'],
            'uncertainty': cls_out['uncertainty'],
            'site_logits': site_logits,
            'latent': z_subject
        }

    def compute_loss(self, out, target, site_target=None, epoch=0, site_weight=0.1):
        """
        Computes composite loss: Evidential Dirichlet Cross-Entropy + Site Adversarial Loss.
        """
        loss_edl = edl_loss(out['alpha'], target, epoch=epoch)
        total_loss = loss_edl
        
        if site_target is not None and site_weight > 0:
            loss_site = nn.functional.cross_entropy(out['site_logits'], site_target)
            total_loss = total_loss + site_weight * loss_site
            
        return total_loss
