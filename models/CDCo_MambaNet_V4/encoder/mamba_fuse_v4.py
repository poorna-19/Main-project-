"""
Tri-Modal Mamba Fusion Engine V4 for CDCo-MambaNet V4.
Unified Calibrated Evidential Cross-Domain Collaborative Mamba Network.

Core Architectural Innovations:
1. Preserves all 3 modalities: sMRI (3D ResNet-18) + fMRI (Compact ConvNeXtV2) + FC (BrainConnectomeGNN).
2. Preserves RAF (Reliability-Aware Adaptive Fusion), DFD (Dynamic Feature Dimension),
   Nested Dropout, and Contrastive Perturbation Regularization.
3. Single Unified Dual-Stream Aggregator:
   - Stream A: Fused Global Class Tokens [fMRI_cls, sMRI_cls, FC_cls] in R^144.
   - Stream B: Multi-Head Attention Pooling over 384 Spatial-Topological Tokens -> R^48.
   - Concatenated into Unified 192-dimensional representation z_unified in R^192.
4. Single Unified Evidential Classification Head:
   - LayerNorm(192) -> Linear(192, 96) -> GELU -> Dropout(0.5) -> Linear(96, 2)
   - Outputs unnormalized logits, non-negative Dirichlet evidence, Dirichlet alpha,
     predictive probabilities, and vacuous epistemic uncertainty.
5. Completely eliminates the dual-head discordance and mutual distillation conflict noise.
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint

from models.CDCo_MambaNet.encoder.smri_encoder import generate_model
from models.CDCo_MambaNet.encoder.fmri_encoder import ConvNeXtV2_Mamba
from models.CDCo_MambaNet.patch_ssm.mamba_vision import MambaVisionMixer_Fuse, MambaVisionMixer_Cls
from .fc_encoder import BrainConnectomeGNN
from .reliability_v4 import ReliabilityGate_V4


def checkpoint_blocks(blocks):
    """Recompute each block's activations in backward pass to fit comfortably in VRAM."""
    for blk in blocks:
        fwd = blk.forward
        blk.forward = lambda x, fwd=fwd, blk=blk: (
            checkpoint(fwd, x, use_reentrant=False) if blk.training and torch.is_grad_enabled() else fwd(x))
        for bn in blk.modules():
            if isinstance(bn, nn.modules.batchnorm._BatchNorm):
                bn.momentum = 1 - (1 - bn.momentum) ** 0.5


class ContrastiveLossModel(nn.Module):
    """Contrastive perturbation consistency loss."""
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.1)

    def contrastive_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels)

    def sf_loss(self, feat_orig, feat_perturbed):
        b = feat_orig.shape[0]
        orig_embeds = F.normalize(feat_orig.reshape(b, -1), dim=-1)
        pert_embeds = F.normalize(feat_perturbed.reshape(b, -1), dim=-1)
        logit_scale = self.logit_scale.exp()
        sim_matrix = torch.matmul(orig_embeds, pert_embeds.t()) * logit_scale
        labels = torch.arange(b, device=feat_orig.device)
        loss = (self.contrastive_loss(sim_matrix, labels) + self.contrastive_loss(sim_matrix.t(), labels)) / 2.0
        return loss


class MLPHead_V3(nn.Module):
    """Projection head for contrastive perturbation learning."""
    def __init__(self, in_channels, projection_size):
        super().__init__()
        mlp_hidden_size = in_channels * 2

        self.l1 = nn.Linear(in_channels, mlp_hidden_size)
        self.bn1 = nn.LayerNorm(mlp_hidden_size)
        self.act1 = nn.GELU()
        self.ln2 = nn.Linear(mlp_hidden_size, in_channels)
        self.bn2 = nn.LayerNorm(in_channels)
        self.act2 = nn.GELU()
        self.ln3 = nn.Linear(in_channels, projection_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.ln2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.ln3(x)
        return x


class SpatialAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling over spatial-topological tokens.
    Dynamically aggregates 384 spatial tokens into a single 48-dimensional vector
    guided by a learnable query vector representing the ideal neuroimaging biomarker profile.
    """
    def __init__(self, embed_dim=48, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        # Learnable query representing ideal biomarker profile
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.query, std=0.02)
        nn.init.kaiming_uniform_(self.k_proj.weight, a=math.sqrt(2.0))
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.kaiming_uniform_(self.v_proj.weight, a=math.sqrt(2.0))
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(2.0))
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) spatial-topological tokens (e.g. B, 384, 48)
        Returns:
            pooled: (B, D) pooled spatial representation (e.g. B, 48)
            attn_weights: (B, num_heads, 1, N) attention weights map
        """
        B, N, D = x.shape
        q = self.query.expand(B, -1, -1)  # (B, 1, D)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, 1, d)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)

        # Multi-head attention map over all 384 spatial tokens
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale    # (B, H, 1, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Weighted combination of values
        out = torch.matmul(attn_weights_drop, v)                            # (B, H, 1, d)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)               # (B, 1, D)
        out = self.out_proj(out).squeeze(1)                                 # (B, D)
        return out, attn_weights


class UnifiedEvidentialHead(nn.Module):
    """
    Unified Evidential Dirichlet Classification Head.
    Takes the concatenated 192-dimensional representation [z_cls (144) || z_spatial (48)]
    and maps it to Dirichlet evidence vectors e = [e_0, e_1].
    """
    def __init__(self, in_features=192, hidden_features=96, num_classes=2, dropout=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(2.0))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(2.0))
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 192) concatenated representation
        Returns:
            logits: (B, 2) unnormalized logits for cross-entropy & temperature scaling
            evidence: (B, 2) non-negative Dirichlet evidence e = softplus(logits)
            alpha: (B, 2) Dirichlet parameters alpha = e + 1
            probs: (B, 2) expected class probabilities p_k = alpha_k / S
            uncertainty: (B, 1) vacuous epistemic uncertainty u = K / S
        """
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        logits = self.fc2(x)

        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / S
        uncertainty = 2.0 / S

        return logits, evidence, alpha, probs, uncertainty


class Mamba_fusion_V4(nn.Module):
    """
    Tri-Modal Cross-Modal Mamba Fusion Engine V4:
    Unified Calibrated Evidential Architecture (~3.22M params).
    sMRI (3D ResNet-18) + fMRI (Compact ConvNeXtV2) + FC (BrainConnectomeGNN).
    """
    def __init__(self, config, in_features=128, in_features_ssm=48):
        super().__init__()
        self.in_features = in_features         # Spatial token length (default 128)
        self.in_features_ssm = in_features_ssm # SSM feature dimension (default 48)

        # Class tokens for sMRI, fMRI, and Connectome FC
        self.cls_token_s = nn.Parameter(torch.zeros(1, 1, in_features_ssm))
        self.cls_token_f = nn.Parameter(torch.zeros(1, 1, in_features_ssm))
        self.cls_token_c = nn.Parameter(torch.zeros(1, 1, in_features_ssm))

        # 1. sMRI Encoder: Right-sized 3D ResNet-18 (~2.08M params)
        smri_depth = config.model.get('smri_depth', 18)
        self.smri_encoder = generate_model(smri_depth)

        # 2. fMRI Encoder: Compact ConvNeXtV2 (~969K params)
        fmri_depths = config.model.get('fmri_encoder_deepth', [1, 1, 2, 1])
        fmri_dims = config.model.get('fmri_encoder_dims', [16, 32, 64, 128, 256])
        self.fmri_encoder = ConvNeXtV2_Mamba(
            depths=fmri_depths,
            dims=fmri_dims
        )

        # 3. FC Encoder: BrainConnectomeGNN (~36K params)
        self.fc_encoder = BrainConnectomeGNN(
            num_rois=116,
            d_model=in_features_ssm,
            target_tokens=in_features,
            threshold=config.model.get('fc_threshold', 0.2),
            dropout=config.model.get('fc_dropout', 0.2)
        )

        # Feature Adapters to SSM Dimension (48)
        smri_feat_shape = config.model.get('smri_feature_shape', 54)
        ds_name = config.dataset.name if hasattr(config, 'dataset') and hasattr(config.dataset, 'name') else 'abide'
        fmri_key = f"{ds_name.lower()}_fmri_feature_shape"
        fmri_feat_shape = config.model.get(fmri_key, 64)

        self.smri_change = nn.Linear(smri_feat_shape, in_features_ssm)
        self.fmri_change = nn.Linear(fmri_feat_shape, in_features_ssm)

        # Bidirectional State-Space Mamba Mixers
        d_state = config.model.get('d_state', 8)
        d_conv = config.model.get('d_conv', 3)
        expand = config.model.get('expand', 4)

        self.mamba_r1 = MambaVisionMixer_Fuse(d_model=in_features_ssm, d_state=d_state, d_conv=d_conv, expand=expand, output_style='two')
        self.mamba_r2 = MambaVisionMixer_Fuse(d_model=in_features_ssm, d_state=d_state, d_conv=d_conv, expand=expand, output_style='two')
        self.mamba_r_fc = MambaVisionMixer_Fuse(d_model=in_features_ssm, d_state=d_state, d_conv=d_conv, expand=expand, output_style='two')
        self.mamba_r3 = MambaVisionMixer_Cls(d_model=in_features_ssm, d_state=d_state, d_conv=d_conv, expand=expand)

        # SSM inner expansion dimension: int(expand * in_features_ssm // 3) = 64
        d_inner = int(expand * in_features_ssm // 3)
        self.predictor = MLPHead_V3(d_inner, in_features_ssm)
        self.sl1 = ContrastiveLossModel()

        # E4 Innovation: Spatial Multi-Head Attention Pooling
        attn_heads = config.model.get('spatial_attn_heads', 4)
        self.spatial_attn_pool = SpatialAttentionPooling(
            embed_dim=in_features_ssm,
            num_heads=attn_heads,
            dropout=config.model.get('spatial_attn_dropout', 0.1)
        )

        # E4 Innovation: Single Unified Evidential Dirichlet Classification Head
        cls_num = config.model.get('cls_num', 2)
        hidden_head = config.model.get('unified_head_hidden', 96)
        drop_head = config.model.get('unified_head_dropout', 0.5)
        total_unified_dim = (in_features_ssm * 3) + in_features_ssm  # 144 + 48 = 192
        self.unified_head = UnifiedEvidentialHead(
            in_features=total_unified_dim,
            hidden_features=hidden_head,
            num_classes=cls_num,
            dropout=drop_head
        )

        # Preserved Reliability Gate V4 (RAF + DFD + Multi-Task EDL on 3 modalities)
        raf = config.model.get('reliability_fusion', True)
        dfd = config.model.get('dynamic_dim', True)
        min_dim = config.model.get('min_dim', 12)
        nested_dropout = config.model.get('nested_dropout', 0.3)
        self.gate = ReliabilityGate_V4(
            d=in_features_ssm,
            n_cls=cls_num,
            num_modalities=3,
            raf=raf,
            dfd=dfd,
            min_dim=min_dim,
            nested_dropout=nested_dropout
        ) if (raf or dfd) else None

        self.apply(self._init_weights)

        # Gradient checkpointing to fit comfortably in 2 GB VRAM
        if config.model.get('grad_checkpoint', True):
            blocks = [b for layer in (self.smri_encoder.layer1, self.smri_encoder.layer2,
                                      self.smri_encoder.layer3, self.smri_encoder.layer4) for b in layer]
            checkpoint_blocks(blocks + [b for stage in self.fmri_encoder.stages for b in stage])

    def _init_weights(self, m):
        trunc_normal_(self.cls_token_s, std=0.02)
        trunc_normal_(self.cls_token_f, std=0.02)
        trunc_normal_(self.cls_token_c, std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.0))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, smri, fmri, fmri_con, avail=None):
        """
        Args:
            smri: (B, 1, 96, 96, 96) sMRI volume
            fmri: (B, 116, 115) fMRI BOLD timeseries
            fmri_con: (B, 116, 116) Functional Connectivity Pearson correlation matrix
            avail: (B, 2) or (B, 3) bool tensor indicating modality availability
        Returns:
            logits: (B, 2) unnormalized logits for classification
            evidence: (B, 2) non-negative Dirichlet evidence
            alpha: (B, 2) Dirichlet parameters alpha = evidence + 1
            probs: (B, 2) expected probability distribution
            uncertainty: (B, 1) vacuous epistemic uncertainty
            attn_weights: (B, num_heads, 1, 384) spatial attention map
            r_loss: scalar contrastive perturbation loss
        """
        b = smri.shape[0]
        c = self.in_features  # 128

        # 1. Encode Modalities
        sMRI_f = self.smri_encoder(smri)                     # (B, 128, 6, 3, 3)
        fMRI_f = self.fmri_encoder(fmri.unsqueeze(1))        # (B, 128, 8, 8)

        # 2. Reshape and project to SSM dimension (48)
        sMRI_f = self.smri_change(sMRI_f.reshape(b, c, -1))     # (B, 128, 48)
        fMRI_f = self.fmri_change(fMRI_f.reshape(b, c, -1))     # (B, 128, 48)

        # 3. Encode Functional Connectivity Connectome Graph
        FC_f, FC_global = self.fc_encoder(fmri_con)          # (B, 128, 48), (B, 1, 48)

        # 4. Tri-Modal Evidential Reliability Gating (RAF + DFD)
        if self.gate is not None:
            sMRI_f, fMRI_f, FC_f = self.gate([sMRI_f, fMRI_f, FC_f], avail)

        # 5. Prepend Class Tokens for Global Stream
        cls_token_s = self.cls_token_s.expand(b, -1, -1)
        cls_token_f = self.cls_token_f.expand(b, -1, -1)
        cls_token_c = self.cls_token_c.expand(b, -1, -1) + FC_global

        sMRI_c = torch.cat((cls_token_s, sMRI_f), dim=1)     # (B, 129, 48)
        fMRI_c = torch.cat((cls_token_f, fMRI_f), dim=1)     # (B, 129, 48)
        FC_c = torch.cat((cls_token_c, FC_f), dim=1)         # (B, 129, 48)

        # 6. Tri-Modal SSM Fusion with Cross-Modal Hadamard Interaction
        fuse_out, r_loss = self.fuse_method(fMRI_c, sMRI_c, FC_c)  # (B, 387, 48)

        # 7. Stream A: Global Class Tokens [fMRI, sMRI, FC]
        token_len = c + 1  # 129
        fmri_cls = fuse_out[:, 0, :]                         # (B, 48)
        smri_cls = fuse_out[:, token_len, :]                 # (B, 48)
        fc_cls = fuse_out[:, token_len * 2, :]               # (B, 48)
        z_cls = torch.cat((fmri_cls, smri_cls, fc_cls), dim=-1)  # (B, 144)

        # 8. Stream B: Spatial-Topological Tokens with Multi-Head Attention Pooling
        fmri_spatial = fuse_out[:, 1:token_len, :]           # (B, 128, 48)
        smri_spatial = fuse_out[:, token_len + 1:token_len * 2, :]  # (B, 128, 48)
        fc_spatial = fuse_out[:, token_len * 2 + 1:, :]      # (B, 128, 48)
        x_spatial = torch.cat((fmri_spatial, smri_spatial, fc_spatial), dim=1)  # (B, 384, 48)

        z_spatial, attn_weights = self.spatial_attn_pool(x_spatial)  # (B, 48), (B, H, 1, 384)

        # 9. Concatenated Unified Representation
        z_unified = torch.cat((z_cls, z_spatial), dim=-1)    # (B, 192)

        # 10. Single Unified Calibrated Evidential Head
        logits, evidence, alpha, probs, uncertainty = self.unified_head(z_unified)

        return logits, evidence, alpha, probs, uncertainty, attn_weights, r_loss

    def fuse_method(self, fmri, smri, fc):
        # Joint tri-modal SSM mixing
        fuse_feature = torch.cat((fmri, smri, fc), dim=1)    # (B, 387, 48)
        fuse_feature = self.mamba_r3(fuse_feature)

        smri_y, smri_z = self.mamba_r1(smri)
        fmri_y, fmri_z = self.mamba_r2(fmri)
        fc_y, fc_z = self.mamba_r_fc(fc)

        # Cross-modal bidirectional interactions
        fmri_fuse = (fmri_y * smri_z)
        smri_fuse = (smri_y * fmri_z)
        fc_fuse = (fc_y * 0.5 * (smri_z + fmri_z))
        sf_feature_o = torch.cat((fmri_fuse, smri_fuse, fc_fuse), dim=1)

        # Contrastive perturbation during training
        if self.training:
            fmri_cls = fmri[:, 0:1, :]
            fmri_body = fmri[:, 1:, :] + torch.sign(fmri[:, 1:, :]) * F.normalize(torch.rand_like(fmri[:, 1:, :]))
            fmri_rc = torch.cat((fmri_cls, fmri_body), dim=1)

            smri_cls = smri[:, 0:1, :]
            smri_body = smri[:, 1:, :] + torch.sign(smri[:, 1:, :]) * F.normalize(torch.rand_like(smri[:, 1:, :]))
            smri_rc = torch.cat((smri_cls, smri_body), dim=1)

            fc_cls = fc[:, 0:1, :]
            fc_body = fc[:, 1:, :] + torch.sign(fc[:, 1:, :]) * F.normalize(torch.rand_like(fc[:, 1:, :]))
            fc_rc = torch.cat((fc_cls, fc_body), dim=1)

            smri_y_rc, smri_z_rc = self.mamba_r1(smri_rc)
            fmri_y_rc, fmri_z_rc = self.mamba_r2(fmri_rc)
            fc_y_rc, fc_z_rc = self.mamba_r_fc(fc_rc)

            fmri_fuse_rc = (fmri_y_rc * smri_z_rc)
            smri_fuse_rc = (smri_y_rc * fmri_z_rc)
            fc_fuse_rc = (fc_y_rc * 0.5 * (smri_z_rc + fmri_z_rc))
            sf_feature = torch.cat((fmri_fuse_rc, smri_fuse_rc, fc_fuse_rc), dim=1)
            r_loss = self.sl1.sf_loss(sf_feature_o, sf_feature)
        else:
            r_loss = torch.tensor(0.0, device=fmri.device)

        fuse = self.predictor(sf_feature_o + fuse_feature)
        return fuse, r_loss
