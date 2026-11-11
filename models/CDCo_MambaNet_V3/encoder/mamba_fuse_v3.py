"""
Tri-Modal Mamba Fusion Engine V3 for CDCo-MambaNet V3 (Parameter-Efficient Right-Sized Architecture).
Fuses sMRI volume tokens, fMRI BOLD tokens, and Functional Connectivity (FC) graph tokens.

Key Right-Sizing Advancements:
1. sMRI Branch: 3D ResNet-18 (~2.08M params, down from 2.9M ResNet-50) or ResNet-10 (~0.90M params).
2. fMRI Branch: Compact ConvNeXtV2 (~969K params, down from 24.7M in baseline).
3. FC Branch: BrainConnectomeGNN (~36K params, target_tokens=128).
4. Spatial Token Sequence Length: 128 (down from 512), maintaining rich spatial representation while reducing SSM/MLP computation by 4x.
5. All Original Novelty Components Preserved:
   - Bidirectional Mamba SSM Mixers (mamba_r1, mamba_r2, mamba_r_fc, mamba_r3).
   - ReliabilityGate_V3 (RAF + DFD + Multi-Task EDL + Nested Dropout).
   - Head 1 Regularization (LayerNorm + Dropout).
   - Dual Classification Heads (cls_linear and mlp_cls).
   - Training contrastive perturbation loss.
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
from .reliability_v3 import ReliabilityGate_V3


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


class Mlp_cls_V3(nn.Module):
    """Head 2: Spatial-Topological MLP Classifier over tri-modal fused tokens."""
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.5, cls_num=2, cls_in=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        # Tri-modal concatenation has cls_in * 3 tokens (e.g. 128 * 3 = 384)
        total_tokens = cls_in * 3
        self.norm = nn.LayerNorm(total_tokens)
        self.cls = nn.Linear(total_tokens, cls_num)

    def forward(self, x):
        x = self.fc1(x).squeeze(-1)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        cls = self.cls(x)
        return cls


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


class Mamba_fusion_V3(nn.Module):
    """
    Tri-Modal Cross-Modal Mamba Fusion Engine V3:
    Right-sized architecture (<3.5M params):
    sMRI (3D ResNet-18/10) + fMRI (Compact ConvNeXtV2) + FC (BrainConnectomeGNN).
    """
    def __init__(self, config, in_features=128, in_features_ssm=48):
        super().__init__()
        self.in_features = in_features         # Spatial token length (default 128)
        self.in_features_ssm = in_features_ssm # SSM feature dimension (default 48)

        # Class tokens for sMRI, fMRI, and Connectome FC
        self.cls_token_s = nn.Parameter(torch.zeros(1, 1, in_features_ssm))
        self.cls_token_f = nn.Parameter(torch.zeros(1, 1, in_features_ssm))
        self.cls_token_c = nn.Parameter(torch.zeros(1, 1, in_features_ssm))

        # Modality Encoders
        # 1. sMRI: Right-sized 3D ResNet-18 (~2.08M params) or ResNet-10 (~0.90M params)
        smri_depth = config.model.get('smri_depth', 18)
        self.smri_encoder = generate_model(smri_depth)

        # 2. fMRI: Compact ConvNeXtV2 (~969K params)
        fmri_depths = config.model.get('fmri_encoder_deepth', [1, 1, 2, 1])
        fmri_dims = config.model.get('fmri_encoder_dims', [16, 32, 64, 128, 256])
        self.fmri_encoder = ConvNeXtV2_Mamba(
            depths=fmri_depths,
            dims=fmri_dims
        )

        # 3. FC: BrainConnectomeGNN (~36K params)
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

        # Head 1: Linear on concatenated tri-modal class tokens [fMRI, sMRI, FC] (48 * 3 = 144)
        cls_num = config.model.get('cls_num', 2)
        self.regularize_head1 = config.model.get('head1_regularization', True)
        if self.regularize_head1:
            self.head1_norm = nn.LayerNorm(in_features_ssm * 3)
            self.head1_drop = nn.Dropout(config.model.get('head1_dropout', 0.5))
        self.cls_linear = nn.Linear(in_features_ssm * 3, cls_num)

        # Head 2: Spatial-Topological MLP classifier (in_features * 3 = 384 tokens)
        self.mlp_cls = Mlp_cls_V3(
            in_features=in_features_ssm,
            act_layer=nn.GELU,
            drop=config.model.get('head2_dropout', 0.5),
            cls_num=cls_num,
            cls_in=in_features
        )

        # Preserved Reliability Gate V3 (RAF + DFD + Multi-Task EDL on 3 modalities)
        raf = config.model.get('reliability_fusion', True)
        dfd = config.model.get('dynamic_dim', True)
        min_dim = config.model.get('min_dim', 12)
        nested_dropout = config.model.get('nested_dropout', 0.3)
        self.gate = ReliabilityGate_V3(
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
        """
        b = smri.shape[0]
        c = self.in_features  # 128

        # 1. Encode Modalities
        sMRI_f = self.smri_encoder(smri)                     # (B, 128, 6, 3, 3)
        fMRI_f = self.fmri_encoder(fmri.unsqueeze(1))        # (B, 128, 8, 8)

        # 2. Reshape and project to SSM dimension (48)
        sMRI_f = self.smri_change(sMRI_f.view(b, c, -1))     # (B, 128, 48)
        fMRI_f = self.fmri_change(fMRI_f.view(b, c, -1))     # (B, 128, 48)

        # 3. Encode Functional Connectivity Connectome Graph
        FC_f, FC_global = self.fc_encoder(fmri_con)          # (B, 128, 48), (B, 1, 48)

        # 4. Tri-Modal Evidential Reliability Gating (RAF + DFD)
        if self.gate is not None:
            sMRI_f, fMRI_f, FC_f = self.gate([sMRI_f, fMRI_f, FC_f], avail)

        # 5. Prepend Class Tokens for Head 1
        cls_token_s = self.cls_token_s.expand(b, -1, -1)
        cls_token_f = self.cls_token_f.expand(b, -1, -1)
        cls_token_c = self.cls_token_c.expand(b, -1, -1) + FC_global

        sMRI_c = torch.cat((cls_token_s, sMRI_f), dim=1)     # (B, 129, 48)
        fMRI_c = torch.cat((cls_token_f, fMRI_f), dim=1)     # (B, 129, 48)
        FC_c = torch.cat((cls_token_c, FC_f), dim=1)         # (B, 129, 48)

        # 6. Tri-Modal Fusion with Class Tokens
        fuse_out, r_loss = self.fuse_method(fMRI_c, sMRI_c, FC_c)

        # Extract fused class tokens from each stream
        token_len = c + 1  # 129
        fmri_age = fuse_out[:, 0, :]
        smri_age = fuse_out[:, token_len, :]
        fc_age = fuse_out[:, token_len * 2, :]
        age_out_linear = torch.cat((fmri_age, smri_age, fc_age), dim=-1)  # (B, 144)

        if getattr(self, 'regularize_head1', False):
            age_out_linear = self.head1_norm(age_out_linear)
            age_out_linear = self.head1_drop(age_out_linear)
        age_out = self.cls_linear(age_out_linear)             # (B, 2)

        # 7. Tri-Modal Spatial-Topological Fusion for Head 2
        mlp_out, r_loss1 = self.fuse_method1(fMRI_f, sMRI_f, FC_f)
        mlp_out = self.mlp_cls(mlp_out)                      # (B, 2)

        return age_out, mlp_out, (r_loss + r_loss1)

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

    def fuse_method1(self, fmri, smri, fc):
        # Joint spatial-topological SSM mixing
        fuse_feature = torch.cat((fmri, smri, fc), dim=1)    # (B, 384, 48)
        fuse_feature = self.mamba_r3(fuse_feature)

        smri_y, smri_z = self.mamba_r1(smri)
        fmri_y, fmri_z = self.mamba_r2(fmri)
        fc_y, fc_z = self.mamba_r_fc(fc)

        fmri_fuse = (fmri_y * smri_z)
        smri_fuse = (smri_y * fmri_z)
        fc_fuse = (fc_y * 0.5 * (smri_z + fmri_z))
        sf_feature_o = torch.cat((fmri_fuse, smri_fuse, fc_fuse), dim=1)

        if self.training:
            fmri_n = fmri + torch.sign(fmri) * F.normalize(torch.rand_like(fmri))
            smri_n = smri + torch.sign(smri) * F.normalize(torch.rand_like(smri))
            fc_n = fc + torch.sign(fc) * F.normalize(torch.rand_like(fc))

            smri_y_n, smri_z_n = self.mamba_r1(smri_n)
            fmri_y_n, fmri_z_n = self.mamba_r2(fmri_n)
            fc_y_n, fc_z_n = self.mamba_r_fc(fc_n)

            fmri_fuse_n = (fmri_y_n * smri_z_n)
            smri_fuse_n = (smri_y_n * fmri_z_n)
            fc_fuse_n = (fc_y_n * 0.5 * (smri_z_n + fmri_z_n))
            sf_feature = torch.cat((fmri_fuse_n, smri_fuse_n, fc_fuse_n), dim=1)
            r_loss = self.sl1.sf_loss(sf_feature_o, sf_feature)
            fuse = self.predictor(sf_feature + fuse_feature)
        else:
            r_loss = torch.tensor(0.0, device=fmri.device)
            fuse = self.predictor(sf_feature_o + fuse_feature)

        return fuse, r_loss
