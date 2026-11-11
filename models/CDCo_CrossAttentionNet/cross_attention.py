import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Morpho-Temporal Cross-Attention Module.
    Allows structural brain tokens (sMRI) to query functional temporal dynamics (fMRI),
    and functional tokens to query structural morphological properties.
    Combines with GNN functional connectivity representations.
    Total parameters: ~295K
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.2):
        super(BidirectionalCrossAttention, self).__init__()
        self.d_model = d_model

        # Multi-Head Cross Attention: sMRI queries fMRI
        self.cross_s2f = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm_s = nn.LayerNorm(d_model)

        # Multi-Head Cross Attention: fMRI queries sMRI
        self.cross_f2s = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm_f = nn.LayerNorm(d_model)

        # Tri-modal Fusion Projection (sMRI + fMRI + GNN FC)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Subject-level pooling projection (Mean + Max pooling = 256 * 2 -> 256)
        self.subject_pool = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, h_smri, h_fmri, h_gnn):
        # h_smri: (B, 116, 128)
        # h_fmri: (B, 116, 128)
        # h_gnn:  (B, 116, 128)

        # 1. sMRI queries fMRI
        attn_s2f, _ = self.cross_s2f(query=h_smri, key=h_fmri, value=h_fmri)
        z_s = self.norm_s(h_smri + attn_s2f)

        # 2. fMRI queries sMRI
        attn_f2s, _ = self.cross_f2s(query=h_fmri, key=h_smri, value=h_smri)
        z_f = self.norm_f(h_fmri + attn_f2s)

        # 3. Concatenate all three aligned representations
        tri_modal = torch.cat([z_s, z_f, h_gnn], dim=-1) # (B, 116, 384)
        node_latents = self.fusion_proj(tri_modal)        # (B, 116, 256)

        # 4. Subject-level pooling (Mean + Max pool across the 116 anatomical ROIs)
        mean_pool = torch.mean(node_latents, dim=1) # (B, 256)
        max_pool = torch.max(node_latents, dim=1)[0] # (B, 256)
        subject_latent = self.subject_pool(torch.cat([mean_pool, max_pool], dim=-1)) # (B, 256)

        return subject_latent, node_latents
