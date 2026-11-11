"""
Connectome Graph Neural Network (BrainConnectomeGNN) for CDCo-MambaNet V3
Encodes 116x116 Pearson Functional Connectivity (FC) matrices into topological embeddings.

Architecture:
- Symmetric Normalized Adjacency with learnable soft thresholding.
- Multi-Head Connectome Graph Convolution / Attention layers.
- Residual skip connections and LayerNorm for numerical stability.
- Dual readout: Global Connectome Token (B, 1, d) + Spatial Feature Sequence (B, target_tokens, d).
Default target_tokens = 128 (compact right-sized configuration).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectomeGraphConv(nn.Module):
    """
    Graph Convolution Layer modulated by Functional Connectivity adjacency.
    H' = LeakyReLU( LN( A_norm @ H @ W ) ) + H_residual
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        if in_features != out_features:
            self.res = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res = nn.Identity()

    def forward(self, h, adj):
        # h: (B, N, in_features)
        # adj: (B, N, N) normalized adjacency matrix
        residual = self.res(h)
        support = self.linear(h)                     # (B, N, out_features)
        out = torch.matmul(adj, support)             # (B, N, out_features)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + residual


class BrainConnectomeGNN(nn.Module):
    """
    Connectome-Aware Feature Extractor.
    Processes the (B, 116, 116) Pearson correlation matrix.

    Outputs:
    - spatial_tokens: (B, target_tokens, d_model) matching sMRI and fMRI token counts.
    - global_token:  (B, 1, d_model) global connectome class token.
    """
    def __init__(self, num_rois=116, d_model=48, target_tokens=128, threshold=0.2, dropout=0.2):
        super().__init__()
        self.num_rois = num_rois
        self.d_model = d_model
        self.target_tokens = target_tokens
        self.threshold = threshold

        # Initial ROI feature projection: each ROI's profile is its 116 connectivity values
        self.input_proj = nn.Sequential(
            nn.Linear(num_rois, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # 2-stage Graph Convolution
        self.gc1 = ConnectomeGraphConv(d_model, d_model, dropout=dropout)
        self.gc2 = ConnectomeGraphConv(d_model, d_model, dropout=dropout)

        # Global Attention Pooling for class token
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # Spatial Expansion: projects 116 ROI tokens into target_tokens
        self.spatial_proj = nn.Sequential(
            nn.Linear(num_rois, target_tokens),
            nn.LayerNorm(target_tokens)
        )
        self.out_norm = nn.LayerNorm(d_model)

    def compute_norm_adjacency(self, corr):
        """
        Computes degree-normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
        with correlation thresholding to filter noise.
        """
        # corr: (B, N, N)
        adj = torch.where(torch.abs(corr) >= self.threshold, corr, torch.zeros_like(corr))
        b, n, _ = corr.shape
        eye = torch.eye(n, device=corr.device).unsqueeze(0).expand(b, -1, -1)
        adj_loop = torch.abs(adj) + eye

        # Degree matrix normalization
        degree = torch.sum(adj_loop, dim=-1) # (B, N)
        deg_inv_sqrt = torch.pow(degree.clamp(min=1e-6), -0.5)
        d_mat = torch.diag_embed(deg_inv_sqrt) # (B, N, N)
        norm_adj = torch.matmul(torch.matmul(d_mat, adj_loop), d_mat)
        return norm_adj

    def forward(self, corr):
        """
        Args:
            corr: (B, 116, 116) Pearson correlation matrix
        Returns:
            spatial_tokens: (B, target_tokens, d_model)
            global_token:  (B, 1, d_model)
        """
        # Ensure symmetric correlation input
        corr_sym = (corr + corr.transpose(1, 2)) / 2.0

        # Compute normalized adjacency
        adj = self.compute_norm_adjacency(corr_sym)

        # Initial node embedding: (B, N, N) -> (B, N, d_model)
        h = self.input_proj(corr_sym)

        # Graph Convolutions
        h = self.gc1(h, adj)
        h = self.gc2(h, adj)

        # 1. Global Connectome Token via attention pooling: (B, 1, d_model)
        attn_weights = self.attention_pool(h) # (B, N, 1)
        global_token = torch.sum(h * attn_weights, dim=1, keepdim=True) # (B, 1, d_model)

        # 2. Spatial projection from N (116) to target_tokens (e.g. 128)
        # Transpose to (B, d_model, N) -> linear over N -> (B, d_model, target_tokens) -> transpose back
        h_t = h.transpose(1, 2)
        spatial_tokens = self.spatial_proj(h_t).transpose(1, 2) # (B, target_tokens, d_model)
        spatial_tokens = self.out_norm(spatial_tokens)

        return spatial_tokens, global_token
