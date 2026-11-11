"""
Connectome Graph Neural Network (BrainConnectomeGNN) for CDCo-MambaNet V2
Encodes 116x116 Pearson Functional Connectivity (FC) matrices into topological embeddings.

Architecture:
- Symmetric Normalized Adjacency with learnable soft thresholding.
- Multi-Head Connectome Graph Convolution / Attention layers.
- Residual skip connections and LayerNorm for numerical stability.
- Dual readout: Global Connectome Token (B, 1, d) + Spatial Feature Sequence (B, 512, d).
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
    - spatial_tokens: (B, target_tokens=512, d_model) matching sMRI and fMRI token counts.
    - global_token:  (B, 1, d_model) global connectome class token.
    """
    def __init__(self, num_rois=116, d_model=48, target_tokens=512, threshold=0.2, dropout=0.2):
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

        # Spatial Expansion: projects 116 ROI tokens into target_tokens (512)
        # to match sMRI and fMRI spatial token dimension
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
        # Soft-threshold positive and negative edges
        adj = torch.where(torch.abs(corr) >= self.threshold, corr, torch.zeros_like(corr))
        # Add self-loops (self-correlation = 1.0)
        b, n, _ = corr.shape
        eye = torch.eye(n, device=corr.device).unsqueeze(0).expand(b, -1, -1)
        adj_loop = torch.abs(adj) + eye

        # Degree matrix normalization
        degree = torch.sum(adj_loop, dim=-1)  # (B, N)
        deg_inv_sqrt = torch.pow(torch.clamp(degree, min=1e-5), -0.5)
        deg_mat = torch.diag_embed(deg_inv_sqrt)  # (B, N, N)

        norm_adj = torch.matmul(torch.matmul(deg_mat, adj_loop), deg_mat)
        return norm_adj

    def forward(self, corr):
        """
        Args:
            corr: (B, 116, 116) Pearson correlation matrices.
        Returns:
            spatial_tokens: (B, 512, d_model)
            global_token:  (B, 1, d_model)
        """
        b, n, _ = corr.shape

        # 1. Normalize Connectome Adjacency
        norm_adj = self.compute_norm_adjacency(corr)

        # 2. Project Initial Node Features
        h0 = self.input_proj(corr)          # (B, 116, d_model)

        # 3. Message Passing over Brain Network
        h1 = self.gc1(h0, norm_adj)         # (B, 116, d_model)
        h2 = self.gc2(h1, norm_adj)         # (B, 116, d_model)
        h_rois = self.out_norm(h2 + h0)     # (B, 116, d_model)

        # 4. Global Connectome Readout Token (Attention-weighted pooling)
        attn_weights = self.attention_pool(h_rois)  # (B, 116, 1)
        global_token = torch.sum(h_rois * attn_weights, dim=1, keepdim=True)  # (B, 1, d_model)

        # 5. Spatial Sequence Expansion (116 -> 512 tokens)
        # Transpose to (B, d_model, 116), apply linear, then transpose back to (B, 512, d_model)
        spatial_tokens = self.spatial_proj(h_rois.transpose(1, 2)).transpose(1, 2)  # (B, 512, d_model)

        return spatial_tokens, global_token
