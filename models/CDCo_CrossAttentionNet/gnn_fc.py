import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    """
    Spectral Graph Convolution on Functional Connectivity Adjacency.
    H' = D^{-1/2} A D^{-1/2} H W
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, h, adj):
        # h: (B, N, in_features)
        # adj: (B, N, N) normalized adjacency
        h = self.dropout(h)
        support = torch.matmul(h, self.weight) # (B, N, out_features)
        output = torch.matmul(adj, support) + self.bias
        return output

class BrainGNN(nn.Module):
    """
    Graph Neural Network for Functional Connectivity (FC).
    Encodes the 116x116 Pearson Correlation matrix into ROI-level topological embeddings.
    Input: (B, num_rois=116, num_rois=116)
    Output: (B, num_rois=116, d_model=128)
    Total parameters: ~165K
    """
    def __init__(self, num_rois=116, d_model=128, threshold=0.2, dropout=0.25):
        super(BrainGNN, self).__init__()
        self.num_rois = num_rois
        self.d_model = d_model
        self.threshold = threshold

        # Initial node feature projection: each ROI's initial feature is its FC profile across all 116 ROIs
        self.input_proj = nn.Sequential(
            nn.Linear(num_rois, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2-layer Graph Convolution Network
        self.gc1 = GraphConvolutionLayer(d_model, d_model, dropout=dropout)
        self.gc2 = GraphConvolutionLayer(d_model, d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def normalize_adj(self, adj):
        # adj shape: (B, N, N)
        # Apply threshold to remove weak/noisy correlations
        mask = (adj >= self.threshold).float()
        adj_thresh = adj * mask
        
        # Add self-loops
        eye = torch.eye(self.num_rois, device=adj.device).unsqueeze(0)
        adj_loop = adj_thresh + eye
        
        # Degree normalization: D^{-1/2} A D^{-1/2}
        degree = torch.sum(adj_loop, dim=-1) # (B, N)
        deg_inv_sqrt = torch.pow(torch.clamp(degree, min=1e-5), -0.5)
        deg_mat = torch.diag_embed(deg_inv_sqrt) # (B, N, N)
        
        norm_adj = torch.matmul(torch.matmul(deg_mat, adj_loop), deg_mat)
        return norm_adj

    def forward(self, fc_matrix):
        # fc_matrix: (B, 116, 116)
        b, n, _ = fc_matrix.size()
        
        # 1. Normalized Adjacency
        norm_adj = self.normalize_adj(fc_matrix)
        
        # 2. Initial node embedding from functional connectivity profile
        h0 = self.input_proj(fc_matrix) # (B, 116, d_model)
        
        # 3. Message passing across brain connectivity network
        h1 = F.gelu(self.gc1(h0, norm_adj))
        h2 = self.gc2(h1, norm_adj)
        
        # Residual connection
        out = self.norm(h2 + h0)
        return out # (B, 116, 128)
