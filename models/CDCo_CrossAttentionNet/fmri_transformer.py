import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Feature Extractor & Transformer Encoder for fMRI BOLD time series.
    Input: (B, num_rois=116, time_steps=115)
    Output: (B, num_rois=116, d_model=128)
    Total parameters: ~480K
    """
    def __init__(self, num_rois=116, time_steps=115, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.2):
        super(TemporalTransformerEncoder, self).__init__()
        self.num_rois = num_rois
        self.d_model = d_model

        # Temporal filter bank: extracts multi-scale hemodynamic frequency bands
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), # 115 -> 58
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16) # 16 summary temporal channels
        )

        # Projection from multi-channel temporal features to d_model
        self.proj = nn.Linear(64 * 16, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, num_rois, d_model) * 0.02)

        # Pre-LN Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (B, 116, 115)
        b, r, t = x.size()
        
        # Process each ROI time-series through temporal filter bank
        # Reshape to (B * 116, 1, 115)
        x_flat = x.view(b * r, 1, t)
        temp_feat = self.temporal_stem(x_flat).view(b * r, -1) # (B * 116, 64 * 16)
        
        # Project to d_model and reshape back to (B, 116, d_model)
        roi_tokens = self.proj(temp_feat).view(b, r, self.d_model)
        roi_tokens = roi_tokens + self.pos_emb
        
        # Contextualize across brain ROIs using Self-Attention
        out = self.transformer(roi_tokens)
        return self.norm(out) # (B, 116, 128)
