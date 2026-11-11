import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_prob=0.1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(drop_prob) if drop_prob > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LightweightResNet3D(nn.Module):
    """
    Lightweight 3D ResNet encoder for sMRI volumes (1, 96, 96, 96).
    Produces 116 ROI-aligned morphometric tokens of dimension d_model.
    Total parameters: ~2.8M (designed to avoid HDLSS overfitting).
    """
    def __init__(self, in_channels=1, num_rois=116, d_model=128, drop_prob=0.2):
        super(LightweightResNet3D, self).__init__()
        self.num_rois = num_rois
        self.d_model = d_model

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False), # (32, 48, 48, 48)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2) # (32, 24, 24, 24)
        )

        # Residual Stages
        self.layer1 = BasicBlock3D(32, 32, stride=1, drop_prob=drop_prob)
        self.layer2 = BasicBlock3D(32, 64, stride=2, drop_prob=drop_prob)  # (64, 12, 12, 12)
        self.layer3 = BasicBlock3D(64, 128, stride=2, drop_prob=drop_prob) # (128, 6, 6, 6)
        self.layer4 = BasicBlock3D(128, 128, stride=2, drop_prob=drop_prob) # (128, 3, 3, 3)

        # Spatial-to-ROI projection via Anatomical Query Attention
        # Instead of a heavy unconstrained linear projection, 116 learnable ROI queries
        # attend to the 3D spatial morphological feature map.
        self.spatial_pool = nn.AdaptiveAvgPool3d((3, 3, 3)) # 27 spatial tokens
        self.spatial_proj = nn.Linear(128, d_model)
        self.roi_queries = nn.Parameter(torch.randn(1, num_rois, d_model) * 0.02)
        self.roi_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True, dropout=drop_prob)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (B, 1, 96, 96, 96)
        b = x.size(0)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Spatial feature tokens: (B, 128, 3, 3, 3) -> (B, 27, 128)
        spatial_tokens = self.spatial_pool(out).view(b, 128, -1).permute(0, 2, 1) # (B, 27, 128)
        spatial_tokens = self.spatial_proj(spatial_tokens)                         # (B, 27, d_model)

        # 116 Anatomical ROI queries attend to the 27 spatial tokens
        queries = self.roi_queries.expand(b, -1, -1)                              # (B, 116, d_model)
        roi_tokens, _ = self.roi_cross_attn(query=queries, key=spatial_tokens, value=spatial_tokens)
        
        return self.norm(roi_tokens) # (B, 116, 128)
