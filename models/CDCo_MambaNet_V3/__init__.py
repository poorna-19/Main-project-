"""
CDCo-MambaNet V3: Parameter-Efficient Right-Sized Cross-Domain Collaborative Mamba Network.
Addresses severe overparameterization (27.8M -> ~3.2M params) for ABIDE I cohort (N=227).
Preserves:
- Tri-modal inputs: sMRI (3D ResNet-18) + fMRI (Compact ConvNeXtV2) + FC (BrainConnectomeGNN).
- All novelty components: RAF, DFD, Nested Dropout, Multi-Task EDL, Head Regularization, KL Distillation.
"""
from .cdcomnet_v3 import CDCo_MambaNet_V3

__all__ = ["CDCo_MambaNet_V3"]
