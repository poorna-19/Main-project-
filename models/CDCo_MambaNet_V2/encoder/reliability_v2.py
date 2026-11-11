"""
Reliability-Aware Adaptive Fusion (RAF) and Dynamic Feature Dimension (DFD) V2.
Extended to Tri-Modal Inputs: [sMRI, fMRI, Functional Connectivity (FC)].

Preserves:
1. Dirichlet Evidential Uncertainty heads per modality (Han et al., ICLR 2021).
2. RAF: Blends less reliable modalities with learned stand-in tokens.
3. DFD: Dynamically truncates kept feature dimensions k_min..d.
4. Nested Dropout: Enforces ordered representation importance during training.
5. EDL Loss: Multi-task Dirichlet cross-entropy with annealed KL penalty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def edl_loss(evidence, y, kl_weight):
    """
    Expected cross-entropy under Dir(alpha) plus annealed KL(Dir(alpha~) || Dir(1))
    on non-target evidence (TMC). y may be soft (e.g., mixup).
    """
    alpha = evidence + 1
    S = alpha.sum(-1, keepdim=True)
    ce = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
    a = y + (1 - y) * alpha
    Sa = a.sum(-1)
    k = torch.tensor(float(a.shape[-1]), device=a.device)
    kl = (torch.lgamma(Sa) - torch.lgamma(k) - torch.lgamma(a).sum(-1)
          + ((a - 1) * (torch.digamma(a) - torch.digamma(Sa)[:, None])).sum(-1))
    return (ce + kl_weight * kl).mean()


class ReliabilityGate_V2(nn.Module):
    """
    Gates the aligned token sets [sMRI (B, C, d), fMRI (B, C, d), FC (B, C, d)]
    by per-patient evidential reliability across all 3 modalities.
    """
    def __init__(self, d, n_cls, num_modalities=3, raf=True, dfd=True, min_dim=12, nested_dropout=0.3):
        super().__init__()
        self.d = d
        self.n_cls = n_cls
        self.num_modalities = num_modalities
        self.raf = raf
        self.dfd = dfd
        self.min_dim = min_dim
        self.nested_dropout = nested_dropout

        # One evidential head per modality (sMRI, fMRI, FC)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, n_cls)
            ) for _ in range(num_modalities)
        ])

        # Learned stand-ins for [sMRI, fMRI, FC]
        self.missing = nn.Parameter(torch.randn(num_modalities, 1, 1, d))
        self.info = {}

    def width(self, a, d):
        k = self.min_dim + (d - self.min_dim) * a
        if self.training and self.nested_dropout > 0:
            cut = torch.rand_like(k) < self.nested_dropout
            k = torch.where(cut, torch.minimum(k, self.min_dim + (d - self.min_dim) * torch.rand_like(k)), k)
        return k

    def forward(self, tokens, avail=None):
        """
        Args:
            tokens: list of [sMRI, fMRI, FC], each shape (B, C, d)
            avail: (B, num_modalities) bool; if (B, 2), FC inherits fMRI availability
        """
        b = tokens[0].shape[0]
        device = tokens[0].device

        if avail is None:
            avail = torch.ones(b, self.num_modalities, dtype=torch.bool, device=device)
        elif avail.shape[1] == 2 and self.num_modalities == 3:
            # Backward compatibility: FC availability derives from fMRI availability
            avail_fc = avail[:, 1:2]
            avail = torch.cat([avail, avail_fc], dim=1)

        # 1. Compute Evidential Dirichlet outputs
        evidence = [F.softplus(h(t.mean(1))) for h, t in zip(self.heads, tokens)]
        n_cls = evidence[0].shape[-1]

        # 2. Uncertainty & Reliability: r = 1 - K / sum(alpha)
        r = torch.stack([1.0 - n_cls / (e + 1.0).sum(-1) for e in evidence], dim=1).detach() * avail.float()

        # 3. Relative Weights: a_m = r_m / max(r)
        max_r = r.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        a = r / max_r

        out, dims = [], []
        for m, t in enumerate(tokens):
            am = a[:, m, None, None]
            d = t.shape[-1]

            if self.raf:
                # Blend toward learned stand-in based on relative reliability
                t = am * t + (1.0 - am) * self.missing[m]
            else:
                t = torch.where(avail[:, m, None, None], t, self.missing[m].expand_as(t))

            # DFD dynamic kept dimensions
            k = self.width(a[:, m], d) if self.dfd else torch.full_like(a[:, m], d)
            t = t * torch.clamp(k[:, None] - torch.arange(d, device=t.device), 0.0, 1.0)[:, None, :]
            out.append(t)
            dims.append(k)

        self.info = {
            "evidence": evidence,
            "avail": avail,
            "reliability": r,
            "weight": a,
            "dims": torch.stack(dims, dim=1)
        }
        return out

    def loss(self, y, kl_weight):
        ev, avail = self.info["evidence"], self.info["avail"]
        return sum(
            edl_loss(e[avail[:, m]], y[avail[:, m]], kl_weight)
            for m, e in enumerate(ev) if avail[:, m].any()
        )
