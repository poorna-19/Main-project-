import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

class SiteDiscriminator(nn.Module):
    """
    Domain-Adversarial Scanner Site Discriminator.
    Forces feature representations to be invariant across the 17 ABIDE scanner sites.
    """
    def __init__(self, in_features=256, num_sites=17, hidden_dim=128, dropout=0.3):
        super(SiteDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sites)
        )

    def forward(self, x, alpha=1.0):
        reversed_x = grad_reverse(x, alpha)
        return self.net(reversed_x)

class CalibratedEvidentialHead(nn.Module):
    """
    Single Unified Calibrated Evidential Deep Learning (EDL) Head.
    Parameterizes a Dirichlet distribution Dir(alpha) over ASD and Control.
    Eliminates conflicting dual-head architectures.
    Total parameters: ~120K
    """
    def __init__(self, in_features=256, num_classes=2, hidden_dim=128, dropout=0.4):
        super(CalibratedEvidentialHead, self).__init__()
        self.num_classes = num_classes
        
        self.evidence_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, in_features=256)
        logits = self.evidence_net(x)
        # Softplus ensures strictly positive evidence values e_k >= 0
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        strength = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / strength
        uncertainty = self.num_classes / strength
        return {
            'logits': logits,
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty': uncertainty.squeeze(-1)
        }

def edl_loss(alpha, target, epoch=0, anneal_epochs=10, num_classes=2):
    """
    Evidential Cross-Entropy with KL Divergence regularizer against uniform Dirichlet prior.
    """
    # target: (B,) class indices or (B, K) one-hot
    if target.dim() == 1:
        y = F.one_hot(target, num_classes=num_classes).float()
    else:
        y = target.float()

    s = torch.sum(alpha, dim=-1, keepdim=True) # Dirichlet strength
    
    # Expected cross-entropy under Dirichlet distribution
    # L_ace = sum_k y_k * (psi(S) - psi(alpha_k))
    log_likelihood = torch.sum(y * (torch.digamma(s) - torch.digamma(alpha)), dim=-1)
    
    # KL divergence regularizer penalizing misleading evidence on non-target classes
    alpha_tilde = y + (1.0 - y) * alpha
    s_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
    
    kl_div = torch.lgamma(s_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True) \
             - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device)) \
             + torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(s_tilde)), dim=-1, keepdim=True)
    kl_div = kl_div.squeeze(-1)

    anneal_coef = min(1.0, epoch / max(1, anneal_epochs))
    total_loss = torch.mean(log_likelihood + anneal_coef * kl_div)
    return total_loss
