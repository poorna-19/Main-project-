"""
Temperature Scaling and Evidential Calibration Module for CDCo-MambaNet V4 (Experiment E4).

Scientific Principles:
1. Validation-Only Fitting: Temperature T is fitted strictly on validation logits/labels.
2. Test Set Isolation: Test data is NEVER used for fitting or tuning temperature.
3. Separation of Concerns: Calibration is a post-training transformation that preserves
   model rank order while aligning confidence with empirical empirical accuracy.
4. Comprehensive Metrics: Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
   Negative Log-Likelihood (NLL), and Brier Score before and after calibration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibrator (Guo et al., ICML 2017).
    Calibrates logits: z_cal = z / T, where T > 0 is learned by minimizing NLL on validation data.
    """
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(init_temp))

    def forward(self, logits):
        """Scale logits by temperature."""
        t = self.temperature.clamp(min=0.05, max=10.0)
        return logits / t

    def get_temperature(self):
        """Return scalar temperature value."""
        return float(self.temperature.clamp(min=0.05, max=10.0).item())

    def fit(self, val_logits, val_labels, max_iter=150, lr=0.01):
        """
        Fit temperature on validation data only.
        
        Args:
            val_logits: (N, C) torch.Tensor or numpy.ndarray of unnormalized validation logits.
            val_labels: (N,) or (N, C) validation targets (class indices or one-hot).
            max_iter: Maximum optimization iterations.
            lr: Learning rate for Adam optimizer.
            
        Returns:
            dict containing initial_temp, optimal_temp, initial_nll, final_nll, initial_ece, final_ece.
        """
        if isinstance(val_logits, np.ndarray):
            val_logits = torch.from_numpy(val_logits).float()
        if isinstance(val_labels, np.ndarray):
            val_labels = torch.from_numpy(val_labels)

        device = self.temperature.device
        val_logits = val_logits.to(device)

        if val_labels.ndim > 1 and val_labels.shape[-1] > 1:
            targets = val_labels.argmax(dim=-1).to(device).long()
        else:
            targets = val_labels.to(device).long()

        # Compute pre-calibration metrics
        with torch.no_grad():
            init_nll = F.cross_entropy(val_logits, targets).item()
            init_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
            init_ece = compute_ece(init_probs, targets.cpu().numpy())

        # Optimize temperature using Adam with line search / cosine annealing
        optimizer = torch.optim.Adam([self.temperature], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-4)

        best_loss = float('inf')
        best_temp = self.get_temperature()

        for step in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = F.cross_entropy(scaled_logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Clamp temperature
            with torch.no_grad():
                self.temperature.clamp_(min=0.05, max=10.0)

            cur_loss = loss.item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_temp = self.get_temperature()

        # Restore best temperature
        with torch.no_grad():
            self.temperature.fill_(best_temp)
            final_scaled = self.forward(val_logits)
            final_nll = F.cross_entropy(final_scaled, targets).item()
            final_probs = F.softmax(final_scaled, dim=-1).cpu().numpy()
            final_ece = compute_ece(final_probs, targets.cpu().numpy())

        return {
            "initial_temp": 1.0,
            "optimal_temp": best_temp,
            "initial_nll": init_nll,
            "final_nll": final_nll,
            "initial_ece": init_ece,
            "final_ece": final_ece,
            "ece_reduction_pct": ((init_ece - final_ece) / max(1e-6, init_ece)) * 100.0
        }


def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: (N, C) numpy array of predicted probabilities.
        labels: (N,) numpy array of ground-truth integer class labels.
        n_bins: Number of confidence bins (standard is 10).
    """
    if probs.ndim == 2:
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier(probs, labels):
    """
    Compute Brier score: mean squared error between probabilities and one-hot labels.
    """
    N = len(labels)
    if probs.ndim == 2:
        C = probs.shape[1]
        one_hot = np.zeros((N, C))
        one_hot[np.arange(N), labels] = 1.0
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))
    else:
        return float(np.mean((probs - labels) ** 2))


def evaluate_calibration(logits, labels, n_bins=10):
    """
    Comprehensive calibration evaluation.
    
    Returns:
        dict with nll, brier, ece, mce, mean_confidence, accuracy, and bin info.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if labels.ndim > 1 and labels.shape[-1] > 1:
        labels = np.argmax(labels, axis=-1)

    # Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)

    # Negative log-likelihood
    eps = 1e-12
    nll = -float(np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))))
    brier = compute_brier(probs, labels)

    # Bin-wise computation
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / len(labels)) * diff
            mce = max(mce, diff)
            bin_stats.append({
                "bin": i,
                "range": [float(bin_lower), float(bin_upper)],
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf
            })

    return {
        "nll": nll,
        "brier": brier,
        "ece": float(ece),
        "mce": float(mce),
        "accuracy": float(np.mean(accuracies)),
        "mean_confidence": float(np.mean(confidences)),
        "bins": bin_stats
    }
