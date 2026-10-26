import torch.nn as nn
import math
from omegaconf import DictConfig
import os
import torch
import torch.nn.functional as F
from .encoder.mamba_fuse import Mamba_fusion

class CDCo_MambaNet(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.fuse_encoder = Mamba_fusion(config, config.model.fuse_channel, config.model.fuse_shape)
        self.contrastive_loss = None
        self.config = config
        # Justified classification rate: default 0.8 so ground truth classification drives representations
        self.rate1 = config.model.get('rate1', 0.8)
        self.rate2 = config.model.get('rate2', 0.5)
        self.ct_loss = config.model.get('ct_loss', 1)
        self.distill_detach = config.model.get('distill_detach', True)
        self.distill_warmup_epochs = config.model.get('distill_warmup_epochs', 5)
        self.distill_temp = config.model.get('distill_temp', 2.0)
        self.use_legacy_loss = config.model.get('use_legacy_loss', False)
        self.edl_weight = config.model.get('edl_weight', 0.5)
        self.edl_anneal_epochs = config.model.get('edl_anneal_epochs', 10)
        self.epoch = 0  # set by the trainer; anneals distillation and evidential KL term
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(2.))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, fmri_feature, fmri_con, smri_feature, avail=None):
        cls, cls1, contrastive_loss = self.fuse_encoder(smri_feature, fmri_feature, fmri_con, avail)
        self.contrastive_loss = contrastive_loss
        return cls, cls1
    
    def loss_function(self, pred1, label, pred2):
        label = label.float()
        
        # Primary supervised cross-entropy losses for both heads
        l1 = F.cross_entropy(pred1, label)
        l2 = F.cross_entropy(pred2, label)

        if self.use_legacy_loss:
            # Legacy formula preserved for ablation / baseline reproduction
            scaled_p1 = pred1 * self.rate2
            scaled_p2 = pred2 * self.rate2
            t2, t1 = F.softmax(scaled_p2, dim=-1), F.softmax(scaled_p1, dim=-1)
            if self.distill_detach:
                t1, t2 = t1.detach(), t2.detach()
            l3 = F.cross_entropy(scaled_p1, t2)
            l4 = F.cross_entropy(scaled_p2, t1)
            loss = self.rate1 * (l1 + l2) + (1.0 - self.rate1) * (l3 + l4) + self.contrastive_loss * self.ct_loss
        else:
            # Temperature-scaled KL divergence mutual distillation with classification warmup
            t = self.distill_temp
            log_p1 = F.log_softmax(pred1 / t, dim=-1)
            log_p2 = F.log_softmax(pred2 / t, dim=-1)
            p1_soft = F.softmax(pred1 / t, dim=-1)
            p2_soft = F.softmax(pred2 / t, dim=-1)

            if self.distill_detach:
                p1_soft = p1_soft.detach()
                p2_soft = p2_soft.detach()

            # KL divergence: p1 learns from p2, p2 learns from p1
            l3 = F.kl_div(log_p1, p2_soft, reduction='batchmean') * (t ** 2)
            l4 = F.kl_div(log_p2, p1_soft, reduction='batchmean') * (t ** 2)

            # Warmup: ramp distillation weight from 0 up to (1 - rate1) over distill_warmup_epochs
            if self.distill_warmup_epochs > 0:
                warmup_factor = min(1.0, float(self.epoch) / max(1, self.distill_warmup_epochs))
            else:
                warmup_factor = 1.0

            distill_w = (1.0 - self.rate1) * warmup_factor
            cls_w = 1.0 - distill_w

            loss = cls_w * (l1 + l2) + distill_w * (l3 + l4) + self.contrastive_loss * self.ct_loss

        gate = self.fuse_encoder.gate
        if gate is not None:
            loss = loss + self.edl_weight * gate.loss(label, min(1.0, self.epoch / max(1, self.edl_anneal_epochs)))
        return loss


