from utils import TotalMeter, count_params
import json
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from utils import continus_mixup_data
try:
    import wandb
except ImportError:
    wandb = None
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from components import LRScheduler
import logging
from .metrics import calculate_metrics

CONDITIONS = {'clean': None, 'fmri_noise_0.5': ('f', 0.5), 'fmri_noise_1.0': ('f', 1.0),
              'smri_noise_0.5': ('s', 0.5), 'smri_noise_1.0': ('s', 1.0),
              'fmri_missing': ('f', None), 'smri_missing': ('s', None)}


@torch.no_grad()
def robustness(model, loader, conds=CONDITIONS):
    """Test metrics under fixed corruptions, on whatever device the model is on; with the gate, also
    mean reliability r, fusion weight w and kept dims k as [r_s, r_f, w_s, w_f, k_s, k_f], plus
    per-subject values. Noise is seeded per condition, so a subset of conditions draws the same noise."""
    dev = next(model.parameters()).device
    model.eval()
    gate, out = model.fuse_encoder.gate, {}
    for name, cond in conds.items():
        g = torch.Generator(device=dev).manual_seed(0)
        preds, labels, rel = [], [], []
        for f, c, y, s in loader:
            f, c, s = f.to(dev), c.to(dev), s.to(dev)
            avail = torch.ones(len(f), 2, dtype=torch.bool, device=dev)
            if cond:
                m, sigma = cond
                x = f if m == 'f' else s
                noise = torch.randn(x.shape, generator=g, device=dev)
                x = noise if sigma is None else x + sigma * noise  # missing = N(0,1), as in corrupt()
                avail[:, 1 if m == 'f' else 0] = sigma is not None
                f, s = (x, s) if m == 'f' else (f, x)
            _, p1 = model(f, c, s, avail)
            preds += p1.tolist(); labels += y.tolist()
            if gate is not None:
                i = gate.info
                rel.append(torch.cat([i['reliability'], i['weight'], i['dims']], 1).cpu())
        acc, spe, sen, f1, auc = calculate_metrics(preds, labels)
        out[name] = dict(ACC=acc, SPE=spe, SEN=sen, F1=f1, AUC=auc)
        probs = F.softmax(torch.tensor(preds), 1)[:, -1].tolist()
        if rel:
            rel = torch.cat(rel)
            out[name]['mean_r_w_k'] = rel.mean(0).tolist()
            if name == 'clean':
                out[name]['subjects'] = [dict(label=int(np.argmax(l)), prob=p, r_w_k=v)
                                         for l, p, v in zip(labels, probs, rel.tolist())]
        out[name]['probs'] = probs
    return out


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.loss = torch.nn.CrossEntropyLoss()

        self.init_meters()

    def init_meters(self):
        self.train_accuracy1 = None
        self.val_accuracy1 = None
        self.test_accuracy1 = None

        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]
        
    def reset_meters(self):
        self.train_accuracy1 = 0.
        self.val_accuracy1 = 0.
        self.test_accuracy1 = 0.

        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()
       

    def corrupt(self, f, s):
        """Noisy / missing-modality training: per sample, drop one modality (flagged in avail, never
        both) or add Gaussian noise of random strength. avail columns: (sMRI, fMRI). A missing input
        is N(0, 1) noise, not zeros: inputs are z-scored, so it carries no subject information, and an
        all-zero fMRI map hits the ConvNeXt LayerNorm/GRN singularities (NaN gradients upstream too)."""
        t = self.config.training
        p_miss, p_noise = t.get('corrupt_p_missing', 0.0), t.get('corrupt_p_noise', 0.0)
        if p_miss == 0 and p_noise == 0:
            return f, s, None
        b, dev = f.shape[0], f.device
        u = torch.rand(b, device=dev)
        avail = torch.stack([u >= p_miss, (u < p_miss) | (u >= 2 * p_miss)], 1)
        out = []
        for x, ok in ((f, avail[:, 1]), (s, avail[:, 0])):
            shape = (b,) + (1,) * (x.dim() - 1)
            sigma = (torch.rand(shape, device=dev) < p_noise).float() * torch.rand(shape, device=dev) * t.corrupt_max_sigma
            out.append(torch.where(ok.view(shape), x + sigma * torch.randn_like(x), torch.randn_like(x)))
        return out[0], out[1], avail

    def robustness(self):
        return robustness(self.model, self.test_dataloader)

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        pred_list = []
        pred_list1 = []
        label_list = []
        device = next(self.model.parameters()).device
        accum_steps = max(1, self.config.training.get('accumulate_grad_batches', 1))
        optimizer.zero_grad()
        max_b = self.config.training.get('max_batches', None)
        num_batches = len(self.train_dataloader) if not max_b else min(max_b, len(self.train_dataloader))

        for batch_idx, (fMRI_Feature, fMRI_Con, label, sMRI_Feature) in enumerate(self.train_dataloader):
            if max_b is not None and batch_idx >= max_b:
                break
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            fMRI_Feature = fMRI_Feature.to(device)
            fMRI_Con = fMRI_Con.to(device)
            label = label.to(device)
            sMRI_Feature = sMRI_Feature.to(device)
            
            # Configurable mixup (can be disabled or set to mild alpha)
            if self.config.preprocess.get('continus', False):
                alpha = self.config.preprocess.get('alpha', 0.2)
                fMRI_Feature, sMRI_Feature, label = continus_mixup_data(
                    fMRI_Feature, sMRI_Feature, y=label, alpha=alpha, device=str(device)
                )

            fMRI_Feature, sMRI_Feature, avail = self.corrupt(fMRI_Feature, sMRI_Feature)
            predict, predict1 = self.model(fMRI_Feature, fMRI_Con, sMRI_Feature, avail)
            loss = self.model.loss_function(predict, label, predict1)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN or Inf loss encountered at batch {batch_idx + 1} (loss={loss.item()})")

            self.train_loss.update_with_weight_loss(loss.item(), label.shape[0])

            # Gradient accumulation for low-VRAM GPUs (e.g., 2 GB MX450)
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad()

            log_interval = 5 if max_b else 50
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches or batch_idx == 0:
                vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                self.logger.info(f"Epoch {getattr(self, 'epoch', 0)} | Batch {batch_idx + 1}/{num_batches} | Step Loss: {loss.item():.4f} | Avg Loss: {self.train_loss.avg:.4f} | Peak VRAM: {vram_mb:.1f} MB")

            # Primary clean head in pred_list, secondary in pred_list1
            pred_list.extend(predict.tolist())
            pred_list1.extend(predict1.tolist())
            label_list.extend(label.tolist())

        self.train_accuracy.update_with_weight_acc(pred_list, label_list)
        self.train_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
        self.train_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)

    @torch.no_grad()
    def test_per_epoch(self, dataloader, loss_meter, str_select):
        labels = []
        result = []
        label_list = []
        pred_list = []
        pred_list1 = []
        device = next(self.model.parameters()).device
        self.model.eval()
        for fMRI_Feature, fMRI_Con, label, sMRI_Feature in dataloader:
            fMRI_Feature = fMRI_Feature.to(device)
            fMRI_Con = fMRI_Con.to(device)
            label = label.to(device)
            sMRI_Feature = sMRI_Feature.to(device)

            output, output1 = self.model(fMRI_Feature, fMRI_Con, sMRI_Feature)
            label = label.float()
            loss = self.model.loss_function(output, label, output1)
            loss_meter.update_with_weight_loss(
                loss.item(), label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

            # Primary clean head is evaluated as main metric (pred_list)
            pred_list.extend(output.tolist())
            pred_list1.extend(output1.tolist())

            label_list.extend(label.tolist())

        if str_select == "val":
            self.val_accuracy.update_with_weight_acc(pred_list, label_list)
            self.val_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
            self.val_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)
        else:
            self.test_accuracy.update_with_weight_acc(pred_list, label_list)
            self.test_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
            self.test_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)

        accuracy, specificity, sensitivity, f1, auc = calculate_metrics(pred_list, label_list)
        accuracy1, specificity1, sensitivity1, f11, auc1 = calculate_metrics(pred_list1, label_list)
        return [accuracy, specificity, sensitivity, f1, auc], [accuracy1, specificity1, sensitivity1, f11, auc1], pred_list, pred_list1, label_list

    def save_result(self, results, epoch):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"test_max_metrics.npy",
                results, allow_pickle=True)
        np.save(self.save_path/"best_val_metrics.npy",
                results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_path/"model.npy")
        torch.save(self.model.state_dict(), self.save_path/"best_model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        
        max_val_accuracy = float('-inf')
        best_val_epoch = 0
        best_epoch_metrics = {}
        early_stopping_patience = self.config.training.get('early_stopping_patience', None)

        for epoch in range(self.epochs):
            self.model.epoch = epoch
            self.epoch = epoch
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result, val_result1, pred_list_val, pred_list_val1, label_list_val = self.test_per_epoch(self.val_dataloader,
                                            self.val_loss, "val")

            test_result, test_result1, pred_list_test, pred_list_test1, label_list_test = self.test_per_epoch(self.test_dataloader,
                                            self.test_loss, "test")
            
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .5f}',
                f'Train Accuracy:{self.train_accuracy.final: .5f}',
                f'Train Acc:{self.train_mse: .5f}',
                f'Val Loss:{self.val_loss.avg: .5f}',
                f'Val Accuracy:{self.val_accuracy.final: .5f}',
                f'Val Acc:{self.val_mse: .5f}',
                f'Test Loss:{self.test_loss.avg: .5f}',
                f'Test Accuracy:{self.test_accuracy.final: .5f}',
                f'Test Acc:{self.test_mse: .5f}',
                f'Peak VRAM: {vram_mb:.1f} MB',
            ]))

            # Checkpoint selection strictly based on VALIDATION accuracy (zero test leakage)
            if self.val_accuracy.final > max_val_accuracy:
                max_val_accuracy = self.val_accuracy.final
                best_val_epoch = epoch
                best_epoch_metrics = {
                    "Epoch": epoch,

                    'Train Loss': self.train_loss.avg,
                    'Train Accuracy': self.train_accuracy.final,

                    'Val Loss': self.val_loss.avg,
                    'Val Accuracy': self.val_accuracy.final,
                    "Val SPE": val_result[1],
                    "Val SEN": val_result[2],
                    "Val F1": val_result[3],
                    "Val AUC": val_result[4],

                    'Test Loss': self.test_loss.avg,
                    'Test Accuracy': self.test_accuracy.final,
                    "Test SPE": test_result[1],
                    "Test SEN": test_result[2],
                    "Test F1": test_result[3],
                    "Test AUC": test_result[4],
                }
                # Save checkpoint when validation accuracy improves
                self.save_result(best_epoch_metrics, epoch)

            if wandb is not None and getattr(wandb, 'run', None) is not None:
                wandb.run.summary['Max Val Accuracy'] = max_val_accuracy
                wandb.run.summary.update(best_epoch_metrics)

            training_process.append({
                "Epoch": epoch,

                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.final,
                "Val Loss": self.val_loss.avg,
                "Val Accuracy": self.val_accuracy.final,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.final,
                "Test SPE": test_result[1],
                "Test SEN": test_result[2],
                "Test F1": test_result[3],
                "Test AUC": test_result[4],
            })

            # Early stopping check
            if early_stopping_patience is not None and (epoch - best_val_epoch) >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}: No validation improvement for {early_stopping_patience} epochs (best epoch: {best_val_epoch}, best val acc: {max_val_accuracy:.4f}).")
                break

        # Save final epoch model separately
        self.save_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), self.save_path / "final_epoch_model.pt")

        test_at_best_val = [
            best_epoch_metrics.get('Test Accuracy', test_result[0]),
            best_epoch_metrics.get('Test SPE', test_result[1]),
            best_epoch_metrics.get('Test SEN', test_result[2]),
            best_epoch_metrics.get('Test F1', test_result[3]),
            best_epoch_metrics.get('Test AUC', test_result[4]),
        ]

        final = {
            'seed': self.config.get('seed'),
            'best_val_epoch': best_val_epoch,
            'best_val_metrics': best_epoch_metrics,
            'test_at_best_val': {
                'ACC': float(test_at_best_val[0]),
                'SPE': float(test_at_best_val[1]),
                'SEN': float(test_at_best_val[2]),
                'F1': float(test_at_best_val[3]),
                'AUC': float(test_at_best_val[4]),
            },
            'test_last_epoch': dict(zip(['ACC', 'SPE', 'SEN', 'F1', 'AUC'], map(float, test_result))),
            'model': {k: self.config.model.get(k) for k in (
                 'reliability_fusion', 'dynamic_dim', 'min_dim', 'nested_dropout', 'edl_weight')},
            'training': dict(self.config.training),
        }
        if self.config.training.get('robustness_eval', True):
            final['robustness'] = self.robustness()
        json.dump(final, open(self.save_path / 'final.json', 'w'), indent=1, default=float)
        return test_at_best_val