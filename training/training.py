from utils import TotalMeter, count_params
import json
import os
import shutil
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from utils import continus_mixup_data
try:
    import wandb
except ImportError:
    wandb = None
from omegaconf import DictConfig, OmegaConf
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
        self.current_step = 0
        self.optimizer_step = 0
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_path.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(self.config, self.save_path / "config.yaml")
        self.save_learnable_graph = cfg.save_learnable_graph

        if self.config.training.get('resume_checkpoint'):
            ckpt_path = self.config.training.resume_checkpoint
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path))
                self.logger.info(f"Successfully resumed model weights from: {ckpt_path}")
                shutil.copy2(ckpt_path, self.save_path / "best_model.pt")
                shutil.copy2(ckpt_path, self.save_path / "model.npy")
            else:
                self.logger.warning(f"Resume checkpoint not found: {ckpt_path}")

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

            if batch_idx == 0 and getattr(self, 'epoch', 0) == 0:
                fc_enc = getattr(getattr(self.model, 'fuse_encoder', None), 'fc_encoder', None)
                if fc_enc is not None:
                    fc_grads = [p.grad.abs().sum().item() for p in fc_enc.parameters() if p.grad is not None]
                    active_fc_params = sum(g > 0 for g in fc_grads)
                    self.logger.info(f"[VERIFIED GRADIENT FLOW] FC GNN Branch: {active_fc_params}/{len(list(fc_enc.parameters()))} parameters actively receiving non-zero gradients.")

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.optimizer_step += 1
                lr_scheduler.update(optimizer=optimizer, step=self.optimizer_step)

                scaler = getattr(self, 'scaler', None)
                if scaler is not None:
                    scaler.unscale_(optimizer)

                clip_norm = self.config.training.get('clip_grad_norm', None)
                if clip_norm is not None and float(clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_norm))

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
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

            # Record predictions from both heads
            pred_list.extend(output.tolist())
            pred_list1.extend(output1.tolist())
            label_list.extend(label.tolist())

        # Compute ensemble predictions as average of softmax probabilities
        p0 = F.softmax(torch.tensor(pred_list, dtype=torch.float32), dim=-1).numpy()
        p1 = F.softmax(torch.tensor(pred_list1, dtype=torch.float32), dim=-1).numpy()
        p_ens = 0.5 * (p0 + p1)

        m0 = calculate_metrics(pred_list, label_list, return_dict=True)
        m1 = calculate_metrics(pred_list1, label_list, return_dict=True)
        m_ens = calculate_metrics(p_ens, label_list, return_dict=True)

        if str_select == "val":
            self.val_accuracy.update_with_weight_acc(pred_list, label_list)
            self.val_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
            self.val_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)
        else:
            self.test_accuracy.update_with_weight_acc(pred_list, label_list)
            self.test_accuracy1, _, _, _, _ = calculate_metrics(pred_list, label_list)
            self.test_mse, _, _, _, _ = calculate_metrics(pred_list1, label_list)

        return m0, m1, m_ens, pred_list, pred_list1, p_ens.tolist(), label_list

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
        self.optimizer_step = 0
        
        best_val_epoch = 0
        best_epoch_metrics = {}
        early_stopping_patience = self.config.training.get('early_stopping_patience', None)
        no_improvement_count = 0
        checkpoint_metric = self.config.training.get('checkpoint_metric', 'val_acc_ensemble')
        max_val_metric = float('-inf')
        best_val_tie_breaker = float('-inf')

        self.logger.info(f"Checkpoint Selection Metric: {checkpoint_metric} (Strictly Validation-Only)")

        if self.config.training.get('resume_checkpoint'):
            metrics_path = Path(self.config.training.resume_checkpoint).parent / "best_val_metrics.npy"
            if metrics_path.exists():
                try:
                    prev_metrics = np.load(metrics_path, allow_pickle=True).item()
                    max_val_metric = float(prev_metrics.get('Selection_Score', prev_metrics.get('Val Accuracy', float('-inf'))))
                    best_epoch_metrics = prev_metrics
                    best_val_epoch = int(prev_metrics.get('Epoch', 1)) - 1
                    self.logger.info(f"Initialized max_val_metric to {max_val_metric:.4f} from previous best_val_metrics (Epoch {best_val_epoch+1})")
                except Exception as e:
                    self.logger.warning(f"Could not load previous best_val_metrics: {e}")

        for epoch in range(self.epochs):
            self.model.epoch = epoch
            self.epoch = epoch
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_h1, val_h2, val_ens, pred_list_val, pred_list_val1, pred_list_val_ens, label_list_val = self.test_per_epoch(
                self.val_dataloader, self.val_loss, "val"
            )
            
            # Extract validation-only selection score based on configured checkpoint metric
            if checkpoint_metric == 'val_loss':
                current_score = -float(self.val_loss.avg)
                tie_breaker = val_ens['accuracy']
            elif checkpoint_metric == 'val_acc_ensemble':
                current_score = val_ens['accuracy']
                tie_breaker = val_ens['balanced_accuracy'] + 0.01 * val_ens['auc']
            elif checkpoint_metric == 'val_bacc_ensemble':
                current_score = val_ens['balanced_accuracy']
                tie_breaker = val_ens['auc']
            elif checkpoint_metric == 'val_acc_head2':
                current_score = val_h2['accuracy']
                tie_breaker = val_h2['balanced_accuracy'] + 0.01 * val_h2['auc']
            elif checkpoint_metric == 'val_acc_head1':
                current_score = val_h1['accuracy']
                tie_breaker = val_h1['balanced_accuracy'] + 0.01 * val_h1['auc']
            else:
                current_score = val_ens['accuracy']
                tie_breaker = val_ens['balanced_accuracy']

            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            self.logger.info(" | ".join([
                f'Epoch[{epoch+1}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg:.4f}',
                f'Train Acc:{self.train_accuracy.final*100:.2f}%',
                f'Val Loss:{self.val_loss.avg:.4f}',
                f'Val H1 Acc:{val_h1["accuracy"]*100:.2f}% (BAcc: {val_h1["balanced_accuracy"]*100:.2f}%)',
                f'Val H2 Acc:{val_h2["accuracy"]*100:.2f}% (BAcc: {val_h2["balanced_accuracy"]*100:.2f}%)',
                f'Val Ens Acc:{val_ens["accuracy"]*100:.2f}% (BAcc: {val_ens["balanced_accuracy"]*100:.2f}%, AUC: {val_ens["auc"]*100:.2f}%)',
                f'Peak VRAM: {vram_mb:.1f} MB',
            ]))

            # Checkpoint selection strictly based on VALIDATION metric (zero test leakage)
            is_improvement = False
            if current_score > max_val_metric:
                is_improvement = True
            elif current_score == max_val_metric and tie_breaker > best_val_tie_breaker:
                is_improvement = True

            if is_improvement:
                max_val_metric = current_score
                best_val_tie_breaker = tie_breaker
                best_val_epoch = epoch
                no_improvement_count = 0
                best_epoch_metrics = {
                    "Epoch": epoch + 1,
                    "Selection_Metric": checkpoint_metric,
                    "Selection_Score": float(current_score),
                    "Train Loss": float(self.train_loss.avg),
                    "Train Accuracy": float(self.train_accuracy.final),
                    "Val Loss": float(self.val_loss.avg),
                    "Head1": val_h1,
                    "Head2": val_h2,
                    "Ensemble": val_ens,
                    # Backward compatibility keys:
                    "Val Accuracy": float(val_ens['accuracy']),
                    "Val SPE": float(val_ens['specificity']),
                    "Val SEN": float(val_ens['sensitivity']),
                    "Val F1": float(val_ens['f1']),
                    "Val AUC": float(val_ens['auc']),
                }
                self.save_result(best_epoch_metrics, epoch)
                score_str = f"{-current_score:.4f}" if checkpoint_metric == 'val_loss' else f"{current_score*100:.2f}%"
                self.logger.info(f"-> Saved new best checkpoint at Epoch {epoch+1} ({checkpoint_metric}={score_str})")
            else:
                no_improvement_count += 1

            if wandb is not None and getattr(wandb, 'run', None) is not None:
                wandb.run.summary['Max Val Metric'] = max_val_metric
                wandb.run.summary.update(best_epoch_metrics)

            training_process.append({
                "Epoch": epoch + 1,
                "Train Loss": float(self.train_loss.avg),
                "Train Accuracy": float(self.train_accuracy.final),
                "Val Loss": float(self.val_loss.avg),
                "Head1": val_h1,
                "Head2": val_h2,
                "Ensemble": val_ens,
            })
            # Save progress incrementally on each epoch
            self.save_path.mkdir(exist_ok=True, parents=True)
            json.dump(training_process, open(self.save_path / "epoch_history.json", "w"), indent=2, default=float)

            # Early stopping check (enforced strictly on validation metric)
            if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}: No validation improvement for {early_stopping_patience} consecutive epochs (best epoch: {best_val_epoch+1}, best {checkpoint_metric}: {max_val_metric*100:.2f}%).")
                break

        # Save final epoch model separately
        self.save_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), self.save_path / "final_epoch_model.pt")
        json.dump(training_process, open(self.save_path / "epoch_history.json", "w"), indent=2)

        # -------------------------------------------------------------
        # POST-TRAINING HELD-OUT TEST EVALUATION (After Checkpoint Selection)
        # The test set is evaluated ONLY now using the saved checkpoints.
        # -------------------------------------------------------------
        self.logger.info(f"=== HELD-OUT TEST EVALUATION (Best Model from Epoch {best_val_epoch+1}, selected by {checkpoint_metric}) ===")
        best_model_path = self.save_path / "best_model.pt"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
        self.test_loss.reset()
        t_h1, t_h2, t_ens, _, _, _, _ = self.test_per_epoch(
            self.test_dataloader, self.test_loss, "test"
        )
        self.logger.info("Best Validation Model on Held-Out Test Set (Ensemble): " + " | ".join([
            f"Test Loss: {self.test_loss.avg:.5f}",
            f"ACC: {t_ens['accuracy']*100:.2f}%",
            f"BACC: {t_ens['balanced_accuracy']*100:.2f}%",
            f"SPE: {t_ens['specificity']*100:.2f}%",
            f"SEN: {t_ens['sensitivity']*100:.2f}%",
            f"F1: {t_ens['f1']*100:.2f}%",
            f"AUC: {t_ens['auc']*100:.2f}%",
        ]))
        self.logger.info(f"Test Confusion Matrix (Ensemble): {t_ens['confusion_matrix']}")
        self.logger.info(f"Test Set Head 1: ACC={t_h1['accuracy']*100:.2f}%, BACC={t_h1['balanced_accuracy']*100:.2f}%, AUC={t_h1['auc']*100:.2f}%, CM={t_h1['confusion_matrix']}")
        self.logger.info(f"Test Set Head 2: ACC={t_h2['accuracy']*100:.2f}%, BACC={t_h2['balanced_accuracy']*100:.2f}%, AUC={t_h2['auc']*100:.2f}%, CM={t_h2['confusion_matrix']}")

        # Evaluate Final Epoch Model for comparison
        self.logger.info(f"=== HELD-OUT TEST EVALUATION (Final Epoch Model) ===")
        self.model.load_state_dict(torch.load(self.save_path / "final_epoch_model.pt"))
        self.test_loss.reset()
        fin_h1, fin_h2, fin_ens, _, _, _, _ = self.test_per_epoch(
            self.test_dataloader, self.test_loss, "test"
        )
        self.logger.info("Final Epoch Model on Held-Out Test Set (Ensemble): " + " | ".join([
            f"Test Loss: {self.test_loss.avg:.5f}",
            f"ACC: {fin_ens['accuracy']*100:.2f}%",
            f"BACC: {fin_ens['balanced_accuracy']*100:.2f}%",
            f"SPE: {fin_ens['specificity']*100:.2f}%",
            f"SEN: {fin_ens['sensitivity']*100:.2f}%",
            f"F1: {fin_ens['f1']*100:.2f}%",
            f"AUC: {fin_ens['auc']*100:.2f}%",
        ]))

        final = {
            'seed': self.config.get('seed'),
            'checkpoint_metric': checkpoint_metric,
            'best_val_epoch': best_val_epoch + 1,
            'best_val_metrics': best_epoch_metrics,
            'test_at_best_val': {
                'ensemble': t_ens,
                'head1': t_h1,
                'head2': t_h2,
                'accuracy': t_ens['accuracy'],
                'balanced_accuracy': t_ens['balanced_accuracy'],
                'specificity': t_ens['specificity'],
                'sensitivity': t_ens['sensitivity'],
                'f1': t_ens['f1'],
                'auc': t_ens['auc'],
                'confusion_matrix': t_ens['confusion_matrix'],
            },
            'test_at_final_epoch': {
                'ensemble': fin_ens,
                'head1': fin_h1,
                'head2': fin_h2,
                'accuracy': fin_ens['accuracy'],
                'balanced_accuracy': fin_ens['balanced_accuracy'],
                'specificity': fin_ens['specificity'],
                'sensitivity': fin_ens['sensitivity'],
                'f1': fin_ens['f1'],
                'auc': fin_ens['auc'],
                'confusion_matrix': fin_ens['confusion_matrix'],
            },
            'model': {k: self.config.model.get(k) for k in (
                 'reliability_fusion', 'dynamic_dim', 'min_dim', 'nested_dropout', 'edl_weight', 'head1_regularization')},
            'training': dict(self.config.training),
        }
        if self.config.training.get('robustness_eval', False):
            self.model.load_state_dict(torch.load(best_model_path))
            final['robustness'] = self.robustness()
        json.dump(final, open(self.save_path / 'final.json', 'w'), indent=2, default=float)
        
        return [
            t_ens['accuracy'],
            t_ens['specificity'],
            t_ens['sensitivity'],
            t_ens['f1'],
            t_ens['auc'],
        ]