# CDCo-MambaNet V4 — Phase 2 Final Code Diff

**Author:** Senior Deep Learning Research Engineer  
**Date:** November 11, 2026  
**Scope:** Controlled Phase 2 Source Code Modifications  
**Comparison Baseline:** Pre-modification backup files (`*.bak`) created on November 11, 2026  

---

## 1. Summary of Changes

| File | Lines Changed | Primary Objective |
| :--- | :--- | :--- |
| `components/lr_scheduler.py` | +1, -1 | Enable optimizer-update-based scheduling via `total_optimizer_steps` priority lookup |
| `training/training.py` | +21, -2 | Decouple scheduler from micro-batch loop, advance once per optimizer update, insert gradient clipping (`max_norm=1.0`) with AMP unscale safety |
| `dataset/dataloader.py` | +84, -7 | Add `StochasticAugmentedDataset` for E5 training-only augmentation, update step calculations based on optimizer cadence |

---

## 2. Unified Diff: `components/lr_scheduler.py`

```diff
--- components/lr_scheduler.py.bak
+++ components/lr_scheduler.py
@@ -22,7 +22,7 @@ class LRScheduler:
 
         warm_up_from = lr_config.warm_up_from
         warm_up_steps = lr_config.warm_up_steps
-        total_steps = self.training_config.total_steps
+        total_steps = lr_config.get('total_optimizer_steps', None) or self.training_config.get('total_optimizer_steps', None) or self.training_config.total_steps
 
         # assert 0 <= step <= total_steps
         if step < warm_up_steps:
```

---

## 3. Unified Diff: `training/training.py`

```diff
--- training/training.py.bak
+++ training/training.py
@@ -78,6 +78,8 @@ class Train:
         self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
         self.epochs = cfg.training.epochs
         self.total_steps = cfg.total_steps
+        self.current_step = 0
+        self.optimizer_step = 0
         self.optimizers = optimizers
         self.lr_schedulers = lr_schedulers
         self.save_path = Path(cfg.log_path) / cfg.unique_id
@@ -157,7 +159,6 @@ class Train:
             if max_b is not None and batch_idx >= max_b:
                 break
             self.current_step += 1
-            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
             fMRI_Feature = fMRI_Feature.to(device)
             fMRI_Con = fMRI_Con.to(device)
             label = label.to(device)
@@ -191,7 +192,22 @@ class Train:
                     self.logger.info(f"[VERIFIED GRADIENT FLOW] FC GNN Branch: {active_fc_params}/{len(list(fc_enc.parameters()))} parameters actively receiving non-zero gradients.")
 
             if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
-                optimizer.step()
+                self.optimizer_step += 1
+                lr_scheduler.update(optimizer=optimizer, step=self.optimizer_step)
+
+                scaler = getattr(self, 'scaler', None)
+                if scaler is not None:
+                    scaler.unscale_(optimizer)
+
+                clip_norm = self.config.training.get('clip_grad_norm', None)
+                if clip_norm is not None and float(clip_norm) > 0:
+                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_norm))
+
+                if scaler is not None:
+                    scaler.step(optimizer)
+                    scaler.update()
+                else:
+                    optimizer.step()
                 optimizer.zero_grad()
 
             log_interval = 5 if max_b else 50
@@ -264,6 +280,7 @@ class Train:
     def train(self):
         training_process = []
         self.current_step = 0
+        self.optimizer_step = 0
         
         best_val_epoch = 0
         best_epoch_metrics = {}
```

---

## 4. Unified Diff: `dataset/dataloader.py`

```diff
--- dataset/dataloader.py.bak
+++ dataset/dataloader.py
@@ -7,9 +7,55 @@ from typing import List
 from sklearn.model_selection import StratifiedShuffleSplit
 import numpy as np
 import torch.nn.functional as F
-from .data_argument import data_aug
+from .data_argument import data_aug, random_rotate_3d
+import random
 from pathlib import Path
 
+
+class StochasticAugmentedDataset(utils.Dataset):
+    """
+    Training-only stochastic augmentation dataset for CDCo-MambaNet V4 Phase 2.
+    Preserves validation and test set quarantine; applies dynamic on-the-fly random 3D rotations,
+    subtle Gaussian noise, and connectome edge dropout on the training split only.
+    """
+    def __init__(self, timeseries, pearson, labels, smri, rot_range=(-5.0, 5.0), edge_drop=0.05, noise_sigma=0.01, epoch_multiplier=4, seed=0):
+        self.timeseries = timeseries
+        self.pearson = pearson
+        self.labels = labels
+        self.smri = smri
+        self.rot_range = rot_range
+        self.edge_drop = edge_drop
+        self.noise_sigma = noise_sigma
+        self.epoch_multiplier = epoch_multiplier
+        self.base_len = len(labels)
+        self.rng = random.Random(seed)
+
+    def __len__(self):
+        return self.base_len * self.epoch_multiplier
+
+    def __getitem__(self, idx):
+        real_idx = idx % self.base_len
+        t = self.timeseries[real_idx].clone()
+        p = self.pearson[real_idx].clone()
+        y = self.labels[real_idx].clone()
+        s = self.smri[real_idx].clone()
+
+        # On-the-fly randomized 3D rotation within rot_range (e.g. ±5 degrees)
+        if self.rot_range and self.rng.random() > 0.3:
+            s = random_rotate_3d(s, angles_range=self.rot_range)
+
+        # On-the-fly subtle Gaussian noise
+        if self.noise_sigma > 0 and self.rng.random() > 0.5:
+            s = s + self.noise_sigma * torch.randn_like(s)
+
+        # On-the-fly mild connectome edge dropout (p=0.05)
+        if self.edge_drop > 0 and self.rng.random() > 0.5:
+            mask = (torch.rand_like(p) > self.edge_drop).float()
+            p = p * mask
+
+        return t, p, y, s
+
+
 def init_dataloader(cfg: DictConfig,
                     final_timeseires: torch.tensor,
                     final_pearson: torch.tensor,
@@ -122,17 +168,48 @@ def init_stratified_dataloader(cfg: DictConfig,
     if cfg.dataset.expand:
         cache = cfg.dataset.get("aug_cache") or str(Path(cfg.dataset.path).with_name("train_aug_smri.npy"))  # disk-backed 4x sMRI set
         final_timeseires_train, final_pearson_train, labels_train, smri_train = data_aug(final_timeseires_train, final_pearson_train, labels_train, smri_train, cache)
-        with open_dict(cfg):  # upstream counted steps before the 4x expansion (cos_lr period ~1/4 epoch)
+        with open_dict(cfg):
             cfg.steps_per_epoch = (len(labels_train) - 1) // cfg.dataset.batch_size + 1
-            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs
-    
-    
-    train_dataset = utils.TensorDataset(
-        final_timeseires_train,
-        final_pearson_train,
-        labels_train,
-        smri_train,
-    )
+            accum_steps = max(1, cfg.training.get('accumulate_grad_batches', 1))
+            opt_steps_per_epoch = (cfg.steps_per_epoch // accum_steps) + (1 if cfg.steps_per_epoch % accum_steps != 0 else 0)
+            cfg.total_optimizer_steps = opt_steps_per_epoch * cfg.training.epochs
+            cfg.total_steps = cfg.total_optimizer_steps
+        train_dataset = utils.TensorDataset(
+            final_timeseires_train,
+            final_pearson_train,
+            labels_train,
+            smri_train,
+        )
+    elif cfg.dataset.get('stochastic_aug', False):
+        rot_range = tuple(cfg.dataset.get('aug_rotation_range', [-5.0, 5.0]))
+        edge_drop = float(cfg.dataset.get('aug_fc_edge_dropout', 0.05))
+        noise_sigma = float(cfg.dataset.get('aug_smri_noise_sigma', 0.01))
+        epoch_mult = int(cfg.dataset.get('dynamic_epoch_multiplier', 4))
+        seed = int(cfg.get('seed', 0))
+        train_dataset = StochasticAugmentedDataset(
+            final_timeseires_train, final_pearson_train, labels_train, smri_train,
+            rot_range=rot_range, edge_drop=edge_drop, noise_sigma=noise_sigma,
+            epoch_multiplier=epoch_mult, seed=seed
+        )
+        with open_dict(cfg):
+            cfg.steps_per_epoch = (len(train_dataset) - 1) // cfg.dataset.batch_size + 1
+            accum_steps = max(1, cfg.training.get('accumulate_grad_batches', 1))
+            opt_steps_per_epoch = (cfg.steps_per_epoch // accum_steps) + (1 if cfg.steps_per_epoch % accum_steps != 0 else 0)
+            cfg.total_optimizer_steps = opt_steps_per_epoch * cfg.training.epochs
+            cfg.total_steps = cfg.total_optimizer_steps
+    else:
+        with open_dict(cfg):
+            cfg.steps_per_epoch = (len(labels_train) - 1) // cfg.dataset.batch_size + 1
+            accum_steps = max(1, cfg.training.get('accumulate_grad_batches', 1))
+            opt_steps_per_epoch = (cfg.steps_per_epoch // accum_steps) + (1 if cfg.steps_per_epoch % accum_steps != 0 else 0)
+            cfg.total_optimizer_steps = opt_steps_per_epoch * cfg.training.epochs
+            cfg.total_steps = cfg.total_optimizer_steps
+        train_dataset = utils.TensorDataset(
+            final_timeseires_train,
+            final_pearson_train,
+            labels_train,
+            smri_train,
+        )
 
     val_dataset = utils.TensorDataset(
         final_timeseires_val, final_pearson_val, labels_val, smri_val
```
