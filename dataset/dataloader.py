import os
import json
import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
from .data_argument import data_aug, random_rotate_3d
import random
from pathlib import Path


class StochasticAugmentedDataset(utils.Dataset):
    """
    Training-only stochastic augmentation dataset for CDCo-MambaNet V4 Phase 2.
    Preserves validation and test set quarantine; applies dynamic on-the-fly random 3D rotations,
    subtle Gaussian noise, and connectome edge dropout on the training split only.
    """
    def __init__(self, timeseries, pearson, labels, smri, rot_range=(-5.0, 5.0), edge_drop=0.05, noise_sigma=0.01, epoch_multiplier=4, seed=0):
        self.timeseries = timeseries
        self.pearson = pearson
        self.labels = labels
        self.smri = smri
        self.rot_range = rot_range
        self.edge_drop = edge_drop
        self.noise_sigma = noise_sigma
        self.epoch_multiplier = epoch_multiplier
        self.base_len = len(labels)
        self.rng = random.Random(seed)

    def __len__(self):
        return self.base_len * self.epoch_multiplier

    def __getitem__(self, idx):
        real_idx = idx % self.base_len
        t = self.timeseries[real_idx].clone()
        p = self.pearson[real_idx].clone()
        y = self.labels[real_idx].clone()
        s = self.smri[real_idx].clone()

        # On-the-fly randomized 3D rotation within rot_range (e.g. ±5 degrees)
        if self.rot_range and self.rng.random() > 0.3:
            s = random_rotate_3d(s, angles_range=self.rot_range)

        # On-the-fly subtle Gaussian noise
        if self.noise_sigma > 0 and self.rng.random() > 0.5:
            s = s + self.noise_sigma * torch.randn_like(s)

        # On-the-fly mild connectome edge dropout (p=0.05)
        if self.edge_drop > 0 and self.rng.random() > 0.5:
            mask = (torch.rand_like(p) > self.edge_drop).float()
            p = p * mask

        return t, p, y, s


def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    dataset = utils.TensorDataset(
        final_timeseires[:train_length+val_length+test_length],
        final_pearson[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length])
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               smri: torch.tensor,
                               )-> List[utils.DataLoader]:
    
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    exact_splits_path = cfg.dataset.get('exact_splits_path', None)
    if exact_splits_path and os.path.exists(exact_splits_path):
        import json
        with open(exact_splits_path, 'r') as f:
            splits_data = json.load(f)
        train_index = np.array(splits_data["train"]["indices"])
        valid_index = np.array(splits_data["val"]["indices"])
        test_index = np.array(splits_data["test"]["indices"])
        
        final_timeseires_train = final_timeseires[train_index]
        final_pearson_train = final_pearson[train_index]
        labels_train = labels[train_index]
        smri_train = smri[train_index]

        final_timeseires_val = final_timeseires[valid_index]
        final_pearson_val = final_pearson[valid_index]
        labels_val = labels[valid_index]
        smri_val = smri[valid_index]

        final_timeseires_test = final_timeseires[test_index]
        final_pearson_test = final_pearson[test_index]
        labels_test = labels[test_index]
        smri_test = smri[test_index]
    else:
        seed = cfg.get('seed', 0)
        split = StratifiedShuffleSplit(n_splits=1, test_size=val_length + test_length, random_state=seed)
        for train_index, test_valid_index in split.split(final_timeseires, labels):
            final_timeseires_train = final_timeseires[train_index]
            final_pearson_train = final_pearson[train_index]
            labels_train = labels[train_index]
            smri_train = smri[train_index]

            final_timeseires_val_test = final_timeseires[test_valid_index]
            final_pearson_val_test = final_pearson[test_valid_index]
            labels_val_test = labels[test_valid_index]
            smri_val_test = smri[test_valid_index]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=seed)
        for test_index, valid_index in split2.split(final_timeseires_val_test, labels_val_test):
            final_timeseires_test = final_timeseires_val_test[test_index]
            final_pearson_test = final_pearson_val_test[test_index]
            labels_test = labels_val_test[test_index]
            smri_test = smri_val_test[test_index]

            final_timeseires_val = final_timeseires_val_test[valid_index]
            final_pearson_val = final_pearson_val_test[valid_index]
            labels_val = labels_val_test[valid_index]
            smri_val = smri_val_test[valid_index]

    if cfg.dataset.expand:
        cache = cfg.dataset.get("aug_cache") or str(Path(cfg.dataset.path).with_name("train_aug_smri.npy"))  # disk-backed 4x sMRI set
        final_timeseires_train, final_pearson_train, labels_train, smri_train = data_aug(final_timeseires_train, final_pearson_train, labels_train, smri_train, cache)
        with open_dict(cfg):
            cfg.steps_per_epoch = (len(labels_train) - 1) // cfg.dataset.batch_size + 1
            accum_steps = max(1, cfg.training.get('accumulate_grad_batches', 1))
            opt_steps_per_epoch = (cfg.steps_per_epoch // accum_steps) + (1 if cfg.steps_per_epoch % accum_steps != 0 else 0)
            cfg.total_optimizer_steps = opt_steps_per_epoch * cfg.training.epochs
            cfg.total_steps = cfg.total_optimizer_steps
        train_dataset = utils.TensorDataset(
            final_timeseires_train,
            final_pearson_train,
            labels_train,
            smri_train,
        )
    elif cfg.dataset.get('stochastic_aug', False):
        rot_range = tuple(cfg.dataset.get('aug_rotation_range', [-5.0, 5.0]))
        edge_drop = float(cfg.dataset.get('aug_fc_edge_dropout', 0.05))
        noise_sigma = float(cfg.dataset.get('aug_smri_noise_sigma', 0.01))
        epoch_mult = int(cfg.dataset.get('dynamic_epoch_multiplier', 4))
        seed = int(cfg.get('seed', 0))
        train_dataset = StochasticAugmentedDataset(
            final_timeseires_train, final_pearson_train, labels_train, smri_train,
            rot_range=rot_range, edge_drop=edge_drop, noise_sigma=noise_sigma,
            epoch_multiplier=epoch_mult, seed=seed
        )
        with open_dict(cfg):
            cfg.steps_per_epoch = (len(train_dataset) - 1) // cfg.dataset.batch_size + 1
            accum_steps = max(1, cfg.training.get('accumulate_grad_batches', 1))
            opt_steps_per_epoch = (cfg.steps_per_epoch // accum_steps) + (1 if cfg.steps_per_epoch % accum_steps != 0 else 0)
            cfg.total_optimizer_steps = opt_steps_per_epoch * cfg.training.epochs
            cfg.total_steps = cfg.total_optimizer_steps
    else:
        with open_dict(cfg):
            cfg.steps_per_epoch = (len(labels_train) - 1) // cfg.dataset.batch_size + 1
            accum_steps = max(1, cfg.training.get('accumulate_grad_batches', 1))
            opt_steps_per_epoch = (cfg.steps_per_epoch // accum_steps) + (1 if cfg.steps_per_epoch % accum_steps != 0 else 0)
            cfg.total_optimizer_steps = opt_steps_per_epoch * cfg.training.epochs
            cfg.total_steps = cfg.total_optimizer_steps
        train_dataset = utils.TensorDataset(
            final_timeseires_train,
            final_pearson_train,
            labels_train,
            smri_train,
        )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val, smri_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test, smri_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.val_batch_size, shuffle=False, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.test_batch_size, shuffle=False, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
