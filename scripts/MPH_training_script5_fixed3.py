#!/usr/bin/env python3
"""
MPH_training_script5_fixed3.py

Final Integrated training script for Mirage Correction (MPH).
Includes all stability fixes, especially forcing deterministic cuDNN behavior.
"""
import os
import argparse
import random
import time
import glob
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Import your model skeleton
from MPH_model_skeleton import MPHModel

# ----------------------------- Helpers: checkpointing ------------------#

def save_checkpoint(path, model, optimizer, scheduler=None, epoch=None, best_val=None, scaler=None, extra=None):
    """Atomic save of checkpoint including optimizer, scheduler, scaler state."""
    tmp = str(path) + ".tmp"
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val': best_val,
    }
    if scheduler is not None:
        try:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        except Exception:
            pass
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    if extra is not None:
        ckpt['extra'] = extra
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device='cuda'):
    """Load checkpoint and optionally restore optimizer/scheduler/scaler states."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    start_epoch = ckpt.get('epoch', 0)
    best_val = ckpt.get('best_val', None)
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # move optimizer state tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception:
            print("Warning: couldn't load scheduler state.")
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return start_epoch, best_val

# ----------------------------- Reproducibility -------------------------#

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------- EarlyStopping --------------------------#

class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.counter = 0

    def step(self, value: float):
        # We track validation loss, so smaller is better.
        if self.best is None:
            self.best = value
            self.counter = 0
            return False
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

# ----------------------------- Dataset -----------------------------#

class SyntheticTurbulenceDataset(Dataset):
    """
    Loads .npz sequences created by MPH_synthetic_dataset.py
    """
    def __init__(self, root: str, file_list: Optional[List[str]] = None,
                 T_train: int = 5, crop_size: int = 128, augment: bool = True):
        self.root = Path(root)
        self.T_train = int(T_train)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)

        if file_list is not None:
            self.files = [Path(p) for p in file_list]
        else:
            self.files = sorted([Path(p) for p in glob.glob(str(self.root / 'seq_*.npz'))])
        if len(self.files) == 0:
            self.files = sorted(list(self.root.glob('*.npz')))
        if len(self.files) == 0:
            print(f"[Warning] No .npz files found in {self.root}")

        self.color_jitter = T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)
        print(f"SyntheticTurbulenceDataset: found {len(self.files)} files. T_train={self.T_train}, crop={self.crop_size}, augment={self.augment}")

    def __len__(self):
        return len(self.files)

    def _apply_framewise_transforms(self, window: np.ndarray) -> np.ndarray:
        """
        Applies consistent spatial transforms across the entire T-frame window.
        """
        Tt, H, W, C = window.shape
        pil_frames = [Image.fromarray((window[i] * 255.0).astype(np.uint8)) for i in range(Tt)]

        # RandomResizedCrop deterministic params from first frame
        i, j, h, w = T.RandomResizedCrop.get_params(pil_frames[0], scale=(0.7, 1.0), ratio=(0.9, 1.1))
        resized = [TF.resized_crop(img, i, j, h, w, (self.crop_size, self.crop_size)) for img in pil_frames]

        # Random horizontal flip (consistent across frames)
        if random.random() < 0.5:
            resized = [TF.hflip(img) for img in resized]

        # Apply same color jitter instance to all frames (approx consistent)
        cj = self.color_jitter
        resized = [cj(img) for img in resized]

        arr = np.stack([np.array(img).astype(np.float32) / 255.0 for img in resized], axis=0)
        return arr

    def _random_crop(self, frames: np.ndarray) -> np.ndarray:
        Tt, H, W, C = frames.shape
        if H == self.crop_size and W == self.crop_size:
            return frames
        top = np.random.randint(0, max(1, H - self.crop_size + 1))
        left = np.random.randint(0, max(1, W - self.crop_size + 1))
        return frames[:, top:top + self.crop_size, left:left + self.crop_size, :]

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(str(path))
        key = 'seq' if 'seq' in data.files else data.files[0]
        seq = data[key].astype(np.float32)
        T_full, H, W, C = seq.shape

        # sample window
        if T_full >= self.T_train:
            start = np.random.randint(0, T_full - self.T_train + 1)
            window = seq[start:start + self.T_train]
        else:
            pad = self.T_train - T_full
            window = np.concatenate([seq, np.repeat(seq[-1:], pad, axis=0)], axis=0)

        # augmentation or crop
        if self.augment:
            window = self._apply_framewise_transforms(window)
        else:
            window = self._random_crop(window)

        # to tensor (T, C, H, W)
        frames = torch.from_numpy(window).permute(0, 3, 1, 2).contiguous()
        gt = frames[self.T_train // 2].clone()  # center frame as pseudo-GT
        return {'frames': frames, 'gt': gt, 'meta': {'path': str(path)}}

# ----------------------------- Losses & simple simulator ----------------------#

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))

def simple_simulate(restored: torch.Tensor, lpd_mu: Optional[torch.Tensor] = None, noise_scale: float = 0.01):
    """
    Small, stable differentiable simulator for consistency loss.
    """
    B, C, H, W = restored.shape
    device = restored.device
    dtype = restored.dtype

    # separable 3x3 kernel
    kernel = torch.tensor([1., 2., 1.], device=device, dtype=dtype)
    kernel2d = torch.
