#!/usr/bin/env python3
"""
MPH_training_script5_fixed2.py

Final Integrated training script for Mirage Correction (MPH).
Includes all stability fixes, especially the cuDNN benchmark fix.
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
    kernel2d = torch.outer(kernel, kernel)
    kernel2d = kernel2d / kernel2d.sum()
    kernel2d = kernel2d.view(1, 1, 3, 3)
    weight = kernel2d.repeat(C, 1, 1, 1)

    blurred = nn.functional.conv2d(restored, weight, bias=None, stride=1, padding=1, groups=C)

    # amplitude scintillation from luminance smoothed by 9x9 kernel
    lum = 0.299 * blurred[:, 0:1] + 0.587 * blurred[:, 1:2] + 0.114 * blurred[:, 2:3]
    big_k = torch.ones(1, 1, 9, 9, device=device, dtype=dtype) / (9.0 * 9.0)
    amp = nn.functional.conv2d(lum, big_k, padding=4)
    amp_min = amp.amin(dim=(-2, -1), keepdim=True)
    amp_max = amp.amax(dim=(-2, -1), keepdim=True)
    amp = (amp - amp_min) / (amp_max - amp_min + 1e-8)
    amp = 0.98 + 0.06 * amp

    simulated = blurred * amp

    if lpd_mu is not None:
        if lpd_mu.dim() > 1:
            strength = torch.sigmoid(lpd_mu.abs().mean(dim=1)).view(B, 1, 1, 1)
        else:
            strength = torch.sigmoid(lpd_mu.abs()).view(B, 1, 1, 1)
        noise_std = noise_scale * (1.0 + 2.0 * (1.0 - strength))
    else:
        noise_std = noise_scale
    noise = torch.randn_like(simulated) * noise_std
    degraded = (simulated + noise).clamp(0.0, 1.0)
    return degraded

# ----------------------------- Training / Validation ----------------------#

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, cfg):
    model.train()
    rec_loss_fn = nn.L1Loss()
    charbon = CharbonnierLoss()
    running_loss = 0.0
    it = 0
    for batch in dataloader:
        it += 1
        frames = batch['frames'].to(device)         # (B, T, C, H, W)
        gt = batch['gt'].to(device)                 # (B, C, H, W)
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        B = frames.size(0)
        degraded_center = frames[:, cfg.T // 2].to(device)

        with autocast(enabled=cfg.use_amp):
            restored, mu, logvar = model(frames)    # model output
            loss_rec = rec_loss_fn(restored, gt)
            
            # KL Divergence Loss
            if (mu is not None) and (logvar is not None):
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            else:
                kl = torch.tensor(0., device=device)
            
            # Consistency Loss
            sim = simple_simulate(restored, mu)
            loss_cons = charbon(sim, degraded_center)
            
            # Total Loss
            total_loss = loss_rec + cfg.kl_weight * kl + cfg.lambda_cons * loss_cons

        scaler.scale(total_loss).backward()

        # AMP-safe unscale and gradient clipping
        if cfg.use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += total_loss.item()
        if it % cfg.log_iter == 0:
            print(f"[Train] Epoch {epoch} Iter {it}/{len(dataloader)}  loss {running_loss/it:.6f}")

    return running_loss / max(1, it)

@torch.no_grad()
def validate(model, dataloader, device, cfg):
    model.eval()
    rec_loss_fn = nn.L1Loss()
    charbon = CharbonnierLoss()
    running_val = 0.0
    cnt = 0
    for batch in dataloader:
        frames = batch['frames'].to(device)
        gt = batch['gt'].to(device)
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        degraded_center = frames[:, cfg.T // 2]
        restored, mu, logvar = model(frames)
        loss_rec = rec_loss_fn(restored, gt)
        sim = simple_simulate(restored, mu)
        loss_cons = charbon(sim, degraded_center)
        total = loss_rec + cfg.kl_weight * 0.0 + cfg.lambda_cons * loss_cons
        running_val += total.item()
        cnt += 1
    if cnt == 0:
        return float('inf')
    return running_val / cnt

# ----------------------------- CLI & Main -----------------------------#

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='path to synthetic .npz data folder')
    p.add_argument('--expdir', type=str, required=True, help='experiment output dir')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--n_train', type=int, default=9000)
    p.add_argument('--n_val', type=int, default=1000)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--crop_size', type=int, default=128)
    p.add_argument('--lambda_cons', type=float, default=0.05)
    p.add_argument('--kl_weight', type=float, default=0.0)
    p.add_argument('--resume_from', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_amp', action='store_true')
    p.add_argument('--log_iter', type=int, default=50)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--freeze_encoder_epochs', type=int, default=0, help='freeze encoder params for first N epochs')
    p.add_argument('--patience', type=int, default=12, help='Early Stopping patience')
    p.add_argument('--min_delta', type=float, default=1e-4, help='Early Stopping min delta')
    return p.parse_args()

def freeze_encoder(model):
    for name, p in model.named_parameters():
        if any(x in name.lower() for x in ('encoder', 'down', 'fe', 'backbone')):
            p.requires_grad = False

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    os.makedirs(cfg.expdir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.expdir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # CRITICAL FIX: Enable cuDNN benchmarking for stability on low-level convolutions
    torch.backends.cudnn.benchmark = True 
    
    # Model
    model = MPHModel(in_ch=3, feat_ch=64, z_dim=32, T=cfg.T).to(device)

    # Optionally freeze encoder for first N epochs
    if cfg.freeze_encoder_epochs > 0:
        print(f"Freezing encoder for first {cfg.freeze_encoder_epochs} epochs.")
        freeze_encoder(model)

    # Datasets / loaders
    ds_train = SyntheticTurbulenceDataset(cfg.data, T_train=cfg.T, crop_size=cfg.crop_size, augment=True)
    ds_val = SyntheticTurbulenceDataset(cfg.data, T_train=cfg.T, crop_size=cfg.crop_size, augment=False)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=max(1, cfg.num_workers//2), pin_memory=True)

    # Optimizer, scheduler, scaler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)
    scaler = GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    best_val = float('inf')
    earlystop = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    # resume if requested
    if cfg.resume_from:
        try:
            s_epoch, loaded_best = load_checkpoint(cfg.resume_from, model, optimizer, scheduler, scaler, device=device)
            start_epoch = int(s_epoch) + 1
            if loaded_best is not None:
                best_val = loaded_best
            print(f"Resumed from {cfg.resume_from} at epoch {start_epoch}, best_val={best_val:.6f}")
        except Exception as e:
            print(f"Resume failed: {e}. Starting from scratch.")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, cfg)
        val_loss = validate(model, val_loader, device, cfg)
        t1 = time.time()

        # scheduler step (ReduceLROnPlateau)
        scheduler.step(val_loss)

        # logging
        print(f"Epoch {epoch}/{cfg.epochs} - train_loss {train_loss:.6f} - val_loss {val_loss:.6f} - time {t1-t0:.1f}s - lr {optimizer.param_groups[0]['lr']:.2e}")
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # save last checkpoint
        save_checkpoint(os.path.join(cfg.expdir, 'checkpoint_last.pth'), model, optimizer, scheduler, epoch=epoch, best_val=best_val, scaler=scaler)

        # save best
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            save_checkpoint(os.path.join(cfg.expdir, 'checkpoint_best.pth'), model, optimizer, scheduler, epoch=epoch, best_val=best_val, scaler=scaler)
            print(f"Saved new best model at epoch {epoch} (val {val_loss:.6f})")

        # early stopping
        if earlystop.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch}. Best val: {earlystop.best:.6f}")
            break

        # unfreeze encoder if needed
        if cfg.freeze_encoder_epochs > 0 and (epoch + 1) == cfg.freeze_encoder_epochs:
            print("Unfreezing encoder parameters now.")
            unfreeze_all(model)

    writer.close()
    print("Training finished. Best val:", best_val)

if __name__ == '__main__':
    main()
