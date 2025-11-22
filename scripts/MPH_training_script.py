"""
MPH_training_script.py

Training script for the Mamba-PiRN Hybrid (MPH) model skeleton.
Defaults:
 - frame window T=5
 - crop size 256x256
 - batch size 4 (adjustable)
 - optimizer: AdamW lr=5e-4
 - scheduler: cosine annealing with warmup
 - mixed precision (AMP)
 - checkpointing + TensorBoard logging

Notes:
 - This script assumes the MPH model skeleton (MPH_model_skeleton.py) is
   in the same directory and importable. Replace placeholder dataset and
   simulator classes with your real implementations (see comments below).
 - Simulator-in-loop consistency loss and tilt supervision are included as
   placeholders; they require a differentiable simulator function `simulate()`
   that re-degrades images given LPD params or z.

Usage (example):
  python MPH_training_script.py --data /path/to/syn_dataset --expdir ./exp_mph

Author: Generated for Astitva Srivastava (Bachelor Thesis)
"""

import os
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# import the model skeleton created earlier (must be in PYTHONPATH)
from MPH_model_skeleton import MPHModel


# ----------------------------- Placeholder Dataset ----------------------#
class SyntheticTurbulenceDataset(Dataset):
    """
    Placeholder dataset. Replace with actual dataset that returns:
        sample = {
            'frames': Tensor (T, C, H, W),
            'gt': Tensor (C, H, W) or None if unavailable,
            'tilt_flow': Tensor (T, 2, H, W) optional
            'metadata': dict()
        }
    The dataset should produce tensors in range [0,1].
    """
    def __init__(self, root, split='train', T=5, crop_size=256, n_samples=1000):
        super().__init__()
        self.root = root
        self.split = split
        self.T = T
        self.crop_size = crop_size
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # For skeleton: return random noise as degraded frames, and a clean GT
        C = 3
        H = W = self.crop_size
        frames = torch.rand(self.T, C, H, W)
        gt = torch.rand(C, H, W)
        # tilt_flow is optional; here random
        tilt_flow = torch.randn(self.T, 2, H, W) * 2.0
        return {'frames': frames, 'gt': gt, 'tilt_flow': tilt_flow, 'meta': {}}


# ----------------------------- Utilities --------------------------------#

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, fname):
    torch.save(state, fname)


# ----------------------------- Losses ----------------------------------#
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


# ------------------------ Simulator placeholder -------------------------#
def simulate(restored_image: torch.Tensor, lpd_mu: torch.Tensor, lpd_logvar: torch.Tensor, metadata=None):
    """
    Placeholder for a differentiable re-degradation simulator.
    Given the restored image (B,C,H,W) and LPD params (B,z), return a simulated
    degraded image (B,C,H,W) that should match the input degraded frame.

    Replace this with a differentiable thin-screen or DF-P2S simulator.
    """
    # naive simul: add gaussian blur + noise parameterized by lpd mean magnitude
    B, C, H, W = restored_image.shape
    sigma = (lpd_mu.abs().mean(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 0.02) + 0.01
    # perform simple gaussian blur via conv (inefficient but illustrative)
    kernel_size = 5
    padding = kernel_size // 2
    # build a simple mean filter scaled by sigma (non-physically)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=restored_image.device) / (kernel_size * kernel_size)
    sim = restored_image
    # apply to each channel separately
    sim = nn.functional.conv2d(sim, kernel.expand(C, 1, kernel_size, kernel_size), groups=C, padding=padding)
    # add gaussian noise scaled
    sim = sim + torch.randn_like(sim) * sigma
    return sim.clamp(0.0, 1.0)


# ----------------------------- Trainer ---------------------------------#
class Trainer:
    def __init__(self, config):
        self.config = config
        seed_everything(config.seed)

        # model
        self.model = MPHModel(in_ch=3, feat_ch=config.feat_ch, z_dim=config.z_dim, T=config.T)
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config.cpu else 'cpu')
        self.model.to(self.device)

        # data
        train_ds = SyntheticTurbulenceDataset(config.data, split='train', T=config.T, crop_size=config.crop_size, n_samples=config.n_train)
        val_ds = SyntheticTurbulenceDataset(config.data, split='val', T=config.T, crop_size=config.crop_size, n_samples=config.n_val)
        self.train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

        # optimizer & scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        # cosine annealing with warmup (simple)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(1, config.epochs - config.warmup_epochs))

        # losses
        self.rec_loss = CharbonnierLoss()
        self.perc_loss = nn.MSELoss()  # placeholder for VGG perceptual
        self.kl_weight = config.kl_weight
        self.tilt_weight = config.tilt_weight
        self.cons_weight = config.cons_weight

        # amp scaler
        self.scaler = GradScaler(enabled=not config.disable_amp)

        # logging & checkpoint
        self.expdir = Path(config.expdir)
        self.expdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.expdir / 'tb'))
        self.best_val = float('-inf')

        # optionally resume
        if config.resume is not None and os.path.exists(config.resume):
            self._load_checkpoint(config.resume)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.opt.load_state_dict(ckpt['opt'])
        print(f"Resumed from {path}")

    def train(self):
        cfg = self.config
        global_step = 0
        for epoch in range(cfg.epochs):
            self.model.train()
            epoch_loss = 0.0
            tic = time.time()
            for batch_idx, sample in enumerate(self.train_loader):
                frames = sample['frames'].to(self.device)  # (B,T,C,H,W)
                gt = sample['gt'].to(self.device)
                tilt_flow = sample.get('tilt_flow', None)
                if tilt_flow is not None:
                    tilt_flow = tilt_flow.to(self.device)

                with autocast(enabled=not cfg.disable_amp):
                    restored, mu, logvar = self.model(frames)
                    # reconstruction on center frame
                    target = gt
                    loss_rec = self.rec_loss(restored, target)

                    # KL for LPD VAE
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss_kl = self.kl_weight * kl

                    # tilt supervision (if tilt_flow present) -> compare to coarse tilt flows predicted by model
                    # our skeleton model returns no flows; in full model you would compute tilt_loss here
                    loss_tilt = torch.tensor(0.0, device=self.device)
                    if tilt_flow is not None:
                        # placeholder: zero loss
                        loss_tilt = torch.tensor(0.0, device=self.device)

                    # simulator consistency
                    sim = simulate(restored, mu, logvar)
                    # use center input frame as degraded target for sim consistency
                    degraded_center = frames[:, cfg.T//2]
                    loss_cons = self.rec_loss(sim, degraded_center)

                    # total loss
                    loss = loss_rec + loss_kl + self.tilt_weight * loss_tilt + self.cons_weight * loss_cons

                # backward
                self.scaler.scale(loss).backward()
                # gradient clipping optional
                if cfg.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                epoch_loss += loss.item()
                if global_step % cfg.log_every == 0:
                    self.writer.add_scalar('train/loss', loss.item(), global_step)
                    self.writer.add_scalar('train/loss_rec', loss_rec.item(), global_step)
                    self.writer.add_scalar('train/loss_kl', kl.item(), global_step)
                global_step += 1

            toc = time.time()
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{cfg.epochs} - loss {avg_loss:.4f} - time {toc-tic:.1f}s")

            # scheduler step
            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            # validation
            val_score = self.validate(epoch)

            # checkpoint
            ckpt = {
                'model': self.model.state_dict(),
                'opt': self.opt.state_dict(),
                'epoch': epoch,
                'config': vars(cfg),
            }
            ckpt_path = self.expdir / f'checkpoint_epoch{epoch+1}.pth'
            save_checkpoint(ckpt, ckpt_path)
            # keep best by validation score (here lower val loss is better)
            if val_score > self.best_val:
                self.best_val = val_score
                save_checkpoint(ckpt, self.expdir / 'best.pth')

        self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        tot_loss = 0.0
        n = 0
        with torch.no_grad():
            for sample in self.val_loader:
                frames = sample['frames'].to(self.device)
                gt = sample['gt'].to(self.device)
                restored, mu, logvar = self.model(frames)
                loss = self.rec_loss(restored, gt).item()
                tot_loss += loss
                n += 1
        avg = -tot_loss / max(1, n)  # negative so higher is better for "val_score"
        self.writer.add_scalar('val/loss', tot_loss / max(1, n), epoch)
        print(f"Validation epoch {epoch+1}: loss {tot_loss / max(1,n):.4f}")
        return avg


# ----------------------------- Argument parser -------------------------#
def get_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=False, default='./data', help='dataset root')
    p.add_argument('--expdir', type=str, default='./exp', help='where to save logs and checkpoints')
    p.add_argument('--T', type=int, default=5, help='number of frames in window')
    p.add_argument('--crop_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--feat_ch', type=int, default=128)
    p.add_argument('--z_dim', type=int, default=32)
    p.add_argument('--n_train', type=int, default=2000)
    p.add_argument('--n_val', type=int, default=200)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--disable_amp', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--kl_weight', type=float, default=0.01)
    p.add_argument('--tilt_weight', type=float, default=0.5)
    p.add_argument('--cons_weight', type=float, default=0.5)
    p.add_argument('--log_every', type=int, default=50)
    return p


# ----------------------------- Main ------------------------------------#
if __name__ == '__main__':
    parser = get_argparser()
    cfg = parser.parse_args()
    trainer = Trainer(cfg)
    trainer.train()
