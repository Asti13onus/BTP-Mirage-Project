"""
MPH_synthetic_dataset.py

Synthetic dataset generator for Mirage / heat-scintillation image sequences.
Implements a thin-phase-screen DF-P2S-like simulator (FFT-based) to produce
multi-frame sequences affected by atmospheric turbulence (tilt + blur + scintillation).

Features:
 - Kolmogorov PSD phase screen generator (FFT method)
 - Temporal correlation via AR(1) evolution of phase screens
 - Tip/tilt extraction and optional geometric warping
 - PSF generation from pupil * exp(i*phase) and FFT-convolution to blur image
 - Save sequences to disk in a simple folder structure or as .npz files

Notes & caveats:
 - This is a research-grade, still-approximate simulator intended for pretraining
   and ablation studies. It mirrors the common thin-screen approaches used in
   the Mirage literature (TMT / ATSyn style) but is intentionally compact.
 - You should cite the relevant literature when using this in your thesis.

Usage example:
    python MPH_synthetic_dataset.py --input_imgs /path/to/clean_images \
        --out_dir ./synth_data --n_samples 1000 --T 50

Author: Generated for Astitva Srivastava (Bachelor Thesis)
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

# require scipy for fft and convolution
import scipy.fft
import scipy.ndimage


# -------------------------- Utilities ----------------------------------#

def load_image(path, resize=None):
    im = Image.open(path).convert('RGB')
    if resize is not None:
        im = im.resize(resize, Image.BICUBIC)
    arr = np.array(im).astype(np.float32) / 255.0
    return arr


def save_sequence(seq, out_path):
    # seq: (T, H, W, C) float32 [0,1]
    np.savez_compressed(out_path, seq=seq)


# ---------------------- Phase screen generation ------------------------#
def kolmogorov_phase_screen(N, delta, r0, seed=None):
    """
    Generate a single phase screen using the FFT-based Kolmogorov spectrum method.
    N: screen size (assumed square)
    delta: spatial sampling interval (meters per pixel, arbitrary scale)
    r0: Fried's parameter (meters) - smaller r0 -> stronger turbulence
    Returns: phase screen (N,N) in radians

    Reference idea: generate complex gaussian noise in freq domain scaled by
    PSD ~ |k|^{-11/3} and inverse FFT to spatial domain. This is a commonly used
    approximate method in thin-screen simulators.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # frequency coordinates
    fx = np.fft.fftfreq(N, d=delta)
    fy = np.fft.fftfreq(N, d=delta)
    FX, FY = np.meshgrid(fx, fy)
    K = np.sqrt(FX**2 + FY**2)
    K[0,0] = 1e-6  # avoid division by zero

    # Kolmogorov PSD (phase) ~ 0.023 * r0^{-5/3} * |k|^{-11/3}
    PSD_phi = 0.023 * (r0 ** (-5.0/3.0)) * (K ** (-11.0/3.0))

    # random complex field with PSD
    cn = (rng.normal(size=(N,N)) + 1j * rng.normal(size=(N,N)))
    fft_field = cn * np.sqrt(PSD_phi) * (1.0 / (N * N))
    phi = np.real(np.fft.ifft2(fft_field)) * (N * N)
    # normalize to zero mean
    phi = phi - np.mean(phi)
    return phi


# ------------------------ PSF from phase screen ------------------------#
def psf_from_phase(phase, pupil_radius_frac=0.3, pad=0):
    """
    Build a PSF by taking the Fourier transform of a pupil function multiplied by exp(i*phase).
    - phase: (N,N) phase screen (radians)
    - pupil_radius_frac: fraction of N that is pupil radius (circular pupil)
    Returns PSF (N,N) normalized to sum=1
    """
    N = phase.shape[0]
    xv, yv = np.meshgrid(np.linspace(-1,1,N), np.linspace(-1,1,N))
    R = np.sqrt(xv**2 + yv**2)
    pupil = (R <= pupil_radius_frac).astype(np.float32)
    complex_pupil = pupil * np.exp(1j * phase)
    # compute PSF = |FFT(pupil * exp(i phase))|^2
    fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_pupil)))
    psf = np.abs(fft_field)**2
    psf = psf / np.sum(psf)
    return psf


# -------------------------- Temporal evolution -------------------------#
class AR1PhaseEvolution:
    """
    Simple AR(1) temporal evolution for phase screens in Fourier domain.
    phi_{t+1} = alpha * phi_t + sqrt(1 - alpha^2) * w_t
    where w_t is a fresh Kolmogorov phase sample.
    """
    def __init__(self, N, delta, r0, alpha=0.95, seed=None):
        self.N = N
        self.delta = delta
        self.r0 = r0
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # initialize with one screen
        self.phi = kolmogorov_phase_screen(N, delta, r0, seed=self.rng.randint(1e9))

    def step(self):
        new_phi = kolmogorov_phase_screen(self.N, self.delta, self.r0, seed=self.rng.randint(1e9))
        self.phi = self.alpha * self.phi + math.sqrt(max(0.0, 1.0 - self.alpha**2)) * new_phi
        return self.phi


# --------------------------- Warping (tilt) ---------------------------#
def tilt_from_phase(phase, scale=1.0):
    """
    Compute a simple global tilt (dx,dy) from the low-frequency content of the phase
    by fitting a plane to the phase screen (i.e., first-order Zernike: tip/tilt).
    Returns dx, dy offsets in pixels (float)
    """
    N = phase.shape[0]
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    X, Y = np.meshgrid(x, y)
    A = np.stack([X.ravel(), Y.ravel(), np.ones(N*N)], axis=1)
    b = phase.ravel()
    # least-squares plane fit
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a_x, a_y, a_c = coeff
    # convert slope to pixel shift roughly: tilt -> wavefront slope -> displacement
    dx = -a_x * scale * N
    dy = -a_y * scale * N
    return dx, dy


def warp_image(img, dx, dy):
    # img: (H,W,C) or (H,W)
    # use scipy.ndimage.shift (orders can be set)
    return scipy.ndimage.shift(img, shift=(dy, dx, 0), order=1, mode='reflect') if img.ndim==3 else scipy.ndimage.shift(img, shift=(dy, dx), order=1, mode='reflect')


# ----------------------- Convolution via FFT ---------------------------#
def convolve_fft(img, psf):
    """
    Convolve image with PSF using FFT (per-channel)
    img: (H,W,C)
    psf: (H,W)
    returns blurred image same shape
    """
    H, W = psf.shape
    out = np.zeros_like(img)
    # FFT-based convolution: multiply in freq domain
    # pad to avoid circular wrap: use same size and assume PSF centered -> use fftshift
    for c in range(img.shape[2]):
        im = img[:,:,c]
        Im = np.fft.fft2(im)
        PSF = np.fft.fft2(np.fft.ifftshift(psf))
        res = np.real(np.fft.ifft2(Im * PSF))
        out[:,:,c] = res
    # clip
    out = np.clip(out, 0.0, 1.0)
    return out


# --------------------------- Main generator ----------------------------#
def generate_sequence(clean_img, N, T, delta, r0, alpha, pupil_frac, tilt_scale, seed=None):
    """
    Create a T-frame degraded sequence from a clean image.
    Returns sequence array shape (T, H, W, C)
    """
    H0, W0, C = clean_img.shape
    # center-crop or resize to N
    if H0 != N or W0 != N:
        img = np.array(Image.fromarray((clean_img*255).astype(np.uint8)).resize((N,N), Image.BICUBIC)).astype(np.float32)/255.0
    else:
        img = clean_img

    evo = AR1PhaseEvolution(N=N, delta=delta, r0=r0, alpha=alpha, seed=seed)

    seq = np.zeros((T, N, N, C), dtype=np.float32)
    for t in range(T):
        phi = evo.step()
        # tilt
        dx, dy = tilt_from_phase(phi, scale=tilt_scale)
        warped = warp_image(img, dx, dy)
        # psf
        psf = psf_from_phase(phi, pupil_radius_frac=pupil_frac)
        blurred = convolve_fft(warped, psf)
        # add scintillation (amplitude fluctuations) by multiplying by 1 + small random field derived from low-pass of phi
        amp = 1.0 + 0.05 * scipy.ndimage.gaussian_filter(phi - phi.mean(), sigma=5)
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 0.1 + 0.95  # scale to ~[0.95,1.05]
        blurred = blurred * amp[:,:,None]
        seq[t] = np.clip(blurred, 0.0, 1.0)
    return seq


def main(args):
    input_dir = Path(args.input_imgs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob('*'))
    if len(images) == 0:
        raise ValueError('No images found in ' + str(input_dir))

    rng = np.random.RandomState(args.seed)
    for i in tqdm(range(args.n_samples)):
        img_path = str(images[rng.randint(0, len(images))])
        clean = load_image(img_path, resize=(args.N, args.N))
        # sample params
        # Fried parameter r0 in pixels: map a real-world range to pixel units
        # smaller r0 = stronger turbulence. We sample r0 between r0_min and r0_max
        r0 = float(rng.uniform(args.r0_min, args.r0_max))
        alpha = float(rng.uniform(args.alpha_min, args.alpha_max))
        pupil_frac = float(rng.uniform(args.pupil_frac_min, args.pupil_frac_max))
        tilt_scale = float(rng.uniform(args.tilt_scale_min, args.tilt_scale_max))

        seq = generate_sequence(clean, N=args.N, T=args.T, delta=args.delta, r0=r0, alpha=alpha, pupil_frac=pupil_frac, tilt_scale=tilt_scale, seed=int(rng.randint(1e9)))
        out_path = out_dir / f'seq_{i:06d}.npz'
        save_sequence(seq, out_path)
        if (i+1) % 100 == 0:
            print(f'Generated {i+1}/{args.n_samples}')

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_imgs', type=str, required=True, help='folder with clean images (jpg/png)')
    parser.add_argument('--out_dir', type=str, default='./synth_data', help='output folder')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--N', type=int, default=256, help='spatial size (square)')
    parser.add_argument('--T', type=int, default=50, help='frames per sequence')
    parser.add_argument('--delta', type=float, default=0.01, help='spatial sampling (arbitrary units)')
    parser.add_argument('--r0_min', type=float, default=0.1, help='min Fried parameter (weaker->larger)')
    parser.add_argument('--r0_max', type=float, default=0.5, help='max Fried parameter')
    parser.add_argument('--alpha_min', type=float, default=0.85, help='AR(1) alpha min')
    parser.add_argument('--alpha_max', type=float, default=0.99, help='AR(1) alpha max')
    parser.add_argument('--pupil_frac_min', type=float, default=0.15)
    parser.add_argument('--pupil_frac_max', type=float, default=0.4)
    parser.add_argument('--tilt_scale_min', type=float, default=0.5)
    parser.add_argument('--tilt_scale_max', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
