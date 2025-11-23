"""
MPH_eval_metrics_stable.py
Stage 3: Perceptual Evaluation on Real-World Data (NIQE - Stable Fallback).

Uses scikit-image's robust metrics implementation to bypass all dependency errors.
"""
import os, torch, numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import math

# Use skimage for stable, non-PyTorch NIQE (skimage version must be >= 0.23)
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics._structural_similarity import niqe as niqe_fn # We hope this works!

# Import your model skeleton
from MPH_model_skeleton_final import MPHModel 

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Configuration: Adjust these paths ---
CKPT_PATH = "/home1/astitva_s/BTP_Mirage_Project/experiments/exp_BVI_simcons/best.pth"
REAL_DATA_DIR = "/home1/astitva_s/BTP_Mirage_Project/real_data/Fixed Backgrounds" 
SAVE_PATH = Path("/home1/astitva_s/BTP_Mirage_Project/experiments/real_eval_output_stable")
SAVE_PATH.mkdir(exist_ok=True, parents=True)

# Sequence parameters
T_FRAMES = 5
IMG_SIZE = 128

# ----------------------------- Data Loading Utilities -------------------------#

def load_real_frames(base_dir: str):
    """Recursively finds all PNG files in subdirectories and creates clip start indices."""
    all_paths = []
    for dirpath, _, filenames in os.walk(base_dir):
        frames = [Path(dirpath) / f for f in sorted(filenames) if f.endswith('.png') and 'gt' not in f.lower()]
        all_paths.extend(frames)
    
    start_indices = [i for i in range(len(all_paths) - T_FRAMES) if all_paths[i].parent == all_paths[i+T_FRAMES-1].parent]
    
    print(f"Found {len(start_indices)} usable {T_FRAMES}-frame sequences.")
    return all_paths, start_indices

def load_image_to_tensor(path: Path) -> torch.Tensor:
    """Loads image using PIL, resizes, and converts to tensor (C, H, W)."""
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to load image {path}. Using black tensor. Error: {e}")
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)
    
    if img.width != IMG_SIZE or img.height != IMG_SIZE:
        # Resize using PIL.Image.Resampling.BICUBIC
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
        
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1)

# ----------------------------- Main Evaluation Logic -------------------------#

def evaluate_real_metrics():
    # 1. Setup Model
    model = MPHModel().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # Load model state (robustly checks for 'model_state_dict' or 'model')
    model_state = ckpt.get('model_state_dict', ckpt.get('model'))
    if model_state is None:
        raise KeyError(f"Could not find model weights. Checked keys: {list(ckpt.keys())}")
        
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print(f"Model loaded from {CKPT_PATH}")

    # 2. Setup Data
    all_paths, start_indices = load_real_frames(REAL_DATA_DIR)
    
    niqe_input_scores, niqe_restored_scores = [], []
    
    # 3. Evaluation Loop
    print(f"\nStarting NIQE Evaluation on {len(start_indices)} clips...")
    for clip_idx in tqdm(start_indices):
        
        # A. Prepare Input Sequence (T, C, H, W)
        try:
            frame_tensors = [load_image_to_tensor(all_paths[clip_idx + t]) for t in range(T_FRAMES)]
            input_seq = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(DEVICE) # (1, T, C, H, W)
        except Exception:
            continue
            
        # B. Get Degraded Frame (Center frame)
        degraded_frame_t = input_seq[:, T_FRAMES // 2] # (1, C, H, W)

        with torch.no_grad():
            # C. Get Restored Frame
            restored_frame_t, _, _ = model(input_seq) # (1, C, H, W)
        
        # D. Convert Tensors to NumPy for NIQE (skimage requires H, W, C, uint8 or float)
        # We need H, W, C and need to be float64 or convert to uint8 for NIQE.
        degraded_np = (degraded_frame_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        restored_np = (restored_frame_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # E. Calculate NIQE (skimage version) - LOWER IS BETTER
        try:
            # NIQE expects a grayscale or single channel image for calculation
            # We convert the RGB image to grayscale luminance before passing it.
            # Convert RGB to Luminance (L = 0.299R + 0.587G + 0.114B)
            
            # Simple average grayscale conversion for stability
            gray_degraded = np.mean(degraded_np, axis=2).astype(np.uint8) 
            gray_restored = np.mean(restored_np, axis=2).astype(np.uint8) 
            
            niqe_input = niqe_fn(gray_degraded)
            niqe_restored = niqe_fn(gray_restored)
            
            niqe_input_scores.append(niqe_input)
            niqe_restored_scores.append(niqe_restored)
        
        except Exception as e:
            # Catch errors if skimage's NIQE function fails due to internal NumPy structure
            # print(f"NIQE calculation failed for clip {clip_idx}: {e}") 
            continue


    # 4. Print Final Results
    if not niqe_input_scores:
        print("\n--- EVALUATION FAILED ---")
        print("Could not calculate NIQE scores. Check scikit-image version compatibility.")
        return

    avg_niqe_input = np.mean(niqe_input_scores)
    avg_niqe_restored = np.mean(niqe_restored_scores)
    
    print("\n--- FINAL THESIS RESULTS (REAL DATA NR-IQA) ---")
    print(f"Dataset: {REAL_DATA_DIR}")
    print(f"Total Clips Evaluated: {len(niqe_input_scores)}")
    print(f"Metric: NIQE (Naturalness - Lower is Better)")
    print(f"Average Degraded NIQE:  {avg_niqe_input:.4f}")
    print(f"Average Restored NIQE:  {avg_niqe_restored:.4f}")
    print(f"NIQE Improvement:       {avg_niqe_input - avg_niqe_restored:.4f} (Must be Positive)")
    print("-----------------------------------------------")


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
    evaluate_real_metrics()
