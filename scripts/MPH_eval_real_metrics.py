"""
MPH_eval_real_metrics.py
Stage 3: Perceptual Evaluation on Real-World Data (NIQE, BRISQUE).

Calculates multiple No-Reference Image Quality (NR-IQA) metrics 
to prove real-world usability and restoration effectiveness.
"""
import os, torch, numpy as np
import cv2 
from pathlib import Path
from tqdm import tqdm
from piq import niqe, brisque 

# Import your model skeleton
from MPH_model_skeleton_final import MPHModel 

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Configuration: Adjust these paths ---
# Load the FINAL ADAPTED model from the consistency refinement job
CKPT_PATH = "/home1/astitva_s/BTP_Mirage_Project/experiments/exp_BVI_simcons/best.pth"

# Target the folder containing the real video sequences (BVI/OTIS frames)
REAL_DATA_DIR = "/home1/astitva_s/BTP_Mirage_Project/real_data/Fixed Backgrounds" 

# Sequence length for model input
T_FRAMES = 5
IMG_SIZE = 128
SAVE_PATH = Path("/home1/astitva_s/BTP_Mirage_Project/experiments/real_eval_output")
SAVE_PATH.mkdir(exist_ok=True, parents=True)

# ----------------------------- Data Loading Utilities -------------------------#

def load_real_frames(base_dir: str):
    """Recursively finds all PNG files in subdirectories and returns their paths."""
    all_paths = []
    # Note: Use os.path.join for robustness, especially with spaces
    for dirpath, _, filenames in os.walk(base_dir):
        # Filter out potential GTs and ensure we only get PNG frames
        frames = [Path(dirpath) / f for f in sorted(filenames) if f.endswith('.png') and 'gt' not in f.lower()]
        all_paths.extend(frames)
    
    # Create list of starting indices for T_FRAMES long clips (simple parent folder check)
    start_indices = [i for i in range(len(all_paths) - T_FRAMES) if all_paths[i].parent == all_paths[i+T_FRAMES-1].parent]
    
    print(f"Found {len(start_indices)} usable {T_FRAMES}-frame sequences.")
    return all_paths, start_indices

def load_image_to_tensor(path: Path) -> torch.Tensor:
    """Loads image, converts to RGB, normalizes, and converts to tensor (C, H, W)."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        # Fallback to PIL if OpenCV fails, or raise error
        raise FileNotFoundError(f"Failed to load image: {path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    if img_float.shape[0] != IMG_SIZE or img_float.shape[1] != IMG_SIZE:
        img_float = cv2.resize(img_float, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    
    return torch.from_numpy(img_float).permute(2, 0, 1)

# ----------------------------- Main Evaluation Logic -------------------------#

def evaluate_real_metrics():
    # 1. Setup Model
    model = MPHModel().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # Robustly load model state
    model_state = ckpt.get('model_state_dict', ckpt.get('model'))
    if model_state is None:
        raise KeyError(f"Could not find model weights. Checked keys: {list(ckpt.keys())}")
        
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print(f"Model loaded from {CKPT_PATH}")

    # 2. Setup Data
    all_paths, start_indices = load_real_frames(REAL_DATA_DIR)
    
    niqe_in, niqe_out, brisque_in, brisque_out = [], [], [], []

    # 3. Evaluation Loop
    print(f"\nStarting NR-IQA Evaluation on {len(start_indices)} clips...")
    for clip_idx in tqdm(start_indices):
        
        # A. Prepare Input Sequence (T, C, H, W)
        frame_tensors = [load_image_to_tensor(all_paths[clip_idx + t]) for t in range(T_FRAMES)]
        input_seq = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(DEVICE) # (1, T, C, H, W)

        # B. Get Degraded Frame (Center frame)
        degraded_frame = input_seq[:, T_FRAMES // 2] # (1, C, H, W)

        with torch.no_grad():
            # C. Get Restored Frame
            restored_frame, _, _ = model(input_seq) # (1, C, H, W)

        # D. Calculate Metrics (NIQE/BRISQUE) - Lower is better.
        # Data range must be [0, 1] (float32)
        
        # 1. NIQE (Natural Image Quality Evaluator)
        niqe_in.append(niqe(degraded_frame, data_range=1.0, reduction='none').item())
        niqe_out.append(niqe(restored_frame, data_range=1.0, reduction='none').item())
        
        # 2. BRISQUE (Blind/Reference-less Image Spatial Quality Evaluator)
        brisque_in.append(brisque(degraded_frame, data_range=1.0, reduction='none').item())
        brisque_out.append(brisque(restored_frame, data_range=1.0, reduction='none').item())

        # E. Save Sample Visual (For Subjective Evaluation)
        if len(niqe_out) <= 5: # Save the first 5 samples
             # Concatenate Input and Restored side-by-side
            comparison_image = torch.cat([degraded_frame.squeeze(0), restored_frame.squeeze(0)], dim=2)
            # Use torchvision.utils.save_image (requires separate install, but we can write our own utility)
            # For simplicity, we save the first frame as a NumPy array if necessary.
            
            # --- Simple image save using CV2 (C, H, W to H, W, C, then denormalize) ---
            comp_np = comparison_image.permute(1, 2, 0).cpu().numpy()
            comp_np = (comp_np * 255).astype(np.uint8)
            cv2.imwrite(str(SAVE_PATH / f"comp_{len(niqe_out):02d}.png"), cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR))
            
    # 4. Print Final Results
    print("\n--- FINAL THESIS RESULTS (REAL DATA NR-IQA) ---")
    print(f"Dataset: {REAL_DATA_DIR}")
    print(f"Total Clips Evaluated: {len(start_indices)}")
    print(f"Metric: NIQE (Naturalness - Lower is Better)")
    print(f"Average Degraded NIQE:  {np.mean(niqe_in):.4f}")
    print(f"Average Restored NIQE:  {np.mean(niqe_out):.4f}")
    print(f"NIQE Improvement:       {np.mean(niqe_in) - np.mean(niqe_out):.4f}")
    print("-" * 35)
    print(f"Metric: BRISQUE (Spatial Quality - Lower is Better)")
    print(f"Average Degraded BRISQUE: {np.mean(brisque_in):.4f}")
    print(f"Average Restored BRISQUE: {np.mean(brisque_out):.4f}")
    print(f"BRISQUE Improvement:    {np.mean(brisque_in) - np.mean(brisque_out):.4f}")
    print("-" * 35)


if __name__ == '__main__':
    # Ensure torch.backends.cudnn.benchmark is set for consistency (from training script)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
    evaluate_real_metrics()
