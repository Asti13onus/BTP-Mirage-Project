"""
MPH_eval_real_final.py
Stage 3: Perceptual Evaluation on Real-World Data (NIQE).

Uses IQA-Pytorch and removes the unstable PIQ dependency.
"""
import os, torch, numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from IQA_pytorch.niqe import NIQE # CRITICAL: Import NIQE from the stable library

# Import your model skeleton
from MPH_model_skeleton_final import MPHModel 

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Configuration: Adjust these paths ---
CKPT_PATH = "/home1/astitva_s/BTP_Mirage_Project/experiments/exp_BVI_simcons/best.pth"
REAL_DATA_DIR = "/home1/astitva_s/BTP_Mirage_Project/real_data/Fixed Backgrounds" 
SAVE_PATH = Path("/home1/astitva_s/BTP_Mirage_Project/experiments/real_eval_output")
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
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
        
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1)

# ----------------------------- Main Evaluation Logic -------------------------#

def evaluate_real_metrics():
    # 1. Setup Model
    model = MPHModel().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    
    model_state = ckpt.get('model_state_dict', ckpt.get('model'))
    if model_state is None:
        raise KeyError(f"Could not find model weights. Checked keys: {list(ckpt.keys())}")
        
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print(f"Model loaded from {CKPT_PATH}")

    # 2. Setup Data and Metrics
    all_paths, start_indices = load_real_frames(REAL_DATA_DIR)
    
    # Initialize NIQE Evaluator (must be done outside the loop)
    # The NIQE class from IQA-Pytorch is initialized once.
    niqe_evaluator = NIQE(data_range=1., as_loss=False).to(DEVICE)
    
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
        degraded_frame = input_seq[:, T_FRAMES // 2] # (1, C, H, W)

        with torch.no_grad():
            # C. Get Restored Frame
            restored_frame, _, _ = model(input_seq) # (1, C, H, W)

        # D. Calculate NIQE (Inputs must be float32, range [0, 1] and have batch dim)
        
        # Score Degraded Input (Input must be unsqueezed to BATCH: (1, C, H, W))
        niqe_input = niqe_evaluator(degraded_frame).item()
        
        # Score Restored Output
        niqe_restored = niqe_evaluator(restored_frame).item()

        niqe_input_scores.append(niqe_input)
        niqe_restored_scores.append(niqe_restored)

        # E. Save Sample Visual (For Subjective Evaluation)
        if len(niqe_input_scores) <= 5: # Save the first 5 samples
             # Concatenate Input and Restored side-by-side
            comparison_image = torch.cat([degraded_frame.squeeze(0), restored_frame.squeeze(0)], dim=2)
            comp_np = comparison_image.permute(1, 2, 0).cpu().numpy()
            np.savez_compressed(SAVE_PATH / f"comp_{len(niqe_input_scores):02d}.npz", comparison=comp_np)


    # 4. Print Final Results
    print("\n--- FINAL THESIS RESULTS (REAL DATA NR-IQA) ---")
    print(f"Dataset: {REAL_DATA_DIR}")
    print(f"Total Clips Evaluated: {len(niqe_input_scores)}")
    print(f"Metric: NIQE (Naturalness - Lower is Better)")
    print(f"Average Degraded NIQE:  {np.mean(niqe_input_scores):.4f}")
    print(f"Average Restored NIQE:  {np.mean(niqe_restored_scores):.4f}")
    print(f"NIQE Improvement:       {np.mean(niqe_input_scores) - np.mean(niqe_restored_scores):.4f} (Must be Positive)")
    print("-----------------------------------------------")


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
    evaluate_real_metrics()
