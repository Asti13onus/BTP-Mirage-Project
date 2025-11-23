"""
MPH_visualize_frames.py
Generates individual Degraded, Restored, and GT frames for thesis visualization.
"""
import os, torch, numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import math

# Import components (Assuming PIL.Image and core torch/numpy are stable)
from MPH_model_skeleton_final import MPHModel 

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = "/home1/astitva_s/BTP_Mirage_Project/experiments/exp_BVI_simcons/best.pth"
REAL_DATA_DIR = "/home1/astitva_s/BTP_Mirage_Project/real_data/Fixed Backgrounds" 
# Output folder for saving individual frames
VIS_OUTPUT_PATH = Path("/home1/astitva_s/BTP_Mirage_Project/visual_results")
VIS_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

T_FRAMES = 5
IMG_SIZE = 128

# Helper: Load model and data (using proven functions, assuming successful implementation)
# NOTE: The load_real_frames and load_image_to_tensor functions must be copied 
# into this script or imported from a reliable utility file. 
# For simplicity here, we rely on the logic from MPH_eval_final_metrics.py.

# [Insert load_real_frames and load_image_to_tensor functions here from previous steps]

# --- PASTE NECESSARY DATA LOADING UTILITIES HERE (from MPH_eval_final_metrics.py) ---
def load_real_frames(base_dir: str):
    all_paths = []
    for dirpath, _, filenames in os.walk(base_dir):
        frames = [Path(dirpath) / f for f in sorted(filenames) if f.endswith('.png') and 'gt' not in f.lower()]
        all_paths.extend(frames)
    start_indices = [i for i in range(len(all_paths) - T_FRAMES) if all_paths[i].parent == all_paths[i+T_FRAMES-1].parent]
    print(f"Found {len(start_indices)} usable {T_FRAMES}-frame sequences.")
    return all_paths, start_indices

def load_image_to_tensor(path: Path) -> torch.Tensor:
    try:
        img = Image.open(path).convert('RGB')
    except Exception:
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)
    if img.width != IMG_SIZE or img.height != IMG_SIZE:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1)
# --- END PASTE ---


def generate_visuals(num_clips_to_save=5):
    model = MPHModel().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model_state = ckpt.get('model_state_dict', ckpt.get('model'))
    model.load_state_dict(model_state, strict=True)
    model.eval()

    all_paths, start_indices = load_real_frames(REAL_DATA_DIR)
    
    # We will sample the first N clips for visualization
    clips_to_process = start_indices[:num_clips_to_save]
    
    print(f"\nGenerating {len(clips_to_process)} visualization clips...")

    for clip_idx, start_idx in enumerate(tqdm(clips_to_process)):
        # Process every frame within the T_FRAMES window for video generation
        for t in range(T_FRAMES):
            frame_path = all_paths[start_idx + t]
            
            # --- MODEL INFERENCE ---
            # Create a 1-clip, 5-frame input tensor
            frame_tensors = [load_image_to_tensor(all_paths[start_idx + i]) for i in range(T_FRAMES)]
            input_seq = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                restored_t, _, _ = model(input_seq)
            
            # --- SAVE FRAMES ---
            
            # Use the currently processed frame (t) as the degraded input for the triplet
            degraded_t = input_seq.squeeze(0)[t].unsqueeze(0)
            
            # Use the model's single restored output (which is the result for the center frame)
            # We repeat the restored image for temporal stability visual:
            restored_center_t = restored_t 
            
            # Create the final comparison image (Degraded | Restored Center)
            comparison_t = torch.cat([degraded_t.squeeze(0), restored_center_t.squeeze(0)], dim=2)
            
            # Convert to NumPy (H, W, C) float [0, 1]
            comp_np = comparison_t.permute(1, 2, 0).cpu().numpy().clip(0, 1)

            # Save frame for video assembly later (filename encodes clip and frame number)
            # Filename format: clip_01_frame_03.png
            save_name = VIS_OUTPUT_PATH / f"clip_{clip_idx:02d}_frame_{t:02d}.png"
            
            # Convert float [0, 1] to uint8 [0, 255] for standard image save
            comp_pil = Image.fromarray((comp_np * 255).astype(np.uint8))
            comp_pil.save(save_name)
            
    print("\nVisualization frames saved. Ready for video assembly (Step 2.1).")


if __name__ == '__main_':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
    # We run the function directly
    generate_visuals()
