"""
MPH_visualize_static_direct.py
Generates final static, publication-quality triplet images (Distorted | Restored).

This script is optimized for DIRECT IMAGE INPUT (not video sequences) by 
repeating the input image 5 times to satisfy the model's T=5 input shape.
"""
import os, torch, numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import math

# Import components
from MPH_model_skeleton_final import MPHModel 

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = "/home1/astitva_s/BTP_Mirage_Project/experiments/exp_BVI_simcons/best.pth"
# NOTE: This path should point directly to the folder containing the individual .png files.
REAL_DATA_DIR = "/home1/astitva_s/BTP_Mirage_Project/real_data/Fixed Backgrounds/Door" 
# Output folder for saving individual comparison images
VIS_OUTPUT_PATH = Path("/home1/astitva_s/BTP_Mirage_Project/visual_results_direct")
VIS_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

T_FRAMES = 5
IMG_SIZE = 128

# Helper Functions
def find_image_paths(base_dir: str):
    """Finds all static PNG/JPG files directly in the base directory."""
    paths = [Path(base_dir) / f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg'))]
    print(f"Found {len(paths)} static images for processing.")
    return paths

def load_image_to_tensor(path: Path) -> torch.Tensor:
    # Uses PIL logic from previous successful version
    try:
        img = Image.open(path).convert('RGB')
    except Exception:
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)
    if img.width != IMG_SIZE or img.height != IMG_SIZE:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1)


def generate_visuals(num_images_to_save=10):
    # 1. Setup Model
    model = MPHModel().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model_state = ckpt.get('model_state_dict', ckpt.get('model'))
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print(f"Model loaded from {CKPT_PATH}")

    # 2. Setup Data
    all_image_paths = find_image_paths(REAL_DATA_DIR)
    images_to_process = all_image_paths[:num_images_to_save]
    
    print(f"\nGenerating {len(images_to_process)} static comparison images...")

    for img_idx, img_path in enumerate(tqdm(images_to_process)):
        
        # 1. Load single distorted image (C, H, W)
        degraded_img_t = load_image_to_tensor(img_path) 
        
        # 2. Create 5-frame input sequence by stacking the single image (T, C, H, W)
        input_seq = degraded_img_t.unsqueeze(0).repeat(T_FRAMES, 1, 1, 1).unsqueeze(0).to(DEVICE)
        
        # 3. Run Inference (Once per image)
        with torch.no_grad():
            restored_center_t, _, _ = model(input_seq) # restored is (1, C, H, W)
        
        # 4. Create the final visualization (Degraded | Restored)
        # Input tensor needs a batch dimension (unsqueeze(0)) for concatenation
        comparison_t = torch.cat([degraded_img_t, restored_center_t.squeeze(0)], dim=2)
        
        # 5. Save the image
        comp_np = comparison_t.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        save_name = VIS_OUTPUT_PATH / f"final_comparison_{img_idx:02d}.png"
        
        # Convert float [0, 1] to uint8 [0, 255] for standard PNG save
        comp_pil = Image.fromarray((comp_np * 255).astype(np.uint8))
        comp_pil.save(save_name)
        
    print("\nStatic comparison images saved. Ready for transfer.")


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
    generate_visuals()
