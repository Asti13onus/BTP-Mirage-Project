"""
MPH_eval_metrics.py
Quantitative evaluation script for final thesis results (Stage 3).
Measures PSNR, SSIM, and LPIPS on a synthetic test set.
"""
import os, torch, numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from pathlib import Path

# Import your model skeleton - ensure PYTHONPATH or same dir contains this module
from MPH_model_skeleton_final import MPHModel # Use your final stable model name

# --- Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Configuration: Adjust these paths ---
# We will evaluate the FINAL ADAPTED model from the BVI fine-tuning run (Job 2)
CKPT_PATH = "/home1/astitva_s/BTP_Mirage_Project/experiments/exp_BVI_simcons/best.pth"
TEST_DIR = "/home1/astitva_s/BTP_Mirage_Project/synth_data/test" # Assumes you moved files here

def load_and_evaluate():
    # Load model
    model = MPHModel().to(device)
    
    # Fixed block: Check for either 'model_state_dict' (robust) or 'model' (simple)
    # Load the FINAL ADAPTED checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device)
    
    # Check for both possible model state keys
    if 'model_state_dict' in ckpt:
        model_state = ckpt['model_state_dict']
        key_used = 'model_state_dict'
    elif 'model' in ckpt:
        model_state = ckpt['model']
        key_used = 'model'
    else:
        raise KeyError(f"Could not find model weights. Keys found: {list(ckpt.keys())}")
        
    print(f"Loaded model state using key: {key_used}")
        
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # Initialize metrics
    print(f"Initializing LPIPS (VGG) on {device}...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    psnr_list, ssim_list, lpips_list = [], [], []

    print(f"Starting evaluation on {TEST_DIR}...")
    test_files = sorted(Path(TEST_DIR).glob('*.npz'))
    
    for file_path in test_files:
        try:
            data = np.load(str(file_path))
        except Exception:
            continue
            
        # Data is stored as seq (T, H, W, C). GT is the center frame (pseudo-GT).
        seq_np = data['seq'].astype(np.float32)
        gt_np_full = seq_np[2] # Assumes center frame (t=2 for T=5) is the GT

        # Prepare for model input (add batch dim, permute)
        seq_tensor = torch.from_numpy(seq_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(gt_np_full).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            restored_tensor, _, _ = model(seq_tensor)

        # Convert tensors to numpy for PSNR/SSIM calculation
        restored_np = restored_tensor.squeeze(0).permute(1,2,0).cpu().numpy().clip(0,1)
        
        # Calculate metrics
        psnr_list.append(peak_signal_noise_ratio(gt_np_full, restored_np, data_range=1.0))
        ssim_list.append(structural_similarity(gt_np_full, restored_np, channel_axis=-1, data_range=1.0))
        lpips_list.append(lpips_fn(restored_tensor, gt_tensor).item())

    # --- Print Final Results ---
    print("\n--- FINAL QUANTITATIVE RESULTS (Synthetic Test Set) ---")
    print(f"Total Samples: {len(psnr_list)}")
    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_list):.4f} (Lower is Better)")
    print("-----------------------------------------------------")


if __name__ == '__main__':
    # You MUST install lpips and scikit-image first!
    load_and_evaluate()
