"""
simulate_torch.py
Differentiable simulator for tilt/warp distortion.
This function replaces the non-differentiable placeholder in the training script.
"""
import torch
import torch.nn.functional as F

def predict_tilt_flow(mu, H, W, device):
    """
    Differentiably maps the LPD mean (mu) to a global tilt flow field.
    This acts as a differentiable tilt prediction head using the VAE's latent space.
    """
    B, Z = mu.shape

    # 1. Predict unnormalized tilt vector (dx, dy) from the first two latent dimensions
    # This is a key differentiable link between LPD and tilt
    flow_vector_unnorm = mu[:, :2] # (B, 2)
    
    # 2. Normalize and scale the vector to pixel displacement range
    # Scaling factor (0.05) is a hyperparameter to control maximum displacement
    FLOW_SCALE = 0.05
    flow_vector_norm = flow_vector_unnorm * FLOW_SCALE * (H / 128) 

    # 3. Create a constant flow field across the image grid
    flow_field = flow_vector_norm.view(B, 1, 1, 2).repeat(1, H, W, 1)

    return flow_field

def simulate_torch(restored_image: torch.Tensor, lpd_mu: torch.Tensor):
    """
    Simulates tilt distortion on the restored image using LPD parameters.
    
    restored_image: (B, C, H, W) - The image to be degraded (the restored image).
    lpd_mu: (B, Z) - The mean of the LPD latent variable, used to predict tilt.
    
    Returns: simulated_degraded_image (B, C, H, W)
    """
    B, C, H, W = restored_image.shape
    device = restored_image.device

    # 1. Get the tilt-flow field (differentiable link)
    flow_field = predict_tilt_flow(lpd_mu, H, W, device) # (B, H, W, 2)

    # 2. Create the base grid (normalized coordinates)
    norm_x = torch.linspace(-1, 1, W, device=device)
    norm_y = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(norm_y, norm_x, indexing='ij')
    base_grid = torch.stack((grid_x, grid_y), dim=2) # (H, W, 2)

    # 3. Apply the flow to the base grid
    new_grid = base_grid.unsqueeze(0) + flow_field 
    
    # 4. Warp the image (differentiable operation)
    simulated_warped = F.grid_sample(restored_image, 
                                     new_grid.clamp(-1, 1), 
                                     mode='bilinear', 
                                     padding_mode='zeros', 
                                     align_corners=False)

    return simulated_warped.clamp(0.0, 1.0)
