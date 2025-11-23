"""
MPH_model_skeleton2.py
Mirage Image Correction Model Skeleton for Thesis Project
---------------------------------------------------------
[... all class definitions remain unchanged ...]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# Helper: Basic Convolutional Block
# ---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


# ---------------------------------------------------------------
# 1. Tilt Estimation Module (Coarse optical flow-like predictor)
# ---------------------------------------------------------------
class TiltEstimator(nn.Module):
    def __init__(self, in_ch=3, feat_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, feat_ch, 3, 1, 1),
            ConvBlock(feat_ch, feat_ch, 3, 1, 1),
            ConvBlock(feat_ch, 2, 3, 1, 1, act=False)  # output 2D flow
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        flow = self.net(x)
        flow = flow.view(B, T, 2, H, W)
        return flow


# ---------------------------------------------------------------
# 2. Feature Extraction Module
# ---------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, in_ch=3, feat_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, feat_ch, 3, 1, 1),
            ConvBlock(feat_ch, feat_ch, 3, 2, 1),
            ConvBlock(feat_ch, feat_ch, 3, 1, 1),
            ConvBlock(feat_ch, feat_ch, 3, 1, 1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        return self.encoder(x)


# ---------------------------------------------------------------
# 3. Alignment Module (simple attention-like aggregation)
# ---------------------------------------------------------------
class AlignmentModule(nn.Module):
    def __init__(self, feat_ch=64):
        super().__init__()
        self.query = nn.Conv2d(feat_ch, feat_ch // 8, 1)
        self.key = nn.Conv2d(feat_ch, feat_ch // 8, 1)
        self.value = nn.Conv2d(feat_ch, feat_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feats):
        # feats: (B, T, C, H, W)
        B, T, C, H, W = feats.shape
        center = feats[:, T // 2]
        q = self.query(center).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, Cq)
        outputs = []
        for t in range(T):
            k = self.key(feats[:, t]).view(B, -1, H * W)
            v = self.value(feats[:, t]).view(B, -1, H * W)
            attn = torch.bmm(q, k)
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(v, attn.permute(0, 2, 1))
            out = out.view(B, C, H, W)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        aligned = center + self.gamma * outputs.mean(dim=1)
        return aligned


# ---------------------------------------------------------------
# 4. Simple State-Space Fusion Block
# ---------------------------------------------------------------
class SimpleSSMBlock(nn.Module):
    def __init__(self, feat_ch=64):
        super().__init__()
        self.conv_in = ConvBlock(feat_ch, feat_ch)
        self.conv_gate = nn.Conv2d(feat_ch, feat_ch, 1)
        self.conv_out = ConvBlock(feat_ch, feat_ch)

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.conv_in(x)
        g = torch.sigmoid(self.conv_gate(h))
        h = h * g
        out = self.conv_out(h)
        return out


# ---------------------------------------------------------------
# 5. Latent Phase Distortion (LPD) Variational Head
# ---------------------------------------------------------------
class LPDVAE(nn.Module):
    def __init__(self, feat_ch=64, z_dim=32):
        super().__init__()
        self.fc_mu = nn.Linear(feat_ch, z_dim)
        self.fc_logvar = nn.Linear(feat_ch, z_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        pooled = F.adaptive_avg_pool2d(x, 1).view(B, C)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar


# ---------------------------------------------------------------
# 6. Decoder (U-Net-like upsampling)
# ---------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, in_ch=64, out_ch=3):
        super().__init__()
        self.decode = nn.Sequential(
            ConvBlock(in_ch, in_ch, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(in_ch, in_ch // 2, 3, 1, 1),
            ConvBlock(in_ch // 2, in_ch // 4, 3, 1, 1),
            nn.Conv2d(in_ch // 4, out_ch, 3, 1, 1)
        )

    def forward(self, x):
        return self.decode(x)


# ---------------------------------------------------------------
# 7. Full Model Assembly
# ---------------------------------------------------------------
class MPHModel(nn.Module):
    def __init__(self, in_ch=3, feat_ch=64, z_dim=32, T=5):
        super().__init__()
        self.T = T
        self.tilt = TiltEstimator(in_ch, feat_ch // 2)
        self.fe = FeatureExtractor(in_ch, feat_ch)
        self.align = AlignmentModule(feat_ch)
        self.ssm = SimpleSSMBlock(feat_ch)
        self.vae = LPDVAE(feat_ch, z_dim)
        self.decoder = Decoder(feat_ch, in_ch)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # 1. Tilt estimation
        flow = self.tilt(x)

        # 2. Warp frames using predicted tilt
        aligned_frames = []
        for t in range(T):
            grid = self._flow_to_grid(flow[:, t], H, W)
            aligned = F.grid_sample(x[:, t], grid, mode='bilinear', padding_mode='border')
            aligned_frames.append(aligned)
        aligned_stack = torch.stack(aligned_frames, dim=1)

        # 3. Feature extraction
        feats = []
        for t in range(T):
            feats.append(self.fe(aligned_stack[:, t]))
        feats = torch.stack(feats, dim=1)  # (B, T, feat_ch, H', W')

        # 4. Alignment & fusion
        aligned_feat = self.align(feats)
        fused_feat = self.ssm(aligned_feat)

        # 5. VAE head
        mu, logvar = self.vae(fused_feat)

        # 6. Decode restored image
        restored = self.decoder(fused_feat)

        return restored, mu, logvar

    def _flow_to_grid(self, flow, H, W):
        """Convert 2D flow map to normalized grid for grid_sample"""
        B, _, h, w = flow.shape
        norm_x = torch.linspace(-1, 1, W, device=flow.device)
        norm_y = torch.linspace(-1, 1, H, device=flow.device)
        grid_y, grid_x = torch.meshgrid(norm_y, norm_x, indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1) / torch.tensor([W / 2, H / 2], device=flow.device)
        return grid.clamp(-1, 1)


# ---------------------------------------------------------------
# Built-in Smoke Test (Run this file directly)
# ---------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create random input tensor (Batch=1, T=5, C=3, H=W=128)
    # CRITICAL CHANGE: Resolution reduced to 128x128 to solve persistent CUDA OOM
    x = torch.rand(1, 5, 3, 128, 128, device=device)

    model = MPHModel(in_ch=3, feat_ch=64, z_dim=32, T=5).to(device)
    restored, mu, logvar = model(x)

    print("restored:", restored.shape)
    print("mu/logvar:", mu.shape, logvar.shape)
    print("Smoke test passed successfully!")
