import torch
import torch.nn.functional as F
import random
import math
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d

class TimeSeriesAugment(nn.Module):
    def __init__(
        self,
        noise_std=0.002,
        mask_prob=0.05,
        dropout_prob=0.05,
        insert_prob=0.05,
        scale_range=None,
        shift_max=0,
        jitter_std=0,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.dropout_prob = dropout_prob
        self.insert_prob = insert_prob
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.jitter_std = jitter_std
        self.base = 1e-3
        self.scale = 41
        self.start = 30

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        device = x.device
        # Clone to avoid modifying original 
        aug = x.clone()
        if random.random() < 0.75:
            warp_mat = torch.tile(torch.unsqueeze(torch.eye(C), dim = 0), (B, 1, 1)).to(device)
            warp_mat += torch.randn_like(warp_mat, device=device) * random.randint(self.start,self.scale)*self.base
            aug = torch.matmul(aug.transpose(2,1), warp_mat).transpose(2,1)

        if random.random()<0.75:
            noise = torch.randn_like(aug) * random.randint(self.start,self.scale)*self.base
            aug += noise

        if random.random()<0.75:
            if random.random()<0.25:
                mask = torch.rand(B, 1, T, device=device) < random.randint(self.start,self.scale)*self.base*1.25
                aug = aug * (~mask)
            if random.random()<0.80:
                mask = (torch.rand(1, 1, T, device=device) > random.randint(self.start,self.scale*2)*self.base)  # shape [1,1,T]
                keep = mask[0, 0]        # boolean vector [T]
                x_new = aug[:, :, keep]    # keeps columns where mask is True
                T_new = x_new.size(2)
                aug = torch.zeros(B, C, T, device=x.device, dtype=x.dtype)
                aug[:, :, :T_new] = x_new

        # if random.random() < 0.20:
        #     aug = aug[:, :, random.randint(1,4):]                       # [B, C, L_kept]
        #         if pad_len > 0:
        #             pad = torch.zeros(B, C, pad_len, device=device, dtype=aug.dtype)
        #             aug = torch.cat([aug, pad], dim=-1)    # [B, C, target_length]
            # if random.random() < 0.8:
            #     pl = random.randint(1,9)
            #     pad = torch.zeros(B, C, pl, device=device, dtype=aug.dtype)
            #     aug = torch.cat([pad,aug], dim=-1)
            #     pad2 = torch.zeros(B, C, pl%6, device=device, dtype=aug.dtype)
            #     aug = torch.cat([aug,pad2], dim=-1)
            #     if pl>=6:
            #         mask = (torch.rand(C, device=device) < 0.1)
            #         shift = random.randint(1,pl//6+1)
            #         aug[:,mask,shift:-(pl-shift)] = aug[:,mask,pl:]
#         add random walk noise
        # if random.random() < 0.6:
        #     aug += torch.cumsum(torch.randn(aug.shape, device=device)* random.randint(10,20)*1e-3, dim = -1)
        return aug

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=21,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed  # [B, T, C]
