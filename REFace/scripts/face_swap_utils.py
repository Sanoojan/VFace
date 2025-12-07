import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import os 
import numpy as np
from PIL import Image


def save_fft_magnitude_plot(
    fft_q: torch.Tensor,
    batch_idx: int = 0,
    channel_idx: int = 0,
    filename: str = "fft_magnitude.png",
    output_dir: str = "./fft_vis",
    use_log: bool = False
):
    """
    Save the FFT magnitude spectrum of a specific batch and channel as an image.

    Args:
        fft_q (Tensor): Complex-valued FFT tensor of shape (B, C, D)
        batch_idx (int): Batch index to visualize
        channel_idx (int): Channel index to visualize
        filename (str): Output filename
        output_dir (str): Directory to save the plot
        use_log (bool): Whether to use log magnitude for better visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    fft_selected = fft_q[batch_idx, channel_idx, :].detach().cpu()

    # Compute magnitude
    magnitude = torch.abs(fft_selected).numpy()
    if use_log:
        magnitude = np.log1p(magnitude)  # log(1 + |x|)

    # Optional: center zero-frequency (if needed)
    magnitude = np.fft.fftshift(magnitude)

    # Plot and save
    plt.figure(figsize=(8, 3))
    plt.plot(magnitude, label='|FFT|')
    plt.title(f"FFT Magnitude - Batch {batch_idx}, Channel {channel_idx}")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude (log)" if use_log else "Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved FFT magnitude plot to: {save_path}")

def save_first5_attention_maps(
    combined: torch.Tensor,
    batch_idx: int = 0,
    filename: str = "combined_attention_5ch.png",
    output_dir: str = "./attn_vis"
):
    """
    Save the first 5 attention channels (reshaped to 64x64) as a horizontal grid.

    Args:
        combined (torch.Tensor): Tensor of shape (B, 4096, C)
        batch_idx (int): Which batch index to visualize
        filename (str): Name of the output image
        output_dir (str): Directory to save image
    """
    os.makedirs(output_dir, exist_ok=True)

    att = combined[batch_idx].detach().cpu().numpy()  # (4096, 320)
    tokens, channels = att.shape
    h, w = 64, 64

    if tokens != h * w:
        raise ValueError(f"Cannot reshape {tokens} tokens to {h}x{w}")

    num_channels_to_plot = min(5, channels)

    # Transpose to (C, H*W) -> (C, H, W)
    att = att.T[:num_channels_to_plot].reshape(num_channels_to_plot, h, w)

    # Normalize each channel to [0, 255]
    att_norm = (att - att.min(axis=(1, 2), keepdims=True)) / \
               (att.max(axis=(1, 2), keepdims=True) - att.min(axis=(1, 2), keepdims=True) + 1e-8)
    att_uint8 = (att_norm * 255).astype(np.uint8)

    # Stack horizontally: (H, num_channels * W)
    grid_img = np.hstack(att_uint8)

    # Save
    out_path = os.path.join(output_dir, filename)
    Image.fromarray(grid_img).save(out_path)
    print(f"Saved first 5 channel attention grid to {out_path}")

def save_combined_attention(
    combined: torch.Tensor,
    batch_idx: int = 0,
    filename: str = "combined_attention.png",
    output_dir: str = "./attn_vis"
):
    """
    Save the attention tensor (B, 4096, C) as a grid of (64x64) grayscale images, one per channel.

    Args:
        combined (torch.Tensor): Tensor of shape (B, 4096, C)
        batch_idx (int): Which batch index to use
        filename (str): Name of the output image
        output_dir (str): Directory to save image
    """
    os.makedirs(output_dir, exist_ok=True)

    att = combined[batch_idx].detach().cpu().numpy()  # Shape: (4096, 320)
    h, w = 64, 64
    tokens, channels = att.shape

    if tokens != h * w:
        raise ValueError(f"Cannot reshape {tokens} tokens to {h}x{w}. Please adjust.")

    # Transpose to (C, H*W) → (C, H, W)
    att = att.T.reshape(channels, h, w)

    # Normalize each channel separately to [0, 255]
    att_norm = (att - att.min(axis=(1, 2), keepdims=True)) / \
               (att.max(axis=(1, 2), keepdims=True) - att.min(axis=(1, 2), keepdims=True) + 1e-8)
    att_uint8 = (att_norm * 255).astype(np.uint8)

    # Arrange into a grid: auto-calculate grid size
    grid_rows = int(np.floor(np.sqrt(channels)))
    grid_cols = int(np.ceil(channels / grid_rows))
    grid_img = np.zeros((grid_rows * h, grid_cols * w), dtype=np.uint8)

    for i in range(channels):
        r = i // grid_cols
        c = i % grid_cols
        grid_img[r*h:(r+1)*h, c*w:(c+1)*w] = att_uint8[i]

    # Save image
    out_path = os.path.join(output_dir, filename)
    Image.fromarray(grid_img).save(out_path)
    print(f"Saved attention visualization to {out_path}")

def save_combined_attention(
    combined: torch.Tensor,
    batch_idx: int = 0,
    channel_idx: int = 0,
    filename: str = "combined_attention.png",
    reshape_to_square: bool = True,
    output_dir: str = "./attn_vis"
):
    """
    Save the combined attention map as an image.

    Args:
        combined (torch.Tensor): Tensor of shape (b, c, d)
        batch_idx (int): Which batch index to visualize
        channel_idx (int): Which channel index to visualize
        filename (str): Output filename
        reshape_to_square (bool): If True, reshape d to sqrt(d) × sqrt(d) if possible
        output_dir (str): Folder to save the image
    """
    os.makedirs(output_dir, exist_ok=True)
    att = combined[batch_idx, channel_idx, :].detach().cpu().numpy()

    if reshape_to_square:
        d = att.shape[0]
        side = int(np.sqrt(d))
        if side * side == d:
            att = att.reshape((side, side))
        else:
            raise ValueError(f"Cannot reshape dimension {d} to a square. Set reshape_to_square=False to save as 1D.")

    # Normalize to 0–255
    att = (att - att.min()) / (att.max() - att.min() + 1e-8)  # normalize to [0, 1]
    att_img = (att * 255).astype(np.uint8)

    # Save
    image = Image.fromarray(att_img)
    if att_img.ndim == 2:
        image = image.convert("L")  # grayscale
    elif att_img.ndim == 3:
        image = image.convert("RGB")

    save_path = os.path.join(output_dir, filename)
    image.save(save_path)
    print(f"Saved attention image to {save_path}")

def mix_source_and_target(target,source, alpha=0.5):
    """
    Mixes the source and target images based on the given alpha value.
    :param source: Source image
    :param target: Target image
    :param alpha: Mixing ratio (0.0 to 1.0)
    :return: Mixed image
    """
    
    
    return (1 - alpha) * source + alpha * target

# def fft_fusion(noise_A, noise_B, center=16,center_exclude=3):
#     # Apply FFT over H and W for each batch and channel
#     fft_A = torch.fft.fft2(noise_A, dim=(-2, -1))
#     fft_B = torch.fft.fft2(noise_B, dim=(-2, -1))
    
#     fft_A_shift = torch.fft.fftshift(fft_A, dim=(-2, -1))
#     fft_B_shift = torch.fft.fftshift(fft_B, dim=(-2, -1))
#     # breakpoint()
    
#     B, C, H, W = noise_A.shape
#     mask = torch.zeros((H, W), device=noise_A.device)
#     cx, cy = H // 2, W // 2
#     mask[cx - center:cx + center, cy - center:cy + center] = 1
#     mask[cx - center_exclude:cx + center_exclude, cy - center_exclude:cy + center_exclude] = 0
#     mask = mask[None, None, :, :]  # shape: (1, 1, H, W)
    
#     combined_fft = fft_A_shift * (1-mask) + fft_B_shift * (mask)
#     # combined_fft=fft_B_shift
#     combined_fft = torch.fft.ifftshift(combined_fft, dim=(-2, -1))
    
    
#     combined = torch.fft.ifft2(combined_fft, dim=(-2, -1)).real
#     return combined

def fft_fusion(noise_A, noise_B, center=16, center_exclude=3):
    # Apply FFT over H and W for each batch and channel
    fft_A = torch.fft.fft2(noise_A, dim=(-2, -1))
    fft_B = torch.fft.fft2(noise_B, dim=(-2, -1))

    fft_A_shift = torch.fft.fftshift(fft_A, dim=(-2, -1))
    fft_B_shift = torch.fft.fftshift(fft_B, dim=(-2, -1))

    B, C, H, W = noise_A.shape
    cx, cy = H // 2, W // 2

    # Create a circular mask instead of a square one
    Y, X = torch.meshgrid(torch.arange(H, device=noise_A.device), 
                          torch.arange(W, device=noise_A.device), indexing='ij')
    dist = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = ((dist <= center) & (dist > center_exclude)).float()
    mask = mask[None, None, :, :]  # shape: (1, 1, H, W)

    # Blend FFTs using the circular mask
    combined_fft = fft_A_shift * (1 - mask) + fft_B_shift * mask

    # Inverse FFT to return to the spatial domain
    combined_fft = torch.fft.ifftshift(combined_fft, dim=(-2, -1))
    combined = torch.fft.ifft2(combined_fft, dim=(-2, -1)).real

    return combined

def fft_fusion_warp(noise_A, noise_B, center=16, center_exclude=3,lm_src=None,lm_tar=None):
    # Apply FFT over H and W for each batch and channel
    fft_A = torch.fft.fft2(noise_A, dim=(-2, -1))
    fft_B = torch.fft.fft2(noise_B, dim=(-2, -1))

    fft_A_shift = torch.fft.fftshift(fft_A, dim=(-2, -1))
    fft_B_shift = torch.fft.fftshift(fft_B, dim=(-2, -1))

    B, C, H, W = noise_A.shape
    cx, cy = H // 2, W // 2

    # Create a circular mask instead of a square one
    Y, X = torch.meshgrid(torch.arange(H, device=noise_A.device), 
                          torch.arange(W, device=noise_A.device), indexing='ij')
    dist = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = ((dist <= center) & (dist > center_exclude)).float()
    mask = mask[None, None, :, :]  # shape: (1, 1, H, W)

    # Blend FFTs using the circular mask
    combined_fft = fft_A_shift * (1 - mask) + fft_B_shift * mask

    # Inverse FFT to return to the spatial domain
    combined_fft = torch.fft.ifftshift(combined_fft, dim=(-2, -1))
    combined = torch.fft.ifft2(combined_fft, dim=(-2, -1)).real

    return combined



def lpf_fusion(noise_A, noise_B, kernel_size=5, sigma=1.0):
    # Apply LPF over H and W for each batch and channel
    # Structure from low-pass of noise_A
    B, C, H, W = noise_A.shape

    def gaussian_blur(x, kernel_size=5, sigma=1.0):
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid = coords[None, :]**2 + coords[:, None]**2
        kernel = torch.exp(-grid / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Expand to match depthwise conv shape
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(x.device)
        kernel = kernel.repeat(C, 1, 1, 1)  # Shape: (C, 1, k, k)

        return F.conv2d(x, kernel, padding=kernel_size // 2, groups=C)

    # Low-pass from structure source (A)
    structure_part = gaussian_blur(noise_A, kernel_size, sigma)

    # High-pass from identity source (B)
    blurred_B = gaussian_blur(noise_B, kernel_size, sigma)
    identity_part = noise_B - blurred_B

    # Final fusion
    combined_latent = structure_part + identity_part
    return combined_latent

# def AdaIn_fusion(noise_A, noise_B, alpha=0.71,normalized=True):
#     # Apply AdaIn over H and W for each batch and channel
#     # B, C, H, W = noise_A.shape

#     # Compute mean and std for both noise_A and noise_B
#     mean_A = noise_A.mean(dim=(2,3), keepdim=True)
#     std_A = noise_A.std(dim=(2,3), keepdim=True)
#     mean_B = noise_B.mean(dim=(2,3), keepdim=True)
#     std_B = noise_B.std(dim=(2,3), keepdim=True)

#     # Normalize noise_A
#     normalized_A = (noise_A - mean_A) / (std_A + 1e-5)

#     # Scale and shift with noise_B's statistics
#     fused_noise = normalized_A * std_B + mean_B
    
    
#     # normalize the fused noise
#     if normalized:
#         # Normalize the fused noise
#         fused_noise = (fused_noise) / (fused_noise.std() + 1e-5)
#         return fused_noise
#     else:
#         # Standardize the fused noise
#         return alpha*fused_noise
    
    
def AdaIn_fusion(noise_A, noise_B, alpha=0.71,beta=1.0, normalized=True):
    """
    Apply Adaptive Instance Normalization (AdaIN) fusion.
    noise_A: structure source (B, C, H, W)
    noise_B: identity source (B, C, H, W)
    alpha: blend factor (0 -> only noise_A, 1 -> fully fused)
    normalized: if True, normalize noise_A first
    """
    # Compute per-channel mean and std
    mean_A = noise_A.mean(dim=(2,3), keepdim=True)
    std_A = noise_A.std(dim=(2,3), keepdim=True)
    mean_B = noise_B.mean(dim=(2,3), keepdim=True)
    std_B = noise_B.std(dim=(2,3), keepdim=True)

    if normalized:
        # Normalize noise_A
        normalized_A = (noise_A - mean_A) / (std_A + 1e-5)
    else:
        normalized_A = noise_A  # skip normalization if requested

    # Perform AdaIN: shift and scale to noise_B stats
    fused = normalized_A * (std_B + 1e-5) + mean_B

    # Interpolate between original and fused
    output = (1 - alpha) * noise_A + alpha * fused

    return output*beta

def AdaIn_fusion_for_attn(noise_A, noise_B, alpha=0.71,normalized=True):
    # Apply AdaIn over H and W for each batch and channel
    # B, C, H, W = noise_A.shape

    # Compute mean and std for both noise_A and noise_B
    mean_A = noise_A.mean(dim= -1, keepdim=True)
    std_A = noise_A.std(dim=-1, keepdim=True)
    mean_B = noise_B.mean(dim=-1, keepdim=True)
    std_B = noise_B.std(dim=-1, keepdim=True)

    # Normalize noise_A
    normalized_A = (noise_A - mean_A) / (std_A + 1e-5)

    # Scale and shift with noise_B's statistics
    fused_noise = normalized_A * std_B + mean_B
    
    
    # normalize the fused noise
    if normalized:
        # Normalize the fused noise
        fused_noise = (fused_noise) / (fused_noise.std() + 1e-5)
        return fused_noise
    else:
        # Standardize the fused noise
        return alpha*fused_noise


def fft_fusion_for_attn(noise_A, noise_B, center=16, center_exclude=3):
    # Apply FFT over H and W for each batch and channel
    # convert to folat
    noise_A = noise_A.float()
    noise_B = noise_B.float()
    fft_A = torch.fft.fft2(noise_A, dim=(-2, -1))
    fft_B = torch.fft.fft2(noise_B, dim=(-2, -1))

    fft_A_shift = torch.fft.fftshift(fft_A, dim=(-2, -1))
    fft_B_shift = torch.fft.fftshift(fft_B, dim=(-2, -1))

    B, C, H, W = noise_A.shape
    cx, cy = H // 2, W // 2

    # Create a circular mask instead of a square one
    Y, X = torch.meshgrid(torch.arange(H, device=noise_A.device), 
                          torch.arange(W, device=noise_A.device), indexing='ij')
    dist = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = ((dist <= center) & (dist > center_exclude)).float()
    mask = mask[None, None, :, :]  # shape: (1, 1, H, W)

    # Blend FFTs using the circular mask
    combined_fft = fft_A_shift * (1 - mask) + fft_B_shift * mask

    # Inverse FFT to return to the spatial domain
    combined_fft = torch.fft.ifftshift(combined_fft, dim=(-2, -1))
    combined = torch.fft.ifft2(combined_fft, dim=(-2, -1)).real
    
    combined=combined.to(torch.float32)
    # Ensure the output is float32

    return combined

def combine_fft_high_low(q1, q2, split_ratio=0.5):
    """
    Combine high-frequency components from q1 and low-frequency components from q2.
    
    Args:
        q1: Tensor of shape (b, c, d)
        q2: Tensor of shape (b, c, d)
        split_ratio: Ratio to split between low and high frequency. Default is 0.5 (middle).
    
    Returns:
        Tensor of shape (b, c, d) with combined frequency components.
        
    """
    
    if q1.shape[1]==4096:
        print("")
    
    
    q1 = q1.float()  # Ensure q1 is float
    q2 = q2.float()  # Ensure q2 is float
    # FFT along the last dimension
    fft_q1 = torch.fft.fft(q1, dim=-1)
    fft_q2 = torch.fft.fft(q2, dim=-1)

    # Determine split point
    d = q1.size(-1)
    split_point = int(d * split_ratio)

    # Initialize combined FFT
    fft_combined = torch.zeros_like(fft_q1)

    # Copy low frequencies from q2, high from q1
    fft_combined[..., :split_point] = fft_q2[..., :split_point]   # low frequency
    fft_combined[..., split_point:] = fft_q1[..., split_point:]   # high frequency

    # Inverse FFT to get back to time domain
    combined = torch.fft.ifft(fft_combined, dim=-1).real  # Discard imaginary part
    combined = combined.to(torch.float32)  # Ensure the output is float32

    return combined
 
def plot_fft_3d(latent_tensor, batch_idx=0, channel_idx=0, log_scale=True,save_path="out.png"):
    """
    latent_tensor: torch.Tensor of shape (B, C, H, W)
    batch_idx: index of batch to visualize
    channel_idx: index of channel to visualize
    log_scale: whether to apply log(1 + magnitude) for better visualization
    """
    # Select one channel from the batch
    selected = latent_tensor[batch_idx, channel_idx]  # shape: (H, W)
    
    # Compute 2D FFT and shift
    fft2d = torch.fft.fft2(selected)
    fft2d_shifted = torch.fft.fftshift(fft2d)
    magnitude = torch.abs(fft2d_shifted)

    # Optional log scale for better dynamic range
    if log_scale:
        magnitude = torch.log1p(magnitude)

    # Prepare mesh grid
    H, W = selected.shape
    X, Y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.numpy(), Y.numpy(), magnitude.cpu().numpy(), cmap='viridis')

    ax.set_title(f"3D FFT Spectrum (B={batch_idx}, C={channel_idx})")
    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel("Magnitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
 
 
 
 
 
 
 
 
 
 
 # standardize start code
# start_code = (start_code - start_code.mean()) / start_code.std()

# check this hyper parameter
# start_code=(x_noisy_target+x_noisy_src)/1.41

# start_code=x_noisy_src


# Optional
# inp_mask=test_model_kwargs['inpaint_mask']
# inp_mask=inp_mask.repeat(1, 4, 1, 1)
# start_code[inp_mask==1.0]=x_noisy_target[inp_mask==1.0]

# start_code=x_noisy_target
# start_code_noise=torch.randn_like(start_code)
# alpha = 0.5  # Adjust this
# start_code = alpha * start_code + (1 - alpha) * torch.randn_like(start_code)
# start_code=start_code/0.7
# noise = torch.randn_like(z)

# x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
# start_code = x_noisy