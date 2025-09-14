"""Patch-level depth statistics extraction for VLM fusion."""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Union
import numpy as np


def depth_to_patch_stats(
    depth: torch.Tensor,
    patch_size: int,
    stats: Tuple[str, ...] = ("mean", "var", "grad"),
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert dense depth map to patch-level statistics.
    
    Args:
        depth: Dense depth map (B, 1, H, W)
        patch_size: Size of each patch
        stats: Statistics to compute ("mean", "var", "grad", "min", "max")
        eps: Small epsilon for numerical stability
        
    Returns:
        Patch statistics (B, N, len(stats)) where N = (H//patch_size) * (W//patch_size)
    """
    B, C, H, W = depth.shape
    assert C == 1, "Depth should have single channel"
    
    h_patches = H // patch_size
    w_patches = W // patch_size
    
    # Crop to multiple of patch_size if needed
    H_crop = h_patches * patch_size
    W_crop = w_patches * patch_size
    depth_crop = depth[:, :, :H_crop, :W_crop]
    
    # Reshape to patches: (B, 1, h_patches, patch_size, w_patches, patch_size)
    depth_patches = depth_crop.view(B, 1, h_patches, patch_size, w_patches, patch_size)
    
    # Rearrange to: (B, 1, patch_size, patch_size, h_patches, w_patches)
    depth_patches = depth_patches.permute(0, 1, 3, 5, 2, 4)
    
    # Flatten patches: (B, 1, patch_size*patch_size, N) where N = h_patches*w_patches
    patch_pixels = depth_patches.reshape(B, 1, patch_size * patch_size, h_patches * w_patches)
    
    stat_tensors = []
    
    for stat in stats:
        if stat == "mean":
            stat_values = patch_pixels.mean(dim=2)  # (B, 1, N)
            
        elif stat == "var":
            stat_values = patch_pixels.var(dim=2, unbiased=False)  # (B, 1, N)
            
        elif stat == "std":
            stat_values = patch_pixels.std(dim=2, unbiased=False)  # (B, 1, N)
            
        elif stat == "min":
            stat_values = patch_pixels.min(dim=2)[0]  # (B, 1, N)
            
        elif stat == "max":
            stat_values = patch_pixels.max(dim=2)[0]  # (B, 1, N)
            
        elif stat == "grad":
            # Compute gradient magnitude on full resolution, then average over patches
            grad_x = torch.abs(depth_crop[:, :, :, 1:] - depth_crop[:, :, :, :-1])
            grad_x = F.pad(grad_x, (1, 0, 0, 0), value=0)
            
            grad_y = torch.abs(depth_crop[:, :, 1:, :] - depth_crop[:, :, :-1, :])
            grad_y = F.pad(grad_y, (0, 0, 1, 0), value=0)
            
            grad_mag = grad_x + grad_y
            
            # Reshape gradient to patches and average
            grad_patches = grad_mag.view(B, 1, h_patches, patch_size, w_patches, patch_size)
            grad_patches = grad_patches.permute(0, 1, 3, 5, 2, 4)
            grad_patches = grad_patches.reshape(B, 1, patch_size * patch_size, h_patches * w_patches)
            stat_values = grad_patches.mean(dim=2)  # (B, 1, N)
            
        elif stat == "range":
            # Range = max - min
            min_vals = patch_pixels.min(dim=2)[0]
            max_vals = patch_pixels.max(dim=2)[0]
            stat_values = max_vals - min_vals  # (B, 1, N)
            
        elif stat == "median":
            stat_values = patch_pixels.median(dim=2)[0]  # (B, 1, N)
            
        else:
            raise ValueError(f"Unknown statistic: {stat}")
        
        stat_tensors.append(stat_values)
    
    # Combine statistics: (B, len(stats), N) -> (B, N, len(stats))
    patch_stats = torch.cat(stat_tensors, dim=1).transpose(1, 2)
    
    return patch_stats


class PatchStatsExtractor:
    """
    Utility class for extracting patch statistics with various configurations.
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        stats: Tuple[str, ...] = ("mean", "var", "grad"),
        normalize_stats: bool = True,
        log_transform_depth: bool = True,
    ):
        """
        Initialize patch statistics extractor.
        
        Args:
            patch_size: Size of patches to extract statistics from
            stats: Statistics to compute
            normalize_stats: Whether to normalize statistics
            log_transform_depth: Whether to apply log transform to depth before stats
        """
        self.patch_size = patch_size
        self.stats = stats
        self.normalize_stats = normalize_stats
        self.log_transform_depth = log_transform_depth
        
        # For normalization
        self.stat_means = {}
        self.stat_stds = {}
        self.is_fitted = False
    
    def extract(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Extract patch statistics from depth map.
        
        Args:
            depth: Dense depth map (B, 1, H, W)
            
        Returns:
            Patch statistics (B, N, len(stats))
        """
        # Optional log transform
        if self.log_transform_depth:
            depth_processed = torch.log(depth + 1e-8)
        else:
            depth_processed = depth
        
        # Extract statistics
        stats = depth_to_patch_stats(
            depth_processed, self.patch_size, self.stats
        )
        
        # Optional normalization
        if self.normalize_stats and self.is_fitted:
            stats = self._normalize_stats(stats)
        
        return stats
    
    def fit_normalization(self, depth_maps: List[torch.Tensor]) -> None:
        """
        Fit normalization parameters from a collection of depth maps.
        
        Args:
            depth_maps: List of depth maps to compute statistics from
        """
        all_stats = []
        
        for depth in depth_maps:
            stats = self.extract(depth)
            all_stats.append(stats.view(-1, len(self.stats)))
        
        # Combine all statistics
        combined_stats = torch.cat(all_stats, dim=0)  # (total_patches, num_stats)
        
        # Compute mean and std for each statistic
        for i, stat_name in enumerate(self.stats):
            stat_values = combined_stats[:, i]
            self.stat_means[stat_name] = stat_values.mean().item()
            self.stat_stds[stat_name] = stat_values.std().item()
        
        self.is_fitted = True
        print(f"Fitted normalization parameters from {len(depth_maps)} depth maps")
    
    def _normalize_stats(self, stats: torch.Tensor) -> torch.Tensor:
        """Normalize statistics using fitted parameters."""
        normalized = torch.zeros_like(stats)
        
        for i, stat_name in enumerate(self.stats):
            mean = self.stat_means[stat_name]
            std = self.stat_stds[stat_name]
            normalized[..., i] = (stats[..., i] - mean) / (std + 1e-8)
        
        return normalized
    
    def create_embedding_tokens(
        self,
        depth: torch.Tensor,
        token_dim: int,
        method: str = "linear",
    ) -> torch.Tensor:
        """
        Create embedding tokens from depth statistics.
        
        Args:
            depth: Dense depth map (B, 1, H, W)
            token_dim: Dimension of output tokens
            method: Method for creating tokens ("linear", "mlp")
            
        Returns:
            Depth embedding tokens (B, N, token_dim)
        """
        # Extract statistics
        stats = self.extract(depth)  # (B, N, num_stats)
        B, N, num_stats = stats.shape
        
        if method == "linear":
            # Simple linear projection
            if not hasattr(self, "stat_projection"):
                self.stat_projection = torch.nn.Linear(num_stats, token_dim)
            
            tokens = self.stat_projection(stats)
            
        elif method == "mlp":
            # MLP projection
            if not hasattr(self, "stat_mlp"):
                hidden_dim = max(64, token_dim // 2)
                self.stat_mlp = torch.nn.Sequential(
                    torch.nn.Linear(num_stats, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, token_dim),
                )
            
            tokens = self.stat_mlp(stats)
            
        else:
            raise ValueError(f"Unknown token creation method: {method}")
        
        return tokens
    
    def save_normalization_params(self, path: str) -> None:
        """Save normalization parameters to file."""
        if not self.is_fitted:
            raise ValueError("Must fit normalization before saving")
        
        params = {
            "stat_means": self.stat_means,
            "stat_stds": self.stat_stds,
            "patch_size": self.patch_size,
            "stats": self.stats,
            "normalize_stats": self.normalize_stats,
            "log_transform_depth": self.log_transform_depth,
        }
        
        torch.save(params, path)
        print(f"Saved normalization parameters to {path}")
    
    def load_normalization_params(self, path: str) -> None:
        """Load normalization parameters from file."""
        params = torch.load(path, map_location="cpu")
        
        self.stat_means = params["stat_means"]
        self.stat_stds = params["stat_stds"]
        self.patch_size = params["patch_size"]
        self.stats = params["stats"]
        self.normalize_stats = params["normalize_stats"]
        self.log_transform_depth = params["log_transform_depth"]
        self.is_fitted = True
        
        print(f"Loaded normalization parameters from {path}")


def visualize_patch_stats(
    depth: torch.Tensor,
    patch_stats: torch.Tensor,
    stat_names: Tuple[str, ...],
    patch_size: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize patch statistics overlaid on depth map.
    
    Args:
        depth: Original depth map (1, 1, H, W)
        patch_stats: Patch statistics (1, N, num_stats)
        stat_names: Names of statistics
        patch_size: Size of patches
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    depth_np = depth[0, 0].cpu().numpy()
    stats_np = patch_stats[0].cpu().numpy()  # (N, num_stats)
    
    H, W = depth_np.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    
    fig, axes = plt.subplots(1, len(stat_names) + 1, figsize=(4 * (len(stat_names) + 1), 4))
    
    # Original depth
    axes[0].imshow(depth_np, cmap="viridis")
    axes[0].set_title("Original Depth")
    axes[0].axis("off")
    
    # Statistics
    for i, stat_name in enumerate(stat_names):
        stat_values = stats_np[:, i].reshape(h_patches, w_patches)
        
        # Upsample to original resolution for visualization
        stat_upsampled = np.repeat(stat_values, patch_size, axis=0)
        stat_upsampled = np.repeat(stat_upsampled, patch_size, axis=1)
        
        # Crop to original size
        stat_upsampled = stat_upsampled[:H, :W]
        
        im = axes[i + 1].imshow(stat_upsampled, cmap="plasma")
        axes[i + 1].set_title(f"Patch {stat_name}")
        axes[i + 1].axis("off")
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    plt.show()
