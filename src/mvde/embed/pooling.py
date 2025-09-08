"""Patch to ROI pooling for per-object embeddings."""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Union
import numpy as np


class ROIPooler:
    """Pool patch embeddings within bounding box regions."""
    
    def __init__(self, pooling_method: str = "mean"):
        """
        Initialize ROI pooler.
        
        Args:
            pooling_method: Pooling strategy ("mean", "max", "adaptive")
        """
        self.pooling_method = pooling_method
    
    def pool_roi(
        self,
        patch_embeddings: torch.Tensor,
        bbox: Tuple[int, int, int, int],
        patch_grid_size: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Pool embeddings within a bounding box region.
        
        Args:
            patch_embeddings: Patch embeddings [num_patches, embed_dim]
            bbox: Bounding box (x1, y1, x2, y2) in image coordinates
            patch_grid_size: Grid size of patches (height, width)
            image_size: Original image size (height, width)
            
        Returns:
            Pooled embedding for the ROI
        """
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_size
        grid_h, grid_w = patch_grid_size
        
        # Convert bbox to patch coordinates
        patch_x1 = int((x1 / img_w) * grid_w)
        patch_y1 = int((y1 / img_h) * grid_h)
        patch_x2 = int((x2 / img_w) * grid_w)
        patch_y2 = int((y2 / img_h) * grid_h)
        
        # Clamp to valid range
        patch_x1 = max(0, min(patch_x1, grid_w - 1))
        patch_y1 = max(0, min(patch_y1, grid_h - 1))
        patch_x2 = max(patch_x1 + 1, min(patch_x2, grid_w))
        patch_y2 = max(patch_y1 + 1, min(patch_y2, grid_h))
        
        # Reshape to grid
        embed_dim = patch_embeddings.shape[-1]
        patch_grid = patch_embeddings.view(grid_h, grid_w, embed_dim)
        
        # Extract ROI
        roi_patches = patch_grid[patch_y1:patch_y2, patch_x1:patch_x2]
        roi_flat = roi_patches.view(-1, embed_dim)
        
        # Pool
        if self.pooling_method == "mean":
            return roi_flat.mean(dim=0)
        elif self.pooling_method == "max":
            return roi_flat.max(dim=0)[0]
        elif self.pooling_method == "adaptive":
            # Weighted by patch importance (placeholder)
            return roi_flat.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")


class PatchPooler:
    """Pool patch embeddings to object-level representations."""
    
    def __init__(self, method: str = "attention"):
        """
        Initialize patch pooler.
        
        Args:
            method: Pooling method ("attention", "mean", "max", "cls")
        """
        self.method = method
    
    def pool_patches(
        self,
        patch_embeddings: torch.Tensor,
        attention_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Pool patch embeddings to single representation.
        
        Args:
            patch_embeddings: Patch embeddings [num_patches, embed_dim]
            attention_weights: Optional attention weights [num_patches]
            
        Returns:
            Pooled embedding [embed_dim]
        """
        if self.method == "mean":
            return patch_embeddings.mean(dim=0)
        
        elif self.method == "max":
            return patch_embeddings.max(dim=0)[0]
        
        elif self.method == "cls":
            # Assume first token is CLS token
            return patch_embeddings[0]
        
        elif self.method == "attention":
            if attention_weights is None:
                # Compute simple attention weights
                attention_weights = self._compute_attention(patch_embeddings)
            
            # Weighted sum
            weighted_embeddings = patch_embeddings * attention_weights.unsqueeze(-1)
            return weighted_embeddings.sum(dim=0)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.method}")
    
    def _compute_attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute attention weights for patch embeddings."""
        # Simple attention: project to scalar and softmax
        scores = embeddings.norm(dim=-1)  # L2 norm as importance
        weights = F.softmax(scores, dim=0)
        return weights
    
    def pool_multi_scale(
        self,
        patch_embeddings_list: List[torch.Tensor],
        scales: List[float] = None,
    ) -> torch.Tensor:
        """
        Pool embeddings from multiple scales.
        
        Args:
            patch_embeddings_list: List of patch embeddings at different scales
            scales: Optional scale weights
            
        Returns:
            Multi-scale pooled embedding
        """
        if scales is None:
            scales = [1.0] * len(patch_embeddings_list)
        
        pooled_embeddings = []
        for embeddings, scale in zip(patch_embeddings_list, scales):
            pooled = self.pool_patches(embeddings)
            pooled_embeddings.append(pooled * scale)
        
        # Combine across scales
        return torch.stack(pooled_embeddings).mean(dim=0)
