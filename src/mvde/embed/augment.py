"""Embedding augmentation with distance information."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class EmbeddingAugmenter(nn.Module):
    """Augment vision embeddings with distance information."""
    
    def __init__(
        self,
        vision_dim: int,
        distance_dim: int = 1,
        method: str = "concat",
        projection_dim: Optional[int] = None,
        normalize_distance: bool = True,
    ):
        """
        Initialize embedding augmenter.
        
        Args:
            vision_dim: Dimension of vision embeddings
            distance_dim: Dimension of distance representation (1 for scalar)
            method: Augmentation method ("concat", "project", "attention")
            projection_dim: Optional projection dimension for some methods
            normalize_distance: Whether to normalize distance values
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.distance_dim = distance_dim
        self.method = method
        self.normalize_distance = normalize_distance
        
        if method == "concat":
            self.output_dim = vision_dim + distance_dim
            
        elif method == "project":
            if projection_dim is None:
                projection_dim = vision_dim
            self.output_dim = projection_dim
            
            # Projection layers
            self.vision_proj = nn.Linear(vision_dim, projection_dim)
            self.distance_proj = nn.Linear(distance_dim, projection_dim)
            
        elif method == "attention":
            self.output_dim = vision_dim
            
            # Attention mechanism
            self.distance_attention = nn.Sequential(
                nn.Linear(distance_dim, vision_dim),
                nn.Sigmoid()
            )
            
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
    
    def forward(
        self,
        vision_embeddings: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Augment vision embeddings with distance information.
        
        Args:
            vision_embeddings: Vision embeddings [batch_size, vision_dim]
            distances: Distance values [batch_size, distance_dim]
            
        Returns:
            Augmented embeddings [batch_size, output_dim]
        """
        # Normalize distances if requested
        if self.normalize_distance:
            distances = self._normalize_distances(distances)
        
        if self.method == "concat":
            return torch.cat([vision_embeddings, distances], dim=-1)
        
        elif self.method == "project":
            vision_proj = self.vision_proj(vision_embeddings)
            distance_proj = self.distance_proj(distances)
            return vision_proj + distance_proj
        
        elif self.method == "attention":
            attention_weights = self.distance_attention(distances)
            return vision_embeddings * attention_weights
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _normalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """Normalize distance values."""
        # Log transform for better numerical stability
        distances_norm = torch.log(distances + 1e-8)
        
        # Z-score normalization
        mean = distances_norm.mean()
        std = distances_norm.std()
        return (distances_norm - mean) / (std + 1e-8)
    
    def get_distance_encoding(
        self,
        distances: torch.Tensor,
        encoding_type: str = "linear",
    ) -> torch.Tensor:
        """
        Create positional/distance encodings.
        
        Args:
            distances: Raw distance values
            encoding_type: Type of encoding ("linear", "sinusoidal", "learned")
            
        Returns:
            Distance encodings
        """
        if encoding_type == "linear":
            return distances.unsqueeze(-1) if distances.dim() == 1 else distances
        
        elif encoding_type == "sinusoidal":
            # Sinusoidal position encoding adapted for distance
            max_distance = 100.0  # Reasonable max distance in meters
            normalized_dist = distances / max_distance
            
            # Create sinusoidal encoding
            encoding_dim = self.distance_dim if self.distance_dim > 1 else 64
            encodings = []
            
            for i in range(encoding_dim // 2):
                freq = 1.0 / (10000 ** (2 * i / encoding_dim))
                encodings.append(torch.sin(normalized_dist * freq))
                encodings.append(torch.cos(normalized_dist * freq))
            
            return torch.stack(encodings[:encoding_dim], dim=-1)
        
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def create_distance_tokens(
        self,
        distances: torch.Tensor,
        token_dim: int,
    ) -> torch.Tensor:
        """
        Create distance tokens that can be added to vision tokens.
        
        Args:
            distances: Distance values [batch_size]
            token_dim: Dimension of vision tokens
            
        Returns:
            Distance tokens [batch_size, token_dim]
        """
        # Simple learnable embedding approach
        if not hasattr(self, "distance_embedding"):
            self.distance_embedding = nn.Linear(1, token_dim)
        
        distance_input = distances.unsqueeze(-1)
        return self.distance_embedding(distance_input)
