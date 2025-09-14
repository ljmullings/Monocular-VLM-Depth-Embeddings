"""Depth-vision token fusion using residual connections (Option A)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import math


class DepthTokenFusion(nn.Module):
    """
    Fuse depth statistics into vision tokens using residual connection with learnable gate.
    
    This implements Option A: Residual "patch-stats → tokens" approach.
    Starts with zero impact and gradually learns to incorporate depth information.
    """
    
    def __init__(
        self,
        vision_dim: int,
        depth_stats_dim: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        normalize_stats: bool = True,
        gate_init: float = 0.0,
    ):
        """
        Initialize depth token fusion module.
        
        Args:
            vision_dim: Dimension of vision tokens (e.g., 768 for VILA)
            depth_stats_dim: Number of depth statistics per patch (default: 3 for mean,var,grad)
            hidden_dim: Hidden dimension for MLP (defaults to vision_dim)
            dropout: Dropout probability for regularization
            normalize_stats: Whether to normalize depth statistics
            gate_init: Initial value for the learnable gate (0.0 = no initial impact)
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.depth_stats_dim = depth_stats_dim
        self.normalize_stats = normalize_stats
        
        if hidden_dim is None:
            hidden_dim = vision_dim
        
        # Depth statistics to vision token MLP
        self.depth_mlp = nn.Sequential(
            nn.LayerNorm(depth_stats_dim),
            nn.Linear(depth_stats_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vision_dim),
        )
        
        # Learnable gate - controls how much depth to inject
        # Initialize near 0 so we start with baseline performance
        self.gate = nn.Parameter(torch.tensor(gate_init))
        
        # Optional: additional normalization after fusion
        self.output_norm = nn.LayerNorm(vision_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize module weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for stable fusion
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        vision_tokens: torch.Tensor, 
        depth_stats: torch.Tensor,
        return_gate_value: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Fuse depth statistics into vision tokens.
        
        Args:
            vision_tokens: Vision tokens (B, N, vision_dim) - no CLS token
            depth_stats: Depth statistics (B, N, depth_stats_dim)
            return_gate_value: Whether to return the current gate value for monitoring
            
        Returns:
            Fused tokens (B, N, vision_dim) or tuple with gate value
        """
        B, N, C = vision_tokens.shape
        assert C == self.vision_dim, f"Expected vision_dim {self.vision_dim}, got {C}"
        assert depth_stats.shape[:2] == (B, N), f"Depth stats shape mismatch: {depth_stats.shape} vs {(B, N)}"
        
        # Normalize depth statistics for stability
        if self.normalize_stats:
            depth_stats_norm = self._normalize_depth_stats(depth_stats)
        else:
            depth_stats_norm = depth_stats
        
        # Project depth statistics to vision dimension
        depth_features = self.depth_mlp(depth_stats_norm)  # (B, N, vision_dim)
        
        # Apply learnable gate with tanh to keep it bounded
        gate_value = torch.tanh(self.gate)
        depth_contribution = gate_value * depth_features
        
        # Residual fusion
        fused_tokens = vision_tokens + depth_contribution
        
        # Optional output normalization
        fused_tokens = self.output_norm(fused_tokens)
        
        if return_gate_value:
            return fused_tokens, gate_value.item()
        else:
            return fused_tokens
    
    def _normalize_depth_stats(self, depth_stats: torch.Tensor) -> torch.Tensor:
        """
        Normalize depth statistics for training stability.
        
        Uses z-score normalization per image, per statistic type.
        This ensures that different images have comparable depth stat magnitudes.
        """
        B, N, K = depth_stats.shape
        normalized = torch.zeros_like(depth_stats)
        
        for k in range(K):
            stat_values = depth_stats[:, :, k]  # (B, N)
            
            # Compute mean and std per image
            mean_val = stat_values.mean(dim=1, keepdim=True)  # (B, 1)
            std_val = stat_values.std(dim=1, keepdim=True)    # (B, 1)
            
            # Avoid division by zero
            std_val = torch.clamp(std_val, min=1e-8)
            
            # Z-score normalize
            normalized[:, :, k] = (stat_values - mean_val) / std_val
        
        return normalized
    
    def disable_depth(self) -> None:
        """Disable depth fusion for baseline comparison."""
        with torch.no_grad():
            self.gate.data.fill_(0.0)
    
    def enable_depth(self, gate_value: Optional[float] = None) -> None:
        """Enable depth fusion with optional gate value."""
        if gate_value is not None:
            with torch.no_grad():
                self.gate.data.fill_(gate_value)
        # Otherwise, let the gate learn naturally during training
    
    def get_gate_value(self) -> float:
        """Get current gate value for monitoring."""
        return torch.tanh(self.gate).item()
    
    def get_fusion_strength(self) -> str:
        """Get human-readable fusion strength."""
        gate_val = self.get_gate_value()
        if abs(gate_val) < 0.01:
            return "disabled"
        elif abs(gate_val) < 0.3:
            return "weak"
        elif abs(gate_val) < 0.7:
            return "moderate"
        else:
            return "strong"


class DepthAugmentedVisionEncoder(nn.Module):
    """
    Wrapper that combines any vision encoder with depth fusion.
    
    This is a complete solution that takes images, extracts vision tokens,
    computes depth, and fuses them together.
    """
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        depth_model: nn.Module,
        fusion_config: Optional[Dict] = None,
        freeze_depth: bool = True,
        freeze_vision: bool = False,
    ):
        """
        Initialize depth-augmented vision encoder.
        
        Args:
            vision_encoder: Any vision encoder that outputs tokens (B, N, C)
            depth_model: DepthModel instance
            fusion_config: Configuration for DepthTokenFusion
            freeze_depth: Whether to freeze depth model weights
            freeze_vision: Whether to freeze vision encoder weights
        """
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.depth_model = depth_model
        
        # Freeze models if requested
        if freeze_depth:
            for param in self.depth_model.parameters():
                param.requires_grad = False
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # Initialize fusion module
        fusion_config = fusion_config or {}
        
        # Try to infer vision dimension from encoder
        vision_dim = fusion_config.pop('vision_dim', None)  # Remove from config to avoid duplicate
        if vision_dim is None:
            vision_dim = self._infer_vision_dim()
        
        self.fusion = DepthTokenFusion(
            vision_dim=vision_dim,
            **fusion_config
        )
        
        self.freeze_depth = freeze_depth
        self.freeze_vision = freeze_vision
    
    def _infer_vision_dim(self) -> int:
        """Try to infer vision dimension from the encoder."""
        # Common vision dimensions
        if hasattr(self.vision_encoder, 'hidden_size'):
            return self.vision_encoder.hidden_size
        elif hasattr(self.vision_encoder, 'embed_dim'):
            return self.vision_encoder.embed_dim
        elif hasattr(self.vision_encoder, 'config') and hasattr(self.vision_encoder.config, 'hidden_size'):
            return self.vision_encoder.config.hidden_size
        else:
            # Default to common ViT size
            return 768
    
    def forward(
        self, 
        images: torch.Tensor,
        return_depth: bool = False,
        return_gate_value: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with depth-augmented vision encoding.
        
        Args:
            images: Input images (B, 3, H, W)
            return_depth: Whether to return depth maps
            return_gate_value: Whether to return fusion gate value
            
        Returns:
            Dictionary containing:
            - 'tokens': Fused vision tokens (B, N, C)
            - 'depth': Depth maps (if return_depth=True)
            - 'gate_value': Gate value (if return_gate_value=True)
        """
        # Extract vision tokens
        vision_tokens = self.vision_encoder(images)  # (B, N, C)
        
        # Extract depth and patch statistics
        with torch.no_grad() if self.freeze_depth else torch.enable_grad():
            depth_outputs = self.depth_model(images)
            depth_stats = depth_outputs["patch_stats"]  # (B, N, K)
        
        # Fuse depth into vision tokens
        if return_gate_value:
            fused_tokens, gate_value = self.fusion(
                vision_tokens, depth_stats, return_gate_value=True
            )
        else:
            fused_tokens = self.fusion(vision_tokens, depth_stats)
            gate_value = None
        
        # Prepare outputs
        outputs = {"tokens": fused_tokens}
        
        if return_depth:
            outputs["depth"] = depth_outputs["depth"]
        
        if return_gate_value:
            outputs["gate_value"] = gate_value
        
        return outputs
    
    def set_fusion_mode(self, mode: str) -> None:
        """Set fusion mode: 'disabled', 'enabled', or 'learning'."""
        if mode == "disabled":
            self.fusion.disable_depth()
        elif mode == "enabled":
            self.fusion.enable_depth()
        elif mode == "learning":
            # Reset gate to small positive value to encourage learning
            self.fusion.enable_depth(0.1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_fusion_info(self) -> Dict[str, Union[str, float]]:
        """Get information about current fusion state."""
        return {
            "gate_value": self.fusion.get_gate_value(),
            "fusion_strength": self.fusion.get_fusion_strength(),
            "depth_frozen": self.freeze_depth,
            "vision_frozen": self.freeze_vision,
        }


# Utility functions for easy integration
def create_depth_fusion_for_vila(
    vila_vision_encoder,
    depth_model,
    depth_stats_dim: int = 3,
    freeze_depth: bool = True,
) -> DepthAugmentedVisionEncoder:
    """
    Convenience function to create depth fusion for VILA.
    
    Args:
        vila_vision_encoder: VILA's vision encoder
        depth_model: SigLIP depth model
        depth_stats_dim: Number of depth statistics (3 for mean,var,grad)
        freeze_depth: Whether to freeze depth model
        
    Returns:
        Ready-to-use depth-augmented vision encoder
    """
    fusion_config = {
        "vision_dim": 768,  # Standard VILA vision dimension
        "depth_stats_dim": depth_stats_dim,
        "dropout": 0.1,
        "normalize_stats": True,
        "gate_init": 0.0,  # Start with no impact
    }
    
    return DepthAugmentedVisionEncoder(
        vision_encoder=vila_vision_encoder,
        depth_model=depth_model,
        fusion_config=fusion_config,
        freeze_depth=freeze_depth,
        freeze_vision=False,  # Usually want to fine-tune vision for VLM tasks
    )
