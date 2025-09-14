"""DPT-style decoder for depth estimation from multi-scale features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FeatureFusionBlock(nn.Module):
    """Feature fusion block for combining features at different scales."""
    
    def __init__(
        self,
        channels: int,
        activation: str = "gelu",
        use_bn: bool = True,
    ):
        super().__init__()
        
        self.conv1 = ConvBNAct(
            channels * 2, channels, kernel_size=3, padding=1, activation=activation
        )
        self.conv2 = ConvBNAct(
            channels, channels, kernel_size=3, padding=1, activation=activation
        )
        
        # Optional residual connection
        self.residual = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        
    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        """
        Fuse features from high and low resolution.
        
        Args:
            x_high: Higher resolution features (larger spatial size)
            x_low: Lower resolution features (smaller spatial size)
            
        Returns:
            Fused features at high resolution
        """
        # Upsample low-res features to match high-res
        x_low_up = F.interpolate(
            x_low, size=x_high.shape[-2:], mode="bilinear", align_corners=False
        )
        
        # Concatenate features
        x_cat = torch.cat([x_high, x_low_up], dim=1)
        
        # Fusion with residual connection
        residual = self.residual(x_cat)
        x_fused = self.conv1(x_cat)
        x_fused = self.conv2(x_fused)
        
        return x_fused + residual


class DPTDecoder(nn.Module):
    """
    DPT-style decoder for dense depth prediction from multi-scale features.
    
    Takes multi-scale features from vision transformer and produces dense depth map.
    """
    
    def __init__(
        self,
        in_dims: List[int],
        c_dec: int = 256,
        n_blocks: int = 4,
        activation: str = "gelu",
        output_channels: int = 1,
        final_activation: Optional[str] = None,
    ):
        """
        Initialize DPT decoder.
        
        Args:
            in_dims: Input dimensions for each scale (from coarse to fine)
            c_dec: Decoder channel dimension
            n_blocks: Number of refinement blocks
            activation: Activation function to use
            output_channels: Number of output channels (1 for depth)
            final_activation: Optional final activation ("sigmoid", "relu", etc.)
        """
        super().__init__()
        
        self.in_dims = in_dims
        self.c_dec = c_dec
        self.n_blocks = n_blocks
        
        # Project each input scale to decoder dimension
        self.projections = nn.ModuleList([
            nn.Conv2d(in_dim, c_dec, kernel_size=1, bias=False)
            for in_dim in in_dims
        ])
        
        # Feature fusion blocks (one less than number of scales)
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(c_dec, activation=activation)
            for _ in range(len(in_dims) - 1)
        ])
        
        # Additional refinement blocks
        self.refinement_blocks = nn.ModuleList([
            ConvBNAct(c_dec, c_dec, kernel_size=3, padding=1, activation=activation)
            for _ in range(n_blocks)
        ])
        
        # Final output head
        self.output_head = nn.Sequential(
            nn.Conv2d(c_dec, c_dec // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_dec // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Conv2d(c_dec // 2, output_channels, kernel_size=1),
        )
        
        # Optional final activation
        if final_activation == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_activation == "relu":
            self.final_act = nn.ReLU()
        elif final_activation == "softplus":
            self.final_act = nn.Softplus()
        else:
            self.final_act = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            features: List of feature maps from coarse to fine resolution
                     Each tensor has shape (B, C_i, H_i, W_i)
        
        Returns:
            Dense depth map (B, output_channels, H, W)
        """
        assert len(features) == len(self.in_dims), \
            f"Expected {len(self.in_dims)} features, got {len(features)}"
        
        # Project all features to decoder dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Start with the coarsest feature (first in list)
        x = projected[0]
        
        # Progressively fuse with finer features
        for i in range(1, len(projected)):
            x = self.fusion_blocks[i-1](projected[i], x)
        
        # Apply refinement blocks
        for block in self.refinement_blocks:
            x = block(x)
        
        # Generate final output
        depth = self.output_head(x)
        
        # Apply final activation if specified
        if self.final_act is not None:
            depth = self.final_act(depth)
        
        return depth
    
    def forward_with_intermediates(
        self, features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning intermediate features for visualization.
        
        Args:
            features: Input feature list
            
        Returns:
            Tuple of (final_depth, intermediate_features)
        """
        intermediates = []
        
        # Project features
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        intermediates.extend(projected)
        
        # Fusion process
        x = projected[0]
        for i in range(1, len(projected)):
            x = self.fusion_blocks[i-1](projected[i], x)
            intermediates.append(x.clone())
        
        # Refinement
        for block in self.refinement_blocks:
            x = block(x)
            intermediates.append(x.clone())
        
        # Final output
        depth = self.output_head(x)
        if self.final_act is not None:
            depth = self.final_act(depth)
        
        return depth, intermediates


class DPTDepthHead(nn.Module):
    """
    Complete depth estimation head with DPT decoder.
    
    Combines the decoder with optional post-processing and loss computation.
    """
    
    def __init__(
        self,
        in_dims: List[int],
        c_dec: int = 256,
        n_blocks: int = 4,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        scale_invariant: bool = True,
    ):
        """
        Initialize depth head.
        
        Args:
            in_dims: Input feature dimensions
            c_dec: Decoder channels
            n_blocks: Number of refinement blocks
            min_depth: Minimum depth value
            max_depth: Maximum depth value
            scale_invariant: Whether to use scale-invariant depth
        """
        super().__init__()
        
        self.decoder = DPTDecoder(
            in_dims=in_dims,
            c_dec=c_dec,
            n_blocks=n_blocks,
            output_channels=1,
            final_activation="sigmoid" if not scale_invariant else None,
        )
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale_invariant = scale_invariant
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Predict depth from features.
        
        Args:
            features: Multi-scale features
            
        Returns:
            Depth predictions (B, 1, H, W)
        """
        depth_raw = self.decoder(features)
        
        if self.scale_invariant:
            # For scale-invariant depth, return raw logits
            return depth_raw
        else:
            # For metric depth, scale to [min_depth, max_depth]
            depth_scaled = self.min_depth + depth_raw * (self.max_depth - self.min_depth)
            return depth_scaled
    
    def compute_loss(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        loss_type: str = "scale_invariant",
    ) -> torch.Tensor:
        """
        Compute depth estimation loss.
        
        Args:
            pred_depth: Predicted depth (B, 1, H, W)
            target_depth: Ground truth depth (B, 1, H, W)
            mask: Optional valid pixel mask (B, 1, H, W)
            loss_type: Type of loss ("mse", "mae", "scale_invariant", "log")
            
        Returns:
            Loss value
        """
        if mask is not None:
            pred_depth = pred_depth[mask]
            target_depth = target_depth[mask]
        else:
            pred_depth = pred_depth.flatten()
            target_depth = target_depth.flatten()
        
        if loss_type == "mse":
            return F.mse_loss(pred_depth, target_depth)
        elif loss_type == "mae":
            return F.l1_loss(pred_depth, target_depth)
        elif loss_type == "log":
            return F.l1_loss(torch.log(pred_depth + 1e-8), torch.log(target_depth + 1e-8))
        elif loss_type == "scale_invariant":
            # Scale-invariant loss (commonly used in relative depth estimation)
            log_pred = torch.log(pred_depth + 1e-8)
            log_target = torch.log(target_depth + 1e-8)
            log_diff = log_pred - log_target
            
            scale_invariant_error = torch.mean(log_diff ** 2) - 0.5 * (torch.mean(log_diff) ** 2)
            return scale_invariant_error
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
