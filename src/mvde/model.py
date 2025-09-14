"""Unified depth estimation model combining SigLIP backbone with DPT decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig

from .backbones.siglip import SigLIPBackbone
from .heads.dpt_decoder import DPTDecoder, DPTDepthHead


class DepthModel(nn.Module):
    """
    Complete depth estimation model with SigLIP backbone and DPT decoder.
    
    This model can be used for:
    1. Dense depth estimation
    2. Patch-level depth statistics for VLM integration
    3. Multi-scale feature extraction
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize depth model from configuration.
        
        Args:
            config: Configuration containing vit, decoder, and training settings
        """
        super().__init__()
        
        self.config = config
        
        # Initialize SigLIP backbone
        self.backbone = SigLIPBackbone(
            model_name=config.vit.name,
            image_size=config.train.res,
            tap_layers=tuple(config.vit.tap_layers),
            select_feature=config.vit.select_feature,
            interpolate_pos_embeds=True,
            dtype=getattr(torch, config.get("dtype", "float16")),
        )
        
        # Initialize DPT decoder
        feature_dims = [self.backbone.hidden_size] * len(config.vit.tap_layers)
        self.decoder = DPTDepthHead(
            in_dims=feature_dims,
            c_dec=config.decoder.c_dec,
            n_blocks=config.decoder.blocks,
            scale_invariant=config.decoder.get("scale_invariant", True),
        )
        
        # Store useful properties
        self.patch_size = self.backbone.patch_size
        self.hidden_size = self.backbone.hidden_size
        
        # Freezing configuration
        self.freeze_backbone_steps = config.train.get("freeze_backbone_steps", 0)
        self.current_step = 0
        
        # Initially freeze backbone if specified
        if self.freeze_backbone_steps > 0:
            self._freeze_backbone()
    
    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            images: Input images (B, 3, H, W) in [0, 1] range
            return_features: Whether to return multi-scale features
            return_intermediates: Whether to return decoder intermediates
            
        Returns:
            Dictionary containing:
            - 'depth': Dense depth predictions (B, 1, H, W)
            - 'features': Multi-scale features (if requested)
            - 'intermediates': Decoder intermediates (if requested)
            - 'patch_stats': Patch-level statistics
        """
        # Extract multi-scale features
        backbone_outputs = self.backbone(images, output_hidden_states=True)
        features = backbone_outputs["grids"]  # List of (B, C, h, w)
        h, w = backbone_outputs["hw"]
        
        # Predict depth
        if return_intermediates:
            depth_low, intermediates = self.decoder.decoder.forward_with_intermediates(features)
        else:
            depth_low = self.decoder(features)
            intermediates = None
        
        # Upsample depth to original resolution
        target_size = (h * self.patch_size, w * self.patch_size)
        depth = F.interpolate(
            depth_low, size=target_size, mode="bilinear", align_corners=False
        )
        
        # Prepare outputs
        outputs = {"depth": depth}
        
        if return_features:
            outputs["features"] = features
        
        if return_intermediates:
            outputs["intermediates"] = intermediates
        
        # Compute patch-level statistics
        outputs["patch_stats"] = self._compute_patch_stats(depth)
        
        return outputs
    
    def predict_depth(self, images: torch.Tensor) -> torch.Tensor:
        """
        Simple depth prediction interface.
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Depth predictions (B, 1, H, W)
        """
        with torch.no_grad():
            outputs = self.forward(images)
            return outputs["depth"]
    
    def extract_patch_features(
        self, images: torch.Tensor, layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract patch-level features for VLM integration.
        
        Args:
            images: Input images (B, 3, H, W)
            layer_idx: Which tap layer to use (-1 for last)
            
        Returns:
            Patch features (B, N, C) where N = h*w
        """
        with torch.no_grad():
            backbone_outputs = self.backbone(images, output_hidden_states=True)
            
            # Get features from specified layer
            feature_grid = backbone_outputs["grids"][layer_idx]  # (B, C, h, w)
            B, C, h, w = feature_grid.shape
            
            # Reshape to patch tokens
            patch_features = feature_grid.view(B, C, h * w).transpose(1, 2)  # (B, N, C)
            
            return patch_features
    
    def _compute_patch_stats(
        self, depth: torch.Tensor, stats: Tuple[str, ...] = ("mean", "var", "grad")
    ) -> torch.Tensor:
        """
        Compute patch-level depth statistics.
        
        Args:
            depth: Dense depth map (B, 1, H, W)
            stats: Statistics to compute
            
        Returns:
            Patch statistics (B, N, len(stats)) where N = num_patches
        """
        B, _, H, W = depth.shape
        patch_size = self.patch_size
        h, w = H // patch_size, W // patch_size
        
        # Reshape depth to patches
        depth_patches = depth.view(B, 1, h, patch_size, w, patch_size)
        depth_patches = depth_patches.permute(0, 1, 3, 5, 2, 4)  # (B, 1, p, p, h, w)
        depth_patches = depth_patches.reshape(B, 1, patch_size * patch_size, h * w)
        
        stat_list = []
        
        for stat in stats:
            if stat == "mean":
                stat_values = depth_patches.mean(dim=2)  # (B, 1, N)
            elif stat == "var":
                stat_values = depth_patches.var(dim=2)  # (B, 1, N)
            elif stat == "grad":
                # Compute gradient magnitude at original resolution then average
                grad_x = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
                grad_x = F.pad(grad_x, (1, 0, 0, 0))
                
                grad_y = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
                grad_y = F.pad(grad_y, (0, 0, 1, 0))
                
                grad_mag = grad_x + grad_y
                
                # Average over patches
                grad_patches = grad_mag.view(B, 1, h, patch_size, w, patch_size)
                grad_patches = grad_patches.permute(0, 1, 3, 5, 2, 4)
                grad_patches = grad_patches.reshape(B, 1, patch_size * patch_size, h * w)
                stat_values = grad_patches.mean(dim=2)  # (B, 1, N)
            else:
                raise ValueError(f"Unknown statistic: {stat}")
            
            stat_list.append(stat_values)
        
        # Combine statistics: (B, len(stats), N) -> (B, N, len(stats))
        patch_stats = torch.cat(stat_list, dim=1).transpose(1, 2)
        
        return patch_stats
    
    def training_step(self, step: int) -> None:
        """
        Update training state (e.g., unfreezing backbone).
        
        Args:
            step: Current training step
        """
        self.current_step = step
        
        # Unfreeze backbone after specified steps
        if (self.freeze_backbone_steps > 0 and 
            step >= self.freeze_backbone_steps and 
            not self._is_backbone_trainable()):
            self._unfreeze_backbone()
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen for initial training")
    
    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Backbone unfrozen at step {self.current_step}")
    
    def _is_backbone_trainable(self) -> bool:
        """Check if backbone is currently trainable."""
        return any(param.requires_grad for param in self.backbone.parameters())
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.vit.name,
            "image_size": self.config.train.res,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "decoder_channels": self.config.decoder.c_dec,
            "decoder_blocks": self.config.decoder.blocks,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_frozen": not self._is_backbone_trainable(),
        }


class DepthModelFactory:
    """Factory for creating depth models with different configurations."""
    
    @staticmethod
    def create_siglip_model(
        model_name: str = "google/siglip-base-patch16-384",
        image_size: int = 896,
        tap_layers: Tuple[int, ...] = (-18, -12, -6, -2),
        decoder_channels: int = 256,
        decoder_blocks: int = 4,
        **kwargs,
    ) -> DepthModel:
        """
        Create a SigLIP-based depth model with sensible defaults.
        
        Args:
            model_name: SigLIP model name
            image_size: Input image size
            tap_layers: Layers to extract features from
            decoder_channels: Decoder channel dimension
            decoder_blocks: Number of decoder blocks
            **kwargs: Additional configuration options
            
        Returns:
            Configured DepthModel
        """
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            "vit": {
                "name": model_name,
                "tap_layers": tap_layers,
                "select_feature": "patch",
            },
            "decoder": {
                "c_dec": decoder_channels,
                "blocks": decoder_blocks,
                "scale_invariant": True,
            },
            "train": {
                "res": image_size,
                "freeze_backbone_steps": kwargs.get("freeze_backbone_steps", 0),
            },
            "dtype": kwargs.get("dtype", "float16"),
        })
        
        return DepthModel(config)
