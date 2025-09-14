"""SigLIP backbone wrapper for multi-scale depth estimation."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import List, Tuple, Dict, Union
import torch.nn.functional as F


def tokens_to_grid(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Convert patch tokens to spatial grid.
    
    Args:
        tokens: (B, N, C) without CLS token
        h: Grid height
        w: Grid width
        
    Returns:
        Grid tensor (B, C, h, w)
    """
    B, N, C = tokens.shape
    assert N == h * w, f"Token count {N} doesn't match grid size {h}x{w}"
    return tokens.transpose(1, 2).reshape(B, C, h, w)


class SigLIPBackbone(nn.Module):
    """
    Wraps HuggingFace SiglipVisionModel to expose multi-scale (tapped) grids.
    """
    
    def __init__(
        self,
        model_name: str,
        image_size: int = 384,
        tap_layers: Tuple[int, ...] = (-18, -12, -6, -2),
        select_feature: str = "patch",
        interpolate_pos_embeds: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize SigLIP backbone.
        
        Args:
            model_name: HuggingFace model name (e.g., "google/siglip-base-patch16-384")
            image_size: Target image resolution
            tap_layers: Which layers to extract features from (negative indexing)
            select_feature: "patch" (drop CLS) or "cls_patch" (keep CLS)
            interpolate_pos_embeds: Whether to resize positional embeddings
            dtype: Model data type
        """
        super().__init__()
        
        self.model_name = model_name
        self.tap_layers = tap_layers
        self.select_feature = select_feature
        self.dtype = dtype
        self.image_size = image_size
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=False)
        self.model.eval()
        
        # Get model config
        # SigLIP uses vision_config for nested config
        if hasattr(self.model.config, 'vision_config'):
            vision_config = self.model.config.vision_config
        else:
            vision_config = self.model.config
            
        self.patch_size = getattr(vision_config, 'patch_size', 16)
        self.hidden_size = getattr(vision_config, 'hidden_size', 768)
        
        # Optional: resize positional embeddings for higher resolution
        # Note: Disabled for now due to shape mismatch issues with SigLIP
        # if interpolate_pos_embeds and hasattr(self.model, "vision_model"):
        #     old_size = getattr(vision_config, 'image_size', 384)
        #     if image_size != old_size:
        #         self._resize_pos_embeds(image_size)
    
    def forward(
        self,
        images: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[int, int]]]:
        """
        Forward pass through SigLIP vision model.
        
        Args:
            images: Input images (B, 3, H, W) in [0, 1] range
            output_hidden_states: Whether to output intermediate hidden states
            
        Returns:
            Dictionary containing:
            - 'grids': List of feature grids (B, C, h, w) from tapped layers
            - 'last_tokens': Tokens from second-to-last layer (B, N, C)
            - 'hw': Spatial dimensions (h, w) of feature grids
        """
        # Forward through vision model (use vision_model if available)
        if hasattr(self.model, 'vision_model'):
            vision_model = self.model.vision_model
        else:
            vision_model = self.model
            
        # Get device from model parameters
        device = next(vision_model.parameters()).device
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = vision_model(
                pixel_values=images.to(device, dtype=self.dtype),
                output_hidden_states=output_hidden_states,
            )
        
        hidden_states = outputs.hidden_states  # List of (B, 1+N, C) or (B, N, C)
        
        # Infer spatial dimensions
        B = images.shape[0]
        H, W = images.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size
        
        # Extract features from tapped layers
        def select_tokens(layer_tensor: torch.Tensor) -> torch.Tensor:
            """Select patch tokens based on configuration."""
            if self.select_feature == "patch":
                # Check if CLS token is present
                expected_patches = h * w
                if layer_tensor.shape[1] == expected_patches + 1:
                    return layer_tensor[:, 1:, :]  # Drop CLS token
                else:
                    return layer_tensor  # Already patch tokens only
            elif self.select_feature == "cls_patch":
                return layer_tensor  # Keep all tokens
            else:
                raise ValueError(f"Unknown select_feature: {self.select_feature}")
        
        # Extract tapped features and convert to grids
        tapped_tokens = [select_tokens(hidden_states[i]) for i in self.tap_layers]
        
        # Convert to spatial grids
        grid_feats = []
        for tokens in tapped_tokens:
            # Ensure we have patch tokens only for grid conversion
            if tokens.shape[1] > h * w:
                patch_tokens = tokens[:, 1:, :]  # Remove CLS if present
            else:
                patch_tokens = tokens
            
            grid = tokens_to_grid(patch_tokens, h, w)
            grid_feats.append(grid)
        
        # Get tokens from second-to-last layer for compatibility
        last_tokens = select_tokens(hidden_states[-2])
        
        return {
            "grids": grid_feats,
            "last_tokens": last_tokens,
            "hw": (h, w),
        }
    
    def _resize_pos_embeds(self, target_resolution: int) -> None:
        """
        Resize positional embeddings for different input resolution.
        
        Args:
            target_resolution: Target image size
        """
        vision_model = self.model.vision_model
        embeddings = vision_model.embeddings
        
        # Get current embedding parameters
        old_size = embeddings.image_size
        patch_size = embeddings.patch_size
        
        old_grid_size = old_size // patch_size
        new_grid_size = target_resolution // patch_size
        
        if new_grid_size == old_grid_size:
            return  # No change needed
        
        print(f"Resizing positional embeddings from {old_grid_size}x{old_grid_size} "
              f"to {new_grid_size}x{new_grid_size}")
        
        # Get current position embeddings
        pos_embed = embeddings.position_embedding.weight.data  # (old_grid^2, C)
        embed_dim = pos_embed.shape[-1]
        
        # Reshape to 2D grid
        pos_embed_2d = pos_embed.view(1, old_grid_size, old_grid_size, embed_dim)
        pos_embed_2d = pos_embed_2d.permute(0, 3, 1, 2)  # (1, C, H, W)
        
        # Interpolate to new size
        pos_embed_new = F.interpolate(
            pos_embed_2d,
            size=(new_grid_size, new_grid_size),
            mode="bicubic",
            align_corners=False,
        )
        
        # Reshape back to sequence
        pos_embed_new = pos_embed_new.permute(0, 2, 3, 1)  # (1, H, W, C)
        pos_embed_new = pos_embed_new.reshape(new_grid_size * new_grid_size, embed_dim)
        
        # Update embedding layer - check if the sizes match
        current_weight = embeddings.position_embedding.weight.data
        if current_weight.shape[0] == pos_embed_new.shape[0]:
            embeddings.position_embedding.weight.data.copy_(pos_embed_new)
            embeddings.image_size = target_resolution
            embeddings.num_patches = new_grid_size * new_grid_size
            embeddings.num_positions = new_grid_size * new_grid_size
        else:
            print(f"Warning: Position embedding size mismatch. "
                  f"Expected {current_weight.shape[0]}, got {pos_embed_new.shape[0]}. "
                  f"Skipping resize.")
        
        # Update processor crop size
        if hasattr(self.processor, "crop_size"):
            if isinstance(self.processor.crop_size, dict):
                self.processor.crop_size = {
                    "height": target_resolution,
                    "width": target_resolution,
                }
            else:
                self.processor.crop_size = target_resolution
        elif hasattr(self.processor, "size"):
            self.processor.size = {
                "height": target_resolution,
                "width": target_resolution,
            }
    
    def get_feature_info(self) -> Dict[str, Union[int, List[int]]]:
        """Get information about extracted features."""
        return {
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "tap_layers": self.tap_layers,
            "num_features": len(self.tap_layers),
            "feature_dims": [self.hidden_size] * len(self.tap_layers),
        }
    
    @torch.no_grad()
    def extract_features_only(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract only the feature grids (for inference).
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            List of feature grids from tapped layers
        """
        outputs = self.forward(images, output_hidden_states=True)
        return outputs["grids"]
