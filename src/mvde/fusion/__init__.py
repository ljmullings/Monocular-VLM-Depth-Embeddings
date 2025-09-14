"""Fusion modules for combining depth with vision features."""

from .depth_fusion import DepthTokenFusion, DepthAugmentedVisionEncoder

__all__ = [
    "DepthTokenFusion",
    "DepthAugmentedVisionEncoder",
]
