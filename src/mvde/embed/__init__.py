"""Embedding processing modules."""

from .pooling import ROIPooler, PatchPooler
from .augment import EmbeddingAugmenter

__all__ = [
    "ROIPooler",
    "PatchPooler", 
    "EmbeddingAugmenter",
]
