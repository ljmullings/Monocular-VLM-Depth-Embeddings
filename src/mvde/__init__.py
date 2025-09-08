"""
Monocular VLM Depth Embeddings (MVDE)

A research framework for distance-augmented Vision-Language Models.
"""

__version__ = "0.1.0"
__author__ = "Laura Mullings"
__email__ = "laura@example.com"

from .utils.config import load_config
from .utils.logging import get_logger

__all__ = ["load_config", "get_logger"]
