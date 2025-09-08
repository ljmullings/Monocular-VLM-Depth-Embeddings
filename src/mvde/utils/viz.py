"""Visualization utilities for embeddings and depth overlays."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Optional, Tuple, Union

import cv2
from PIL import Image


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = "umap",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize high-dimensional embeddings in 2D.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ("umap", "tsne", "pca")
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # TODO: Implement embedding visualization
    # - Use UMAP/t-SNE/PCA for dimensionality reduction
    # - Create scatter plot with optional color coding
    # - Add legend and annotations
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Placeholder implementation
    if embeddings.shape[1] > 2:
        # Use first 2 dimensions as placeholder
        emb_2d = embeddings[:, :2]
    else:
        emb_2d = embeddings
    
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.7)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Embedding Visualization ({method.upper()})")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_depth_overlay(
    image: Union[np.ndarray, Image.Image],
    depth_map: np.ndarray,
    alpha: float = 0.6,
    colormap: str = "plasma",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create an overlay of depth information on the original image.
    
    Args:
        image: Original RGB image
        depth_map: Depth map of same spatial size as image
        alpha: Transparency of depth overlay
        colormap: Matplotlib colormap for depth visualization
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # TODO: Implement depth overlay visualization
    # - Normalize depth map
    # - Apply colormap
    # - Blend with original image
    # - Add colorbar
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    # Depth map
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    im2 = ax2.imshow(depth_normalized, cmap=colormap)
    ax2.set_title("Depth Map")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3.imshow(image)
    ax3.imshow(depth_normalized, cmap=colormap, alpha=alpha)
    ax3.set_title("Depth Overlay")
    ax3.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_object_distances(
    distances: List[float],
    object_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot object distances as a bar chart.
    
    Args:
        distances: List of distances for each object
        object_names: Names of the objects
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # TODO: Implement object distance visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(object_names, distances)
    ax.set_ylabel("Distance (meters)")
    ax.set_title("Object Distances")
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, distance in zip(bars, distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{distance:.2f}m',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig
