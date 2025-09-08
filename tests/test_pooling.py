"""Tests for ROI/patch pooling correctness."""

import pytest
import torch
import numpy as np

from mvde.embed.pooling import ROIPooler, PatchPooler


class TestROIPooler:
    """Test ROI pooling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pooler = ROIPooler(pooling_method="mean")
        
        # Create dummy patch embeddings (16x16 grid, 768 dim)
        self.patch_embeddings = torch.randn(256, 768)
        self.patch_grid_size = (16, 16)
        self.image_size = (640, 480)
    
    def test_roi_pooling_basic(self):
        """Test basic ROI pooling functionality."""
        # Define a bounding box
        bbox = (100, 100, 200, 200)  # x1, y1, x2, y2
        
        # Pool the ROI
        pooled = self.pooler.pool_roi(
            self.patch_embeddings,
            bbox,
            self.patch_grid_size,
            self.image_size
        )
        
        # Check output shape
        assert pooled.shape == (768,), f"Expected shape (768,), got {pooled.shape}"
        assert torch.isfinite(pooled).all(), "Pooled embedding contains non-finite values"
    
    def test_roi_pooling_edge_cases(self):
        """Test ROI pooling edge cases."""
        # Test bbox at image boundary
        bbox = (0, 0, 50, 50)
        pooled = self.pooler.pool_roi(
            self.patch_embeddings,
            bbox,
            self.patch_grid_size,
            self.image_size
        )
        assert pooled.shape == (768,)
        
        # Test bbox outside image (should be clamped)
        bbox = (700, 500, 800, 600)
        pooled = self.pooler.pool_roi(
            self.patch_embeddings,
            bbox,
            self.patch_grid_size,
            self.image_size
        )
        assert pooled.shape == (768,)
    
    def test_pooling_methods(self):
        """Test different pooling methods."""
        bbox = (100, 100, 200, 200)
        
        # Test mean pooling
        pooler_mean = ROIPooler("mean")
        result_mean = pooler_mean.pool_roi(
            self.patch_embeddings, bbox, self.patch_grid_size, self.image_size
        )
        
        # Test max pooling
        pooler_max = ROIPooler("max")
        result_max = pooler_max.pool_roi(
            self.patch_embeddings, bbox, self.patch_grid_size, self.image_size
        )
        
        # Results should be different
        assert not torch.allclose(result_mean, result_max), "Mean and max pooling should differ"
    
    def test_coordinate_conversion(self):
        """Test coordinate conversion from image to patch space."""
        # Full image bbox should use all patches
        full_bbox = (0, 0, self.image_size[1], self.image_size[0])  # (0, 0, width, height)
        
        pooled_full = self.pooler.pool_roi(
            self.patch_embeddings,
            full_bbox,
            self.patch_grid_size,
            self.image_size
        )
        
        # Should be close to global mean
        global_mean = self.patch_embeddings.mean(dim=0)
        assert torch.allclose(pooled_full, global_mean, atol=1e-5)


class TestPatchPooler:
    """Test patch pooling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.patch_embeddings = torch.randn(64, 768)  # 64 patches, 768 dim
    
    def test_mean_pooling(self):
        """Test mean pooling."""
        pooler = PatchPooler("mean")
        result = pooler.pool_patches(self.patch_embeddings)
        
        expected = self.patch_embeddings.mean(dim=0)
        assert torch.allclose(result, expected)
    
    def test_max_pooling(self):
        """Test max pooling."""
        pooler = PatchPooler("max")
        result = pooler.pool_patches(self.patch_embeddings)
        
        expected = self.patch_embeddings.max(dim=0)[0]
        assert torch.allclose(result, expected)
    
    def test_attention_pooling(self):
        """Test attention-based pooling."""
        pooler = PatchPooler("attention")
        result = pooler.pool_patches(self.patch_embeddings)
        
        assert result.shape == (768,)
        assert torch.isfinite(result).all()
    
    def test_cls_pooling(self):
        """Test CLS token pooling."""
        pooler = PatchPooler("cls")
        result = pooler.pool_patches(self.patch_embeddings)
        
        # Should return first token
        expected = self.patch_embeddings[0]
        assert torch.allclose(result, expected)
    
    def test_multi_scale_pooling(self):
        """Test multi-scale pooling."""
        pooler = PatchPooler("mean")
        
        # Create embeddings at different scales
        embeddings_scale1 = torch.randn(16, 768)
        embeddings_scale2 = torch.randn(64, 768)
        
        result = pooler.pool_multi_scale([embeddings_scale1, embeddings_scale2])
        
        assert result.shape == (768,)
        assert torch.isfinite(result).all()


class TestPoolingInvariances:
    """Test pooling invariances and properties."""
    
    def test_translation_invariance(self):
        """Test that pooling is translation invariant for relative coordinates."""
        pooler = ROIPooler("mean")
        embeddings = torch.randn(256, 768)
        grid_size = (16, 16)
        
        # Same relative bbox in different image sizes
        bbox1 = (50, 50, 150, 150)
        image_size1 = (200, 200)
        
        bbox2 = (100, 100, 300, 300)
        image_size2 = (400, 400)
        
        result1 = pooler.pool_roi(embeddings, bbox1, grid_size, image_size1)
        result2 = pooler.pool_roi(embeddings, bbox2, grid_size, image_size2)
        
        # Should be approximately equal (same relative region)
        assert torch.allclose(result1, result2, atol=1e-4)
    
    def test_deterministic_output(self):
        """Test that pooling gives deterministic output."""
        pooler = ROIPooler("mean")
        embeddings = torch.randn(256, 768)
        bbox = (100, 100, 200, 200)
        grid_size = (16, 16)
        image_size = (640, 480)
        
        result1 = pooler.pool_roi(embeddings, bbox, grid_size, image_size)
        result2 = pooler.pool_roi(embeddings, bbox, grid_size, image_size)
        
        assert torch.allclose(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__])
