#!/usr/bin/env python3
"""Test script for SigLIP depth estimation implementation."""

import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mvde.model import DepthModel, DepthModelFactory
from mvde.backbones.siglip import SigLIPBackbone
from mvde.heads.dpt_decoder import DPTDecoder, DPTDepthHead
from mvde.export.patch_stats import depth_to_patch_stats, PatchStatsExtractor


def test_siglip_backbone():
    """Test SigLIP backbone functionality."""
    print("Testing SigLIP backbone...")
    
    try:
        # Create backbone (this will download the model on first run)
        backbone = SigLIPBackbone(
            model_name="google/siglip-base-patch16-384",
            image_size=896,
            tap_layers=(-18, -12, -6, -2),
            select_feature="patch",
            dtype=torch.float32,  # Use float32 for testing
        )
        
        # Test forward pass
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 896, 896)
        
        outputs = backbone(dummy_images)
        
        print(f"✓ Backbone forward pass successful")
        print(f"  - Number of feature scales: {len(outputs['grids'])}")
        print(f"  - Feature shapes: {[grid.shape for grid in outputs['grids']]}")
        print(f"  - Spatial dimensions: {outputs['hw']}")
        print(f"  - Last tokens shape: {outputs['last_tokens'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backbone test failed: {e}")
        return False


def test_dpt_decoder():
    """Test DPT decoder functionality."""
    print("\nTesting DPT decoder...")
    
    try:
        # Create decoder
        in_dims = [768, 768, 768, 768]  # Same as SigLIP hidden size
        decoder = DPTDecoder(in_dims=in_dims, c_dec=256, n_blocks=4)
        
        # Create dummy multi-scale features
        batch_size = 2
        features = [
            torch.randn(batch_size, 768, 14, 14),    # Coarsest
            torch.randn(batch_size, 768, 28, 28),
            torch.randn(batch_size, 768, 56, 56),
            torch.randn(batch_size, 768, 56, 56),    # Finest
        ]
        
        depth_prediction = decoder(features)
        
        print(f"✓ DPT decoder forward pass successful")
        print(f"  - Input feature shapes: {[f.shape for f in features]}")
        print(f"  - Output depth shape: {depth_prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ DPT decoder test failed: {e}")
        return False


def test_depth_model():
    """Test complete depth model."""
    print("\nTesting complete DepthModel...")
    
    try:
        # Create model using factory
        model = DepthModelFactory.create_siglip_model(
            model_name="google/siglip-base-patch16-384",
            image_size=384,  # Smaller for testing
            tap_layers=(-6, -4, -2),  # Fewer layers for testing
            decoder_channels=128,  # Smaller for testing
            decoder_blocks=2,
        )
        
        # Test forward pass
        batch_size = 1
        dummy_images = torch.randn(batch_size, 3, 384, 384)
        
        outputs = model(dummy_images, return_features=True)
        
        print(f"✓ Complete model forward pass successful")
        print(f"  - Depth shape: {outputs['depth'].shape}")
        print(f"  - Patch stats shape: {outputs['patch_stats'].shape}")
        print(f"  - Number of feature scales: {len(outputs.get('features', []))}")
        
        # Test model info
        info = model.get_model_info()
        print(f"  - Model info: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Complete model test failed: {e}")
        return False


def test_patch_stats():
    """Test patch statistics extraction."""
    print("\nTesting patch statistics extraction...")
    
    try:
        # Create dummy depth map
        batch_size = 1
        depth = torch.randn(batch_size, 1, 256, 256).abs() + 0.1  # Positive depths
        
        # Test basic patch stats function
        patch_stats = depth_to_patch_stats(depth, patch_size=16, stats=("mean", "var", "grad"))
        
        print(f"✓ Basic patch stats extraction successful")
        print(f"  - Input depth shape: {depth.shape}")
        print(f"  - Patch stats shape: {patch_stats.shape}")
        
        # Test PatchStatsExtractor
        extractor = PatchStatsExtractor(
            patch_size=16,
            stats=("mean", "var", "grad"),
            normalize_stats=False,  # Skip normalization for testing
        )
        
        stats = extractor.extract(depth)
        print(f"✓ PatchStatsExtractor successful")
        print(f"  - Extracted stats shape: {stats.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Patch stats test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading with SigLIP settings."""
    print("\nTesting configuration loading...")
    
    try:
        # Test loading base config
        config_path = Path("configs/base.yaml")
        if config_path.exists():
            config = OmegaConf.load(config_path)
            
            print(f"✓ Configuration loaded successfully")
            print(f"  - VIT model: {config.vit.name}")
            print(f"  - Image size: {config.vit.image_size}")
            print(f"  - Tap layers: {config.vit.tap_layers}")
            print(f"  - Decoder channels: {config.decoder.c_dec}")
            
            # Test creating model from config
            model = DepthModel(config)
            print(f"✓ Model created from config successfully")
            
            return True
        else:
            print(f"✗ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing SigLIP Depth Estimation Implementation")
    print("=" * 60)
    
    tests = [
        test_patch_stats,        # Start with simple test
        test_dpt_decoder,        # Test decoder without model downloads
        test_config_loading,     # Test config system
        test_siglip_backbone,    # This will download the model
        test_depth_model,        # Test complete integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! Implementation is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
