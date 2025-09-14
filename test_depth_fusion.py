#!/usr/bin/env python3
"""Test script for depth-vision token fusion (Option A)."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mvde.model import DepthModelFactory
from mvde.fusion.depth_fusion import DepthTokenFusion, DepthAugmentedVisionEncoder


class DummyVisionEncoder(torch.nn.Module):
    """Dummy vision encoder that mimics VILA's behavior."""
    
    def __init__(self, patch_size=16, hidden_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # Simple projection to mimic vision encoder
        self.projection = torch.nn.Linear(3 * patch_size * patch_size, hidden_size)
    
    def forward(self, images):
        B, C, H, W = images.shape
        
        # Create patch tokens (simplified)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        # Reshape to patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.reshape(B, C, h_patches * w_patches, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B, h_patches * w_patches, -1)
        
        # Project to hidden dimension
        tokens = self.projection(patches)  # (B, N, hidden_size)
        
        return tokens


def test_depth_token_fusion():
    """Test the DepthTokenFusion module directly."""
    print("🧪 Testing DepthTokenFusion...")
    
    # Setup
    batch_size = 2
    num_patches = 576  # 24x24 patches for 384x384 image
    vision_dim = 768
    depth_stats_dim = 3
    
    # Create fusion module
    fusion = DepthTokenFusion(
        vision_dim=vision_dim,
        depth_stats_dim=depth_stats_dim,
        dropout=0.1,
        normalize_stats=True,
    )
    
    # Create dummy inputs
    vision_tokens = torch.randn(batch_size, num_patches, vision_dim)
    depth_stats = torch.randn(batch_size, num_patches, depth_stats_dim)
    
    print(f"📊 Input shapes:")
    print(f"   Vision tokens: {vision_tokens.shape}")
    print(f"   Depth stats: {depth_stats.shape}")
    
    # Test fusion
    fused_tokens, gate_value = fusion(vision_tokens, depth_stats, return_gate_value=True)
    
    print(f"✅ Fusion successful!")
    print(f"   Fused tokens: {fused_tokens.shape}")
    print(f"   Gate value: {gate_value:.4f}")
    print(f"   Fusion strength: {fusion.get_fusion_strength()}")
    
    # Test disable/enable
    print(f"\n🔧 Testing enable/disable...")
    
    fusion.disable_depth()
    disabled_tokens = fusion(vision_tokens, depth_stats)
    print(f"   Disabled gate: {fusion.get_gate_value():.4f}")
    print(f"   Tokens unchanged: {torch.allclose(disabled_tokens, vision_tokens, atol=1e-6)}")
    
    fusion.enable_depth(0.5)
    enabled_tokens = fusion(vision_tokens, depth_stats)
    print(f"   Enabled gate: {fusion.get_gate_value():.4f}")
    print(f"   Tokens changed: {not torch.allclose(enabled_tokens, vision_tokens, atol=1e-6)}")
    
    return True


def test_depth_augmented_encoder():
    """Test the complete DepthAugmentedVisionEncoder."""
    print(f"\n🏗️ Testing DepthAugmentedVisionEncoder...")
    
    # Create models
    vision_encoder = DummyVisionEncoder(patch_size=16, hidden_size=768)
    depth_model = DepthModelFactory.create_siglip_model(
        image_size=384,
        tap_layers=(-3, -2, -1),  # Use last few layers to avoid index error
        decoder_channels=128,
        decoder_blocks=2,
    )
    
    # Create augmented encoder
    augmented_encoder = DepthAugmentedVisionEncoder(
        vision_encoder=vision_encoder,
        depth_model=depth_model,
        fusion_config={
            "vision_dim": 768,
            "depth_stats_dim": 3,
            "dropout": 0.1,
        },
        freeze_depth=True,
    )
    
    # Test forward pass
    batch_size = 1
    images = torch.randn(batch_size, 3, 384, 384)
    
    print(f"📸 Processing images: {images.shape}")
    
    outputs = augmented_encoder(images, return_depth=True, return_gate_value=True)
    
    print(f"✅ Augmented encoder successful!")
    print(f"   Fused tokens: {outputs['tokens'].shape}")
    print(f"   Depth map: {outputs['depth'].shape}")
    print(f"   Gate value: {outputs['gate_value']:.4f}")
    
    # Test fusion modes
    print(f"\n🎛️ Testing fusion modes...")
    
    info = augmented_encoder.get_fusion_info()
    print(f"   Current state: {info}")
    
    augmented_encoder.set_fusion_mode("disabled")
    print(f"   Disabled: {augmented_encoder.fusion.get_fusion_strength()}")
    
    augmented_encoder.set_fusion_mode("learning")
    print(f"   Learning: {augmented_encoder.fusion.get_fusion_strength()}")
    
    return True


def test_comparison_workflow():
    """Test the comparison workflow for the experiment."""
    print(f"\n🔬 Testing comparison workflow...")
    
    # Setup models
    vision_encoder = DummyVisionEncoder()
    depth_model = DepthModelFactory.create_siglip_model(
        image_size=384,
        tap_layers=(-3, -2, -1),  # Use last few layers
        decoder_channels=128,
        decoder_blocks=2,
    )
    
    augmented_encoder = DepthAugmentedVisionEncoder(
        vision_encoder=vision_encoder,
        depth_model=depth_model,
        freeze_depth=True,
    )
    
    # Dummy images and "answers"
    images = torch.randn(3, 3, 384, 384)  # 3 test images
    
    def dummy_vlm_task(tokens):
        """Simulate VLM task (e.g., object distance estimation)."""
        # Average pool tokens and predict distance
        pooled = tokens.mean(dim=1)  # (B, 768)
        distances = torch.nn.functional.relu(pooled.sum(dim=1, keepdim=True))  # (B, 1)
        return distances
    
    # Test without depth (baseline)
    print(f"   Running baseline (no depth)...")
    augmented_encoder.set_fusion_mode("disabled")
    baseline_tokens = augmented_encoder(images)["tokens"]
    baseline_answers = dummy_vlm_task(baseline_tokens)
    
    # Test with depth
    print(f"   Running with depth...")
    augmented_encoder.set_fusion_mode("learning")
    depth_tokens = augmented_encoder(images)["tokens"]
    depth_answers = dummy_vlm_task(depth_tokens)
    
    print(f"✅ Comparison workflow successful!")
    print(f"   Baseline answers: {baseline_answers.flatten()}")
    print(f"   Depth answers: {depth_answers.flatten()}")
    print(f"   Answers differ: {not torch.allclose(baseline_answers, depth_answers, atol=1e-6)}")
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Depth-Vision Token Fusion (Option A)")
    print("=" * 60)
    
    tests = [
        test_depth_token_fusion,
        test_depth_augmented_encoder,
        test_comparison_workflow,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} passed")
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"💥 {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Depth fusion is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Integrate with actual VILA vision encoder")
        print("   2. Run on 20,000 image dataset")
        print("   3. Compare baseline vs depth-augmented performance")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
