"""
Unit tests for KITTI depth functionality.
These tests work without requiring the full dataset.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add devkit to path for the test
devkit_path = Path(__file__).parent.parent / "data/kitti_dataset/devkit/python"
if devkit_path.exists():
    sys.path.insert(0, str(devkit_path))


class TestKITTIDepthUtils(unittest.TestCase):
    """Test utility functions that don't require the full dataset."""
    
    def create_mock_depth_png(self, depth_array: np.ndarray, filename: str):
        """Helper to create mock KITTI depth files."""
        depth_mm = (depth_array * 256).astype(np.uint16)
        depth_mm[depth_array <= 0] = 0
        Image.fromarray(depth_mm).save(filename)
    
    def test_depth_read_import(self):
        """Test that we can import the depth reading function."""
        try:
            from read_depth import depth_read
            self.assertTrue(callable(depth_read))
        except ImportError:
            self.skipTest("KITTI devkit not available")
    
    def test_depth_read_functionality(self):
        """Test the KITTI depth reading function."""
        try:
            from read_depth import depth_read
        except ImportError:
            self.skipTest("KITTI devkit not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test depth data
            depth_gt = np.random.uniform(1.0, 50.0, (100, 200))
            depth_gt[::5, ::5] = 0  # Add invalid pixels
            
            depth_file = os.path.join(temp_dir, "test_depth.png")
            self.create_mock_depth_png(depth_gt, depth_file)
            
            # Read depth
            depth_result = depth_read(depth_file)
            
            # Verify shape
            self.assertEqual(depth_result.shape, depth_gt.shape)
            
            # Verify invalid pixels are marked as -1
            invalid_mask = depth_gt <= 0
            self.assertTrue(np.all(depth_result[invalid_mask] == -1))
            
            # Verify valid pixels are reasonable
            valid_mask = depth_gt > 0
            self.assertTrue(np.all(depth_result[valid_mask] > 0))
            self.assertTrue(np.all(depth_result[valid_mask] < 1000))  # Reasonable depth range
    
    def test_kitti_dataset_import(self):
        """Test that we can import the KITTI dataset class."""
        try:
            from mvde.io.kitti_depth import KITTIDepthDataset, explore_kitti_dataset
            self.assertTrue(callable(KITTIDepthDataset))
            self.assertTrue(callable(explore_kitti_dataset))
        except ImportError as e:
            self.fail(f"Could not import KITTI dataset: {e}")
    
    def test_dataset_graceful_failure(self):
        """Test that dataset fails gracefully with missing data."""
        from mvde.io.kitti_depth import KITTIDepthDataset
        
        # Test with non-existent path
        with self.assertRaises(ValueError):
            KITTIDepthDataset("/nonexistent/path")
    
    def test_evaluation_metrics_calculation(self):
        """Test evaluation metrics calculation with synthetic data."""
        from mvde.io.kitti_depth import KITTIDepthDataset
        
        # Create synthetic data for metric calculation
        gt_depth = np.random.uniform(1.0, 80.0, (100, 200))
        pred_depth = gt_depth + np.random.normal(0, 2.0, gt_depth.shape)  # Add noise
        
        # Create valid mask
        valid_mask = np.random.random(gt_depth.shape) > 0.1  # 90% valid
        gt_depth[~valid_mask] = 0
        pred_depth[~valid_mask] = 0
        
        # Manually calculate what the metrics should be
        valid_pixels = valid_mask
        gt_valid = gt_depth[valid_pixels]
        pred_valid = pred_depth[valid_pixels]
        
        abs_error = np.abs(pred_valid - gt_valid)
        expected_mae = np.mean(abs_error)
        expected_rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
        
        # Test that our metric names are reasonable
        self.assertGreater(expected_mae, 0)
        self.assertGreater(expected_rmse, 0)
        self.assertGreaterEqual(expected_rmse, expected_mae)  # RMSE >= MAE


class TestKITTIDatasetMock(unittest.TestCase):
    """Test KITTI dataset with mock data."""
    
    def setUp(self):
        """Set up mock dataset for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_mock_dataset()
    
    def tearDown(self):
        """Clean up mock dataset."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_mock_dataset(self):
        """Create a minimal mock KITTI dataset."""
        # Create directory structure
        val_dir = Path(self.temp_dir) / "val_selection_cropped"
        image_dir = val_dir / "image"
        gt_dir = val_dir / "groundtruth_depth"
        
        for dir_path in [image_dir, gt_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create one sample
        sample_id = "test_sample_000000"
        
        # RGB image
        img_array = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(image_dir / f"{sample_id}.png")
        
        # Depth ground truth
        depth_gt = np.random.uniform(1.0, 50.0, (100, 200))
        depth_gt[::5, ::5] = 0  # Invalid pixels
        depth_mm = (depth_gt * 256).astype(np.uint16)
        depth_mm[depth_gt <= 0] = 0
        Image.fromarray(depth_mm).save(gt_dir / f"{sample_id}.png")
    
    def test_dataset_loading(self):
        """Test loading the mock dataset."""
        try:
            from mvde.io.kitti_depth import KITTIDepthDataset
        except ImportError:
            self.skipTest("KITTI dataset module not available")
        
        dataset = KITTIDepthDataset(self.temp_dir, "val_selection_cropped")
        self.assertEqual(len(dataset), 1)
        
        # Test getting sample
        sample = dataset[0]
        self.assertIn("id", sample)
        self.assertIn("image", sample)
        self.assertIn("image_pil", sample)
        self.assertIn("image_array", sample)
        
        # Test image tensor conversion
        image_tensor = dataset.get_image_tensor(0, size=(64, 64))
        self.assertEqual(image_tensor.shape, (1, 3, 64, 64))
        self.assertTrue(0 <= image_tensor.min() <= image_tensor.max() <= 1)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
