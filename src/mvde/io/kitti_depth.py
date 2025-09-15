"""KITTI Depth Dataset API for loading and evaluating depth predictions."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Import KITTI's original depth reading function with numpy compatibility fix
import sys
devkit_path = Path(__file__).parent.parent.parent.parent / "data/kitti_dataset/devkit/python"
if devkit_path.exists():
    sys.path.append(str(devkit_path))

# Fix numpy compatibility for old KITTI devkit
if not hasattr(np, 'float'):
    np.float = np.float64

from read_depth import depth_read


class KITTIDepthDataset:
    """
    KITTI Depth Completion/Prediction Dataset API.
    
    This class provides:
    - Loading KITTI depth ground truth and sparse Velodyne data
    - Converting between KITTI format and standard depth maps
    - Evaluation metrics for depth prediction
    - Integration with SigLIP depth model
    """
    
    def __init__(self, data_root: str, split: str = "val_selection_cropped"):
        """
        Initialize KITTI depth dataset.
        
        Args:
            data_root: Path to KITTI dataset root
            split: Which split to use ("train", "val", "val_selection_cropped", "test")
        """
        self.data_root = Path(data_root)
        self.split = split
        
        # Verify dataset exists
        if not self.data_root.exists():
            raise ValueError(f"KITTI dataset not found at {data_root}")
        
        # Set up paths based on split
        self._setup_paths()
        
        # Load file lists
        self.samples = self._load_samples()
        
        print(f"KITTI {split} dataset loaded: {len(self.samples)} samples")
    
    def _setup_paths(self):
        """Setup paths for different dataset splits."""
        if self.split == "val_selection_cropped":
            self.image_dir = self.data_root / "val_selection_cropped" / "image"
            self.groundtruth_dir = self.data_root / "val_selection_cropped" / "groundtruth_depth"
            self.velodyne_dir = self.data_root / "val_selection_cropped" / "velodyne_raw"
            
        elif self.split in ["train", "val"]:
            self.split_dir = self.data_root / self.split
            # For train/val, we'll scan all drives
            
        elif self.split.startswith("test"):
            self.image_dir = self.data_root / f"{self.split}_anonymous" / "image"
            if "completion" in self.split:
                self.velodyne_dir = self.data_root / f"{self.split}_anonymous" / "velodyne_raw"
        
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load list of available samples."""
        samples = []
        
        if self.split == "val_selection_cropped":
            # Load cropped validation set
            image_files = sorted(glob.glob(str(self.image_dir / "*.png")))
            
            for img_file in image_files:
                sample_id = Path(img_file).stem
                
                sample = {
                    "id": sample_id,
                    "image": img_file,
                }
                
                # Add ground truth if available
                gt_file = self.groundtruth_dir / f"{sample_id}.png"
                if gt_file.exists():
                    sample["groundtruth"] = str(gt_file)
                
                # Add sparse Velodyne if available
                vel_file = self.velodyne_dir / f"{sample_id}.png"
                if vel_file.exists():
                    sample["velodyne"] = str(vel_file)
                
                samples.append(sample)
        
        elif self.split in ["train", "val"]:
            # Load from drive structure
            drive_dirs = sorted(self.split_dir.glob("*/"))
            
            for drive_dir in drive_dirs:
                # Look for images in proj_depth structure
                image_dirs = drive_dir.glob("proj_depth/*/image_02/")
                for image_dir in image_dirs:
                    image_files = sorted(image_dir.glob("*.png"))
                    
                    for img_file in image_files:
                        sample_id = f"{drive_dir.name}_{img_file.stem}"
                        
                        sample = {
                            "id": sample_id,
                            "image": str(img_file),
                            "drive": drive_dir.name,
                            "frame": img_file.stem,
                        }
                        
                        # Add ground truth
                        gt_file = drive_dir / "proj_depth" / "groundtruth" / "image_02" / img_file.name
                        if gt_file.exists():
                            sample["groundtruth"] = str(gt_file)
                        
                        # Add Velodyne
                        vel_file = drive_dir / "proj_depth" / "velodyne_raw" / "image_02" / img_file.name
                        if vel_file.exists():
                            sample["velodyne"] = str(vel_file)
                        
                        samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, np.ndarray, torch.Tensor]]:
        """Get a single sample."""
        sample = self.samples[idx].copy()
        
        # Load image
        if "image" in sample:
            image = Image.open(sample["image"]).convert("RGB")
            sample["image_pil"] = image
            sample["image_array"] = np.array(image)
        
        # Load ground truth depth
        if "groundtruth" in sample:
            depth_gt = depth_read(sample["groundtruth"])
            sample["depth_gt"] = depth_gt
            sample["valid_mask"] = (depth_gt > 0).astype(np.float32)
        
        # Load sparse Velodyne depth
        if "velodyne" in sample:
            depth_sparse = depth_read(sample["velodyne"])
            sample["depth_sparse"] = depth_sparse
            sample["sparse_mask"] = (depth_sparse > 0).astype(np.float32)
        
        return sample
    
    def get_sample_by_id(self, sample_id: str) -> Dict:
        """Get sample by ID."""
        for i, sample in enumerate(self.samples):
            if sample["id"] == sample_id:
                return self[i]
        raise ValueError(f"Sample {sample_id} not found")
    
    def get_image_tensor(self, idx: int, size: Tuple[int, int] = (896, 896)) -> torch.Tensor:
        """
        Get image as PyTorch tensor ready for SigLIP model.
        
        Args:
            idx: Sample index
            size: Target size (H, W)
            
        Returns:
            Image tensor (1, 3, H, W) in [0, 1] range
        """
        sample = self[idx]
        image = sample["image_pil"]
        
        # Resize and convert to tensor
        if size != image.size[::-1]:  # PIL uses (W, H)
            image = image.resize(size[::-1], Image.BILINEAR)
        
        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def evaluate_depth_predictions(
        self,
        predictions: List[np.ndarray],
        sample_indices: Optional[List[int]] = None,
        max_depth: float = 80.0,
    ) -> Dict[str, float]:
        """
        Evaluate depth predictions using KITTI metrics.
        
        Args:
            predictions: List of predicted depth maps
            sample_indices: Indices of samples (if None, use first len(predictions))
            max_depth: Maximum depth for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if sample_indices is None:
            sample_indices = list(range(len(predictions)))
        
        assert len(predictions) == len(sample_indices)
        
        # Collect errors
        abs_errors = []
        sq_errors = []
        abs_rel_errors = []
        sq_rel_errors = []
        delta_1_25_errors = []
        delta_1_25_2_errors = []
        delta_1_25_3_errors = []
        
        for pred_depth, idx in zip(predictions, sample_indices):
            sample = self[idx]
            
            if "depth_gt" not in sample:
                continue
            
            gt_depth = sample["depth_gt"]
            valid_mask = sample["valid_mask"]
            
            # Apply max depth threshold
            valid_mask = valid_mask.astype(bool) & (gt_depth <= max_depth) & (gt_depth > 0)
            
            if np.sum(valid_mask) == 0:
                continue
            
            # Get valid pixels
            pred_valid = pred_depth[valid_mask]
            gt_valid = gt_depth[valid_mask]
            
            # Absolute errors
            abs_error = np.abs(pred_valid - gt_valid)
            abs_errors.extend(abs_error)
            
            # Squared errors
            sq_error = (pred_valid - gt_valid) ** 2
            sq_errors.extend(sq_error)
            
            # Relative errors
            abs_rel_error = abs_error / gt_valid
            abs_rel_errors.extend(abs_rel_error)
            
            sq_rel_error = sq_error / gt_valid
            sq_rel_errors.extend(sq_rel_error)
            
            # Delta accuracy
            ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
            delta_1_25_errors.extend((ratio < 1.25).astype(np.float32))
            delta_1_25_2_errors.extend((ratio < 1.25**2).astype(np.float32))
            delta_1_25_3_errors.extend((ratio < 1.25**3).astype(np.float32))
        
        # Compute metrics
        metrics = {
            "mae": np.mean(abs_errors),
            "rmse": np.sqrt(np.mean(sq_errors)),
            "abs_rel": np.mean(abs_rel_errors),
            "sq_rel": np.mean(sq_rel_errors),
            "delta_1.25": np.mean(delta_1_25_errors),
            "delta_1.25^2": np.mean(delta_1_25_2_errors),
            "delta_1.25^3": np.mean(delta_1_25_3_errors),
            "num_valid_pixels": len(abs_errors),
        }
        
        return metrics
    
    def visualize_sample(
        self,
        idx: int,
        prediction: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
    ):
        """Visualize a sample with optional prediction."""
        sample = self[idx]
        
        # Determine number of subplots
        num_plots = 1  # Always have image
        if "depth_gt" in sample:
            num_plots += 1
        if "depth_sparse" in sample:
            num_plots += 1
        if prediction is not None:
            num_plots += 1
        
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Image
        axes[plot_idx].imshow(sample["image_array"])
        axes[plot_idx].set_title(f"Image: {sample['id']}")
        axes[plot_idx].axis("off")
        plot_idx += 1
        
        # Ground truth depth
        if "depth_gt" in sample:
            depth_gt = sample["depth_gt"]
            im = axes[plot_idx].imshow(depth_gt, cmap="viridis", vmin=0, vmax=80)
            axes[plot_idx].set_title("Ground Truth Depth")
            axes[plot_idx].axis("off")
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1
        
        # Sparse depth
        if "depth_sparse" in sample:
            depth_sparse = sample["depth_sparse"]
            im = axes[plot_idx].imshow(depth_sparse, cmap="viridis", vmin=0, vmax=80)
            axes[plot_idx].set_title("Sparse Velodyne")
            axes[plot_idx].axis("off")
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1
        
        # Prediction
        if prediction is not None:
            im = axes[plot_idx].imshow(prediction, cmap="viridis", vmin=0, vmax=80)
            axes[plot_idx].set_title("Prediction")
            axes[plot_idx].axis("off")
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def run_siglip_evaluation(
        self,
        depth_model,
        num_samples: Optional[int] = None,
        batch_size: int = 1,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Run evaluation of SigLIP depth model on KITTI dataset.
        
        Args:
            depth_model: trained SigLIP depth model
            num_samples: Number of samples to evaluate (None = all)
            batch_size: Batch size for inference
            device: Device to run on
            
        Returns:
            Evaluation metrics
        """
        depth_model.eval()
        depth_model.to(device)
        
        if num_samples is None:
            num_samples = len(self)
        
        predictions = []
        sample_indices = []
        
        print(f"Evaluating SigLIP depth model on {num_samples} KITTI samples...")
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_indices = list(range(i, end_idx))
            
            # Prepare batch
            batch_images = []
            for idx in batch_indices:
                image_tensor = self.get_image_tensor(idx)
                batch_images.append(image_tensor)
            
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            
            # Predict depth
            with torch.no_grad():
                depth_outputs = depth_model(batch_tensor)
                pred_depths = depth_outputs["depth"]  # (B, 1, H, W)
            
            # Process predictions
            for j, idx in enumerate(batch_indices):
                pred_depth = pred_depths[j, 0].cpu().numpy()  # (H, W)
                
                # Resize prediction to match ground truth
                sample = self[idx]
                if "depth_gt" in sample:
                    gt_shape = sample["depth_gt"].shape
                    if pred_depth.shape != gt_shape:
                        pred_depth = np.array(Image.fromarray(pred_depth).resize(
                            (gt_shape[1], gt_shape[0]), Image.BILINEAR
                        ))
                
                predictions.append(pred_depth)
                sample_indices.append(idx)
            
            if (i + batch_size) % 50 == 0:
                print(f"   Processed {i + batch_size}/{num_samples} samples")
        
        # Evaluate predictions
        metrics = self.evaluate_depth_predictions(predictions, sample_indices)
        
        print(f"Evaluation complete!")
        print(f"   MAE: {metrics['mae']:.3f}m")
        print(f"   RMSE: {metrics['rmse']:.3f}m")
        print(f"   δ<1.25: {metrics['delta_1.25']:.3f}")
        print(f"   Abs Rel: {metrics['abs_rel']:.3f}")
        
        return metrics


def explore_kitti_dataset(data_root: str):
    """Quick exploration of KITTI dataset structure."""
    print("Exploring KITTI dataset...")
    
    data_path = Path(data_root)
    
    # Check what splits are available
    print(f"\nAvailable data in {data_root}:")
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            print(f"  {item.name}")
            
            # Count files in each directory
            if "image" in item.name:
                png_files = list(item.glob("*.png"))
                print(f"     {len(png_files)} images")
    
    # Try to load validation set if available
    val_cropped = data_path / "val_selection_cropped"
    if val_cropped.exists():
        print(f"\nValidation set available:")
        try:
            dataset = KITTIDepthDataset(data_root, "val_selection_cropped")
            print(f"   Loaded {len(dataset)} samples")
            
            # Show sample info
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   Sample structure: {list(sample.keys())}")
                
                if "depth_gt" in sample:
                    gt_shape = sample["depth_gt"].shape
                    valid_pixels = np.sum(sample["valid_mask"])
                    print(f"   Ground truth: {gt_shape}, {valid_pixels} valid pixels")
                
        except Exception as e:
            print(f"   Error loading dataset: {e}")
    
    else:
        print(f"\nval_selection_cropped not found. You may need to download the full KITTI depth dataset.")


if __name__ == "__main__":
    # Example usage
    data_root = "/Users/lauramullings/git/Monocular-VLM-Depth-Embeddings-/data/kitti_dataset"
    explore_kitti_dataset(data_root)
