#!/usr/bin/env python3
"""
Example integration of depth fusion with VILA for a 20,000 image experiment.

This shows how to:
1. Load the SigLIP depth model
2. Wrap VILA's vision encoder with depth fusion
3. Run comparison experiments (baseline vs depth-augmented)
4. Collect metrics for evaluation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mvde.model import DepthModelFactory
from mvde.fusion.depth_fusion import DepthAugmentedVisionEncoder


class VILADepthExperiment:
    """
    Complete experimental setup for comparing VILA with/without depth.
    
    This class handles:
    - Loading a trained SigLIP depth model
    - Wrapping VILA with depth fusion
    - Running baseline vs augmented comparisons
    - Collecting evaluation metrics
    """
    
    def __init__(
        self,
        vila_model,  # Loaded VILA model
        depth_model_config: Dict[str, Any],
        fusion_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the experiment setup.
        
        Args:
            vila_model: Loaded VILA model instance
            depth_model_config: Config for creating/loading SigLIP depth model
            fusion_config: Configuration for depth fusion module
        """
        self.vila_model = vila_model
        
        # Create or load depth model
        self.depth_model = self._setup_depth_model(depth_model_config)
        
        # Extract VILA's vision encoder
        self.original_vision_encoder = self._extract_vila_vision_encoder()
        
        # Create depth-augmented version
        self.augmented_vision_encoder = self._create_augmented_encoder(fusion_config)
        
        # Track experiment state
        self.current_mode = "baseline"  # or "augmented"
        self.results = {"baseline": [], "augmented": []}
    
    def _setup_depth_model(self, config: Dict[str, Any]):
        """Setup the SigLIP depth model."""
        if "checkpoint_path" in config:
            # Load from checkpoint
            depth_model = DepthModelFactory.create_siglip_model(**config["model_params"])
            checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
            depth_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Loaded depth model from {config['checkpoint_path']}")
        else:
            # Create new model (for testing)
            depth_model = DepthModelFactory.create_siglip_model(**config)
            print(f"✅ Created new depth model")
        
        depth_model.eval()
        return depth_model
    
    def _extract_vila_vision_encoder(self):
        """Extract VILA's vision encoder component."""
        # This depends on the VILA implementation
        # Common patterns:
        if hasattr(self.vila_model, 'vision_tower'):
            return self.vila_model.vision_tower
        elif hasattr(self.vila_model, 'vision_encoder'):
            return self.vila_model.vision_encoder
        elif hasattr(self.vila_model, 'visual_model'):
            return self.vila_model.visual_model
        else:
            raise ValueError("Could not find vision encoder in VILA model")
    
    def _create_augmented_encoder(self, fusion_config: Dict[str, Any]):
        """Create depth-augmented vision encoder."""
        default_config = {
            "vision_dim": 768,  # Adjust based on the VILA variant
            "depth_stats_dim": 3,  # mean, var, grad
            "dropout": 0.1,
            "normalize_stats": True,
            "gate_init": 0.0,  # Start with no impact
        }
        
        if fusion_config:
            default_config.update(fusion_config)
        
        augmented_encoder = DepthAugmentedVisionEncoder(
            vision_encoder=self.original_vision_encoder,
            depth_model=self.depth_model,
            fusion_config=default_config,
            freeze_depth=True,  # Keep depth model frozen
            freeze_vision=False,  # Allow vision fine-tuning
        )
        
        return augmented_encoder
    
    def set_mode(self, mode: str):
        """Set experiment mode: 'baseline' or 'augmented'."""
        if mode == "baseline":
            # Use original VILA vision encoder (no depth)
            self._replace_vila_vision_encoder(self.original_vision_encoder)
            self.current_mode = "baseline"
            
        elif mode == "augmented":
            # Use depth-augmented vision encoder
            self._replace_vila_vision_encoder(self.augmented_vision_encoder)
            self.augmented_vision_encoder.set_fusion_mode("learning")
            self.current_mode = "augmented"
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        print(f"🔧 Switched to {mode} mode")
    
    def _replace_vila_vision_encoder(self, new_encoder):
        """Replace VILA's vision encoder with the new one."""
        # This depends on the VILA implementation
        if hasattr(self.vila_model, 'vision_tower'):
            self.vila_model.vision_tower = new_encoder
        elif hasattr(self.vila_model, 'vision_encoder'):
            self.vila_model.vision_encoder = new_encoder
        elif hasattr(self.vila_model, 'visual_model'):
            self.vila_model.visual_model = new_encoder
    
    def run_inference(self, images: torch.Tensor, questions: List[str]) -> Dict[str, Any]:
        """
        Run VILA inference in current mode.
        
        Args:
            images: Batch of images (B, 3, H, W)
            questions: List of questions/prompts
            
        Returns:
            Inference results with metadata
        """
        with torch.no_grad():
            # Run VILA inference
            outputs = self.vila_model(images, questions)
            
            # Add metadata
            result = {
                "outputs": outputs,
                "mode": self.current_mode,
                "batch_size": images.shape[0],
            }
            
            # If augmented mode, add fusion info
            if self.current_mode == "augmented":
                fusion_info = self.augmented_vision_encoder.get_fusion_info()
                result["fusion_info"] = fusion_info
            
            return result
    
    def run_comparison_batch(
        self, 
        images: torch.Tensor, 
        questions: List[str],
        ground_truth: Any = None,
    ) -> Dict[str, Any]:
        """
        Run both baseline and augmented inference on the same batch.
        
        Args:
            images: Batch of images
            questions: Questions/prompts
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            Comparison results
        """
        results = {}
        
        # Baseline inference
        self.set_mode("baseline")
        baseline_result = self.run_inference(images, questions)
        results["baseline"] = baseline_result
        
        # Augmented inference
        self.set_mode("augmented")
        augmented_result = self.run_inference(images, questions)
        results["augmented"] = augmented_result
        
        # Compare outputs if ground truth provided
        if ground_truth is not None:
            results["comparison"] = self._compare_results(
                baseline_result["outputs"],
                augmented_result["outputs"],
                ground_truth
            )
        
        return results
    
    def _compare_results(self, baseline_outputs, augmented_outputs, ground_truth):
        """Compare baseline vs augmented results against ground truth."""
        # This depends on specific task and evaluation metrics
        # Example for distance estimation:
        
        def extract_distances(outputs):
            # Extract predicted distances from VILA outputs
            # This depends on output format
            return outputs.get("distances", [])
        
        baseline_distances = extract_distances(baseline_outputs)
        augmented_distances = extract_distances(augmented_outputs)
        
        comparison = {
            "baseline_accuracy": self._compute_accuracy(baseline_distances, ground_truth),
            "augmented_accuracy": self._compute_accuracy(augmented_distances, ground_truth),
            "improvement": None,
        }
        
        if comparison["baseline_accuracy"] is not None and comparison["augmented_accuracy"] is not None:
            comparison["improvement"] = comparison["augmented_accuracy"] - comparison["baseline_accuracy"]
        
        return comparison
    
    def _compute_accuracy(self, predictions, ground_truth):
        """Compute accuracy metric for a specific task."""
        # Implement evaluation metric here
        # For example, for distance estimation:
        if not predictions or not ground_truth:
            return None
        
        # Example: Mean Absolute Error
        errors = [abs(pred - true) for pred, true in zip(predictions, ground_truth)]
        mae = sum(errors) / len(errors)
        return -mae  # Negative MAE so higher is better
    
    def run_large_scale_experiment(
        self,
        image_dataset,  # run on 20,000 image dataset
        questions_dataset,  # Corresponding questions
        ground_truth_dataset,  # Ground truth answers
        batch_size: int = 8,
        save_results: bool = True,
        results_path: str = "depth_experiment_results.pt",
    ):
        """
        Run the complete large-scale experiment on 20,000 images.
        
        Args:
            image_dataset: Dataset of images
            questions_dataset: Dataset of questions
            ground_truth_dataset: Ground truth answers
            batch_size: Batch size for processing
            save_results: Whether to save results
            results_path: Path to save results
        """
        print(f"🚀 Starting large-scale experiment on {len(image_dataset)} images...")
        
        all_results = []
        total_batches = (len(image_dataset) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_dataset))
            
            # Get batch
            batch_images = torch.stack([image_dataset[i] for i in range(start_idx, end_idx)])
            batch_questions = [questions_dataset[i] for i in range(start_idx, end_idx)]
            batch_gt = [ground_truth_dataset[i] for i in range(start_idx, end_idx)]
            
            # Run comparison
            batch_results = self.run_comparison_batch(batch_images, batch_questions, batch_gt)
            batch_results["batch_idx"] = batch_idx
            batch_results["image_indices"] = list(range(start_idx, end_idx))
            
            all_results.append(batch_results)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1}/{total_batches} batches")
        
        # Aggregate results
        aggregated = self._aggregate_experiment_results(all_results)
        
        print(f"✅ Experiment complete!")
        print(f"   Baseline accuracy: {aggregated['baseline_avg_accuracy']:.4f}")
        print(f"   Augmented accuracy: {aggregated['augmented_avg_accuracy']:.4f}")
        print(f"   Average improvement: {aggregated['avg_improvement']:.4f}")
        
        if save_results:
            torch.save({
                "all_results": all_results,
                "aggregated": aggregated,
                "config": {
                    "depth_model": str(self.depth_model),
                    "fusion_config": self.augmented_vision_encoder.get_fusion_info(),
                }
            }, results_path)
            print(f"💾 Results saved to {results_path}")
        
        return aggregated
    
    def _aggregate_experiment_results(self, all_results):
        """Aggregate results from all batches."""
        baseline_accuracies = []
        augmented_accuracies = []
        improvements = []
        
        for batch_result in all_results:
            if "comparison" in batch_result:
                comp = batch_result["comparison"]
                if comp["baseline_accuracy"] is not None:
                    baseline_accuracies.append(comp["baseline_accuracy"])
                if comp["augmented_accuracy"] is not None:
                    augmented_accuracies.append(comp["augmented_accuracy"])
                if comp["improvement"] is not None:
                    improvements.append(comp["improvement"])
        
        return {
            "baseline_avg_accuracy": sum(baseline_accuracies) / len(baseline_accuracies) if baseline_accuracies else 0,
            "augmented_avg_accuracy": sum(augmented_accuracies) / len(augmented_accuracies) if augmented_accuracies else 0,
            "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "num_batches": len(all_results),
            "total_comparisons": len(improvements),
        }


def example_usage():
    """Example of how to use the VILADepthExperiment class."""
    
    # 1. Load the VILA model (placeholder)
    # vila_model = load_vila_model("path/to/vila/checkpoint")
    vila_model = None  # Replace with actual loading
    
    # 2. Configure depth model
    depth_config = {
        # Option A: Load from checkpoint
        "checkpoint_path": "path/to/the/trained/depth/model.pt",
        "model_params": {
            "image_size": 896,
            "tap_layers": (-18, -12, -6, -2),
            "decoder_channels": 256,
        }
        
        # Option B: Create new model for testing
        # "image_size": 896,
        # "tap_layers": (-18, -12, -6, -2),
        # "decoder_channels": 256,
    }
    
    # 3. Configure fusion
    fusion_config = {
        "vision_dim": 768,  # Adjust for the VILA variant
        "dropout": 0.1,
        "gate_init": 0.0,
    }
    
    # 4. Create experiment
    experiment = VILADepthExperiment(
        vila_model=vila_model,
        depth_model_config=depth_config,
        fusion_config=fusion_config,
    )
    
    # 5. Run small test
    test_images = torch.randn(2, 3, 896, 896)
    test_questions = ["How far is the object?", "What is the distance to the car?"]
    test_gt = [5.2, 12.8]  # Ground truth distances in meters
    
    comparison = experiment.run_comparison_batch(test_images, test_questions, test_gt)
    print("Test comparison:", comparison)
    
    # 6. Run full experiment (commented out)
    # results = experiment.run_large_scale_experiment(
    #     image_dataset=20k_images,
    #     questions_dataset=questions,
    #     ground_truth_dataset=ground_truth,
    #     batch_size=8,
    # )


if __name__ == "__main__":
    example_usage()
