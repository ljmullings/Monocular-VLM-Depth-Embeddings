"""End-to-end inference pipeline."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
from PIL import Image

from ..vila.client import VILAClient
from ..depth.midas import MiDaSEstimator
from ..depth.zoe import ZoeDepthEstimator
from ..detectors.yolo import YOLODetector
from ..detectors.referent import ReferentSelector
from ..embed.pooling import ROIPooler
from ..embed.augment import EmbeddingAugmenter
from ..heads.mlp import MLPHead
from ..heads.lora import LoRAHead
from ..utils.logging import get_logger


logger = get_logger(__name__)


class InferencePipeline:
    """End-to-end pipeline for distance-augmented VLM inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inference pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self._init_vila_model()
        self._init_depth_estimator()
        self._init_detectors()
        self._init_embedding_components()
        self._init_distance_head()
        
        logger.info("Inference pipeline initialized")
    
    def _init_vila_model(self):
        """Initialize VILA model."""
        vila_config = self.config["model"]["vila"]
        self.vila_client = VILAClient(**vila_config)
        logger.info(f"Loaded VILA model: {vila_config['model_name']}")
    
    def _init_depth_estimator(self):
        """Initialize depth estimation model."""
        depth_config = self.config["model"]["depth"]
        estimator_type = depth_config["estimator"]
        
        if estimator_type == "midas":
            self.depth_estimator = MiDaSEstimator(
                model_type=depth_config["model_type"],
                device=depth_config["device"],
            )
        elif estimator_type == "zoe":
            self.depth_estimator = ZoeDepthEstimator(
                model_type=depth_config["model_type"],
                device=depth_config["device"],
            )
        else:
            raise ValueError(f"Unknown depth estimator: {estimator_type}")
        
        logger.info(f"Loaded depth estimator: {estimator_type}")
    
    def _init_detectors(self):
        """Initialize object detection and referent selection."""
        self.object_detector = YOLODetector()
        self.referent_selector = ReferentSelector()
        logger.info("Loaded object detectors")
    
    def _init_embedding_components(self):
        """Initialize embedding processing components."""
        self.roi_pooler = ROIPooler()
        
        # TODO: Get actual embedding dimensions from VILA
        vision_dim = 768  # Placeholder
        self.embedding_augmenter = EmbeddingAugmenter(
            vision_dim=vision_dim,
            distance_dim=1,
            method="concat",
        )
        logger.info("Initialized embedding components")
    
    def _init_distance_head(self):
        """Initialize distance prediction head."""
        head_config = self.config["model"]["head"]
        
        if head_config["type"] == "mlp":
            self.distance_head = MLPHead(
                input_dim=768 + 1,  # vision + distance placeholder
                hidden_dims=head_config["hidden_dims"],
                output_dim=head_config["output_dim"],
                dropout=head_config["dropout"],
            )
        elif head_config["type"] == "lora":
            self.distance_head = LoRAHead(
                input_dim=768 + 1,
                output_dim=head_config["output_dim"],
            )
        else:
            raise ValueError(f"Unknown head type: {head_config['type']}")
        
        # Load pretrained weights if available
        checkpoint_path = self.config.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            self.distance_head.load_state_dict(torch.load(checkpoint_path))
            logger.info(f"Loaded distance head from {checkpoint_path}")
    
    def infer(
        self,
        image: Image.Image,
        text: str,
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Run end-to-end inference.
        
        Args:
            image: Input image
            text: Input text/question
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Inference results including response and embeddings
        """
        results = {}
        
        # 1. Object detection
        detected_objects = self.object_detector.detect(image)
        results["detected_objects"] = [obj.to_dict() for obj in detected_objects]
        logger.info(f"Detected {len(detected_objects)} objects")
        
        # 2. Depth estimation
        depth_map = self.depth_estimator.estimate_depth(image)
        results["depth_map"] = depth_map
        
        # 3. Referent selection (if question refers to specific object)
        target_object = self.referent_selector.select_referent(
            text, detected_objects, image.size
        )
        if target_object:
            results["target_object"] = target_object.object_box.to_dict()
            logger.info(f"Selected referent: {target_object.object_box.class_name}")
        
        # 4. Extract vision embeddings
        vision_embeddings = self.vila_client.extract_embeddings(image, text)
        
        # 5. Pool embeddings for detected objects
        object_embeddings = []
        object_distances = []
        
        for obj in detected_objects:
            # Pool vision embedding for object region
            bbox = (obj.x1, obj.y1, obj.x2, obj.y2)
            
            # Get object depth
            distance = self.depth_estimator.get_object_depth(depth_map, bbox)
            object_distances.append(distance)
            
            # TODO: Implement actual ROI pooling
            # For now, use global embedding
            object_embeddings.append(vision_embeddings)
        
        if object_embeddings:
            object_embeddings = torch.stack(object_embeddings)
            object_distances = torch.tensor(object_distances).unsqueeze(-1)
            
            # 6. Augment embeddings with distance
            augmented_embeddings = self.embedding_augmenter(
                object_embeddings, object_distances
            )
            
            # 7. Predict refined distances
            predicted_distances = self.distance_head.predict_distance(augmented_embeddings)
            results["predicted_distances"] = predicted_distances.tolist()
        
        # 8. Generate final response
        response = self.vila_client.generate_response(image, text)
        results["response"] = response
        
        if return_intermediate:
            results["vision_embeddings"] = vision_embeddings
            results["augmented_embeddings"] = augmented_embeddings if object_embeddings else None
        
        return results
    
    def batch_infer(
        self,
        image_text_pairs: List[tuple],
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of image-text pairs.
        
        Args:
            image_text_pairs: List of (image, text) tuples
            batch_size: Batch size for processing
            
        Returns:
            List of inference results
        """
        results = []
        
        for i in range(0, len(image_text_pairs), batch_size):
            batch = image_text_pairs[i:i + batch_size]
            
            for image, text in batch:
                result = self.infer(image, text)
                results.append(result)
        
        return results


def main():
    """CLI entry point for inference."""
    import argparse
    from ..utils.config import load_config
    
    parser = argparse.ArgumentParser(description="Run MVDE inference")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--text", required=True, help="Input text/question")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize pipeline
    pipeline = InferencePipeline(config)
    
    # Load image
    image = Image.open(args.image)
    
    # Run inference
    results = pipeline.infer(image, args.text, return_intermediate=True)
    
    # Print results
    print(f"Question: {args.text}")
    print(f"Answer: {results['response']}")
    print(f"Detected {len(results['detected_objects'])} objects")
    
    if "target_object" in results:
        target = results["target_object"]
        print(f"Target object: {target['class_name']} at {target['center']}")
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            # Convert tensors to lists for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, torch.Tensor):
                    json_results[k] = v.tolist()
                elif hasattr(v, "tolist"):
                    json_results[k] = v.tolist()
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)


if __name__ == "__main__":
    main()
