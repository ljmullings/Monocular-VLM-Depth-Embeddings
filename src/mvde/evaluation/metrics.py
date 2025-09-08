"""Evaluation metrics for depth and distance estimation."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union


class DepthMetrics:
    """Standard depth estimation metrics."""
    
    @staticmethod
    def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Root Mean Square Error."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        return np.sqrt(np.mean((pred - target) ** 2))
    
    @staticmethod
    def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Mean Absolute Error."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def abs_rel(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Absolute Relative Error."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        return np.mean(np.abs(pred - target) / target)
    
    @staticmethod
    def sq_rel(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Squared Relative Error."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        return np.mean(((pred - target) ** 2) / target)
    
    @staticmethod
    def delta_accuracy(
        pred: np.ndarray, 
        target: np.ndarray, 
        threshold: float = 1.25,
        mask: np.ndarray = None
    ) -> float:
        """Delta accuracy (percentage of pixels within threshold)."""
        if mask is not None:
            pred = pred[mask]
            target = target[mask]
        
        ratio = np.maximum(pred / target, target / pred)
        return np.mean(ratio < threshold)
    
    @classmethod
    def compute_all(
        cls,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray = None,
    ) -> Dict[str, float]:
        """Compute all depth metrics."""
        return {
            "rmse": cls.rmse(pred, target, mask),
            "mae": cls.mae(pred, target, mask),
            "abs_rel": cls.abs_rel(pred, target, mask),
            "sq_rel": cls.sq_rel(pred, target, mask),
            "delta_1": cls.delta_accuracy(pred, target, 1.25, mask),
            "delta_2": cls.delta_accuracy(pred, target, 1.25**2, mask),
            "delta_3": cls.delta_accuracy(pred, target, 1.25**3, mask),
        }


class DistanceMetrics:
    """Metrics for object distance estimation."""
    
    @staticmethod
    def distance_error(
        pred_distances: List[float],
        true_distances: List[float],
    ) -> Dict[str, float]:
        """Compute distance estimation errors."""
        pred = np.array(pred_distances)
        true = np.array(true_distances)
        
        errors = np.abs(pred - true)
        relative_errors = errors / true
        
        return {
            "mae": np.mean(errors),
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "mape": np.mean(relative_errors) * 100,  # Mean Absolute Percentage Error
            "median_error": np.median(errors),
            "std_error": np.std(errors),
        }
    
    @staticmethod
    def ordinal_accuracy(
        pred_distances: List[float],
        true_distances: List[float],
        pairs: List[Tuple[int, int]] = None,
    ) -> float:
        """
        Compute ordinal accuracy (correct relative ordering).
        
        Args:
            pred_distances: Predicted distances
            true_distances: Ground truth distances
            pairs: Optional specific pairs to evaluate
            
        Returns:
            Accuracy of ordinal relationships
        """
        if pairs is None:
            # Generate all pairs
            n = len(pred_distances)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        correct = 0
        total = len(pairs)
        
        for i, j in pairs:
            pred_closer = pred_distances[i] < pred_distances[j]
            true_closer = true_distances[i] < true_distances[j]
            
            if pred_closer == true_closer:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def depth_ranking_correlation(
        pred_distances: List[float],
        true_distances: List[float],
    ) -> float:
        """Compute Spearman rank correlation for distance ordering."""
        from scipy.stats import spearmanr
        corr, _ = spearmanr(pred_distances, true_distances)
        return corr
    
    @classmethod
    def compute_all(
        cls,
        pred_distances: List[float],
        true_distances: List[float],
    ) -> Dict[str, float]:
        """Compute all distance metrics."""
        metrics = cls.distance_error(pred_distances, true_distances)
        metrics["ordinal_accuracy"] = cls.ordinal_accuracy(pred_distances, true_distances)
        
        try:
            metrics["rank_correlation"] = cls.depth_ranking_correlation(
                pred_distances, true_distances
            )
        except:
            metrics["rank_correlation"] = 0.0
        
        return metrics


class EmbeddingMetrics:
    """Metrics for embedding quality evaluation."""
    
    @staticmethod
    def embedding_similarity(
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        metric: str = "cosine",
    ) -> float:
        """Compute similarity between embedding sets."""
        if metric == "cosine":
            sim = torch.cosine_similarity(embeddings1, embeddings2, dim=-1)
            return sim.mean().item()
        elif metric == "l2":
            dist = torch.norm(embeddings1 - embeddings2, dim=-1)
            return dist.mean().item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def embedding_clustering_score(
        embeddings: torch.Tensor,
        labels: List[str],
    ) -> Dict[str, float]:
        """Evaluate clustering quality of embeddings."""
        # TODO: Implement clustering metrics (silhouette score, etc.)
        return {"silhouette_score": 0.0, "adjusted_rand_index": 0.0}
