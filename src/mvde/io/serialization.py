"""Serialization utilities for embeddings and metadata."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def save_embeddings(
    embeddings: Union[np.ndarray, torch.Tensor],
    metadata: Dict[str, Any],
    save_path: str,
    format: str = "numpy",
) -> None:
    """
    Save embeddings and metadata to disk.
    
    Args:
        embeddings: Embedding vectors
        metadata: Associated metadata (object names, distances, etc.)
        save_path: Path to save files (without extension)
        format: Save format ("numpy", "torch", "pickle")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert torch tensors to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Save embeddings
    if format == "numpy":
        np.save(f"{save_path}_embeddings.npy", embeddings_np)
    elif format == "torch":
        if isinstance(embeddings, torch.Tensor):
            torch.save(embeddings, f"{save_path}_embeddings.pt")
        else:
            torch.save(torch.from_numpy(embeddings_np), f"{save_path}_embeddings.pt")
    elif format == "pickle":
        with open(f"{save_path}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_np, f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Save metadata as JSON
    with open(f"{save_path}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_embeddings(
    load_path: str,
    format: str = "numpy",
) -> tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
    """
    Load embeddings and metadata from disk.
    
    Args:
        load_path: Path to load files from (without extension)
        format: Load format ("numpy", "torch", "pickle")
        
    Returns:
        Tuple of (embeddings, metadata)
    """
    load_path = Path(load_path)
    
    # Load embeddings
    if format == "numpy":
        embeddings = np.load(f"{load_path}_embeddings.npy")
    elif format == "torch":
        embeddings = torch.load(f"{load_path}_embeddings.pt")
    elif format == "pickle":
        with open(f"{load_path}_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Load metadata
    with open(f"{load_path}_metadata.json", "r") as f:
        metadata = json.load(f)
    
    return embeddings, metadata


def save_experiment_results(
    results: Dict[str, Any],
    save_dir: str,
    experiment_name: str,
) -> None:
    """
    Save complete experiment results.
    
    Args:
        results: Dictionary containing all experiment results
        save_dir: Directory to save results
        experiment_name: Name of the experiment
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each component
    for key, value in results.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            save_embeddings(
                value, 
                {"component": key, "experiment": experiment_name},
                save_dir / f"{experiment_name}_{key}"
            )
        else:
            # Save as JSON
            with open(save_dir / f"{experiment_name}_{key}.json", "w") as f:
                json.dump(value, f, indent=2, default=str)


def load_experiment_results(
    load_dir: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Load complete experiment results.
    
    Args:
        load_dir: Directory to load results from
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary containing all experiment results
    """
    load_dir = Path(load_dir)
    results = {}
    
    # Load all files matching the experiment name
    for file_path in load_dir.glob(f"{experiment_name}*"):
        if file_path.suffix == ".npy":
            key = file_path.stem.replace(f"{experiment_name}_", "").replace("_embeddings", "")
            embeddings, metadata = load_embeddings(
                str(file_path).replace("_embeddings.npy", "")
            )
            results[key] = embeddings
        elif file_path.suffix == ".json" and "metadata" not in file_path.name:
            key = file_path.stem.replace(f"{experiment_name}_", "")
            with open(file_path, "r") as f:
                results[key] = json.load(f)
    
    return results


def dump_embeddings_cli() -> None:
    """CLI entry point for dumping embeddings."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dump embeddings to files")
    parser.add_argument("--input", required=True, help="Input embeddings file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--format", default="numpy", choices=["numpy", "torch", "pickle"])
    
    args = parser.parse_args()
    
    # TODO: Implement CLI functionality
    print(f"Dumping embeddings from {args.input} to {args.output} in {args.format} format")


if __name__ == "__main__":
    dump_embeddings_cli()
