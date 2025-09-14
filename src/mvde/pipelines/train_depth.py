"""Training pipeline for end-to-end depth estimation with SigLIP backbone."""

import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from ..model import DepthModel, DepthModelFactory
from ..evaluation.metrics import DepthMetrics
from ..export.patch_stats import PatchStatsExtractor
from ..utils.logging import get_logger, setup_logging
from ..utils.config import load_config
from ..io.datasets import load_dataset, create_dataloader

logger = get_logger(__name__)


class DepthTrainingPipeline:
    """
    Training pipeline for end-to-end depth estimation using SigLIP + DPT.
    
    Supports:
    - Multi-scale feature extraction with SigLIP
    - Dense depth prediction with DPT decoder
    - Backbone freezing/unfreezing schedule
    - Scale-invariant depth losses
    - Integration with existing evaluation metrics
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize depth training pipeline.
        
        Args:
            config: Configuration with vit, decoder, train sections
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_step = 0
        self.best_metric = float('inf')
        
        # Initialize model
        self._init_model()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Initialize data loaders
        self._init_data_loaders()
        
        # Initialize metrics and patch stats extractor
        self._init_evaluation()
        
        # Initialize logging
        self._init_logging()
        
        logger.info("Depth training pipeline initialized")
        logger.info(f"Model info: {self.model.get_model_info()}")
    
    def _init_model(self) -> None:
        """Initialize SigLIP depth model."""
        self.model = DepthModel(self.config)
        self.model.to(self.device)
        
        # Log model information
        info = self.model.get_model_info()
        logger.info(f"Initialized SigLIP depth model:")
        logger.info(f"  - Model: {info['model_name']}")
        logger.info(f"  - Image size: {info['image_size']}")
        logger.info(f"  - Patch size: {info['patch_size']}")
        logger.info(f"  - Total parameters: {info['total_parameters']:,}")
        logger.info(f"  - Trainable parameters: {info['trainable_parameters']:,}")
        logger.info(f"  - Backbone frozen: {info['backbone_frozen']}")
    
    def _init_optimizer(self) -> None:
        """Initialize optimizer, scheduler, and loss function."""
        train_config = self.config.train
        
        # Create parameter groups for different learning rates
        backbone_params = list(self.model.backbone.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        param_groups = [
            {
                "params": decoder_params,
                "lr": train_config.lr,
                "name": "decoder"
            }
        ]
        
        # Add backbone params with potentially different LR
        if self.model._is_backbone_trainable():
            backbone_lr = train_config.get("backbone_lr", train_config.lr * 0.1)
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "name": "backbone"
            })
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=train_config.get("weight_decay", 1e-4),
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.get("max_steps", 50000),
            eta_min=train_config.lr * 0.01,
        )
        
        # Loss function
        self.loss_type = train_config.get("loss_type", "scale_invariant")
        if self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            # Use model's built-in loss computation for scale-invariant
            self.criterion = None
        
        logger.info(f"Initialized optimizer with {len(param_groups)} parameter groups")
        logger.info(f"Loss type: {self.loss_type}")
    
    def _init_data_loaders(self) -> None:
        """Initialize training and validation data loaders."""
        # TODO: Implement actual data loading
        # For now, create dummy loaders
        logger.warning("Using dummy data loaders - implement actual dataset loading")
        
        # Placeholder - you'll need to implement actual dataset classes
        self.train_loader = None
        self.val_loader = None
        
        # Log data info when implemented
        # logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        # logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _init_evaluation(self) -> None:
        """Initialize evaluation metrics and patch statistics extractor."""
        self.depth_metrics = DepthMetrics()
        
        # Initialize patch stats extractor for VLM integration
        self.patch_extractor = PatchStatsExtractor(
            patch_size=self.model.patch_size,
            stats=("mean", "var", "grad"),
            normalize_stats=True,
            log_transform_depth=True,
        )
    
    def _init_logging(self) -> None:
        """Initialize experiment logging (wandb, etc.)."""
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "mvde-depth"),
                name=self.config.get("experiment_name", "siglip-depth"),
                config=OmegaConf.to_container(self.config, resolve=True),
            )
    
    def train_step(
        self,
        images: torch.Tensor,
        depth_targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            images: Input images (B, 3, H, W)
            depth_targets: Target depth maps (B, 1, H, W)
            masks: Optional valid pixel masks (B, 1, H, W)
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(images)
        pred_depth = outputs["depth"]
        
        # Compute loss
        if self.criterion is not None:
            # Standard loss functions
            if masks is not None:
                pred_masked = pred_depth[masks]
                target_masked = depth_targets[masks]
                loss = self.criterion(pred_masked, target_masked)
            else:
                loss = self.criterion(pred_depth, depth_targets)
        else:
            # Use model's scale-invariant loss
            loss = self.model.decoder.compute_loss(
                pred_depth, depth_targets, masks, self.loss_type
            )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.train.get("grad_clip", 1.0)
        )
        
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Update model state (for backbone unfreezing)
        self.model.training_step(self.current_step)
        self.current_step += 1
        
        # Compute metrics
        with torch.no_grad():
            metrics = self._compute_metrics(pred_depth, depth_targets, masks)
        
        # Add loss and training info
        metrics.update({
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self.current_step,
        })
        
        return metrics
    
    def validation_step(
        self,
        images: torch.Tensor,
        depth_targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            images: Input images (B, 3, H, W)
            depth_targets: Target depth maps (B, 1, H, W)
            masks: Optional valid pixel masks (B, 1, H, W)
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(images)
            pred_depth = outputs["depth"]
            
            # Compute loss
            if self.criterion is not None:
                if masks is not None:
                    pred_masked = pred_depth[masks]
                    target_masked = depth_targets[masks]
                    loss = self.criterion(pred_masked, target_masked)
                else:
                    loss = self.criterion(pred_depth, depth_targets)
            else:
                loss = self.model.decoder.compute_loss(
                    pred_depth, depth_targets, masks, self.loss_type
                )
            
            # Compute metrics
            metrics = self._compute_metrics(pred_depth, depth_targets, masks)
            metrics["val_loss"] = loss.item()
        
        return metrics
    
    def _compute_metrics(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute depth estimation metrics."""
        # Convert to numpy for metrics computation
        pred_np = pred_depth.detach().cpu().numpy()
        target_np = target_depth.detach().cpu().numpy()
        mask_np = masks.detach().cpu().numpy() if masks is not None else None
        
        # Compute standard depth metrics
        metrics = {}
        for i in range(pred_np.shape[0]):  # Batch dimension
            batch_metrics = self.depth_metrics.compute_all(
                pred_np[i, 0], target_np[i, 0], mask_np[i, 0] if mask_np is not None else None
            )
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average across batch
        for key in metrics:
            metrics[key] = sum(metrics[key]) / len(metrics[key])
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        if self.train_loader is None:
            logger.error("Training data loader not implemented")
            return {}
        
        self.model.train()
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract batch data (implement based on your dataset format)
            images = batch["images"].to(self.device)
            depth_targets = batch["depth"].to(self.device)
            masks = batch.get("masks", None)
            if masks is not None:
                masks = masks.to(self.device)
            
            # Training step
            metrics = self.train_step(images, depth_targets, masks)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rmse": f"{metrics.get('rmse', 0):.3f}",
                "lr": f"{metrics['lr']:.2e}",
            })
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                wandb.log(metrics, step=self.current_step)
        
        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] = sum(epoch_metrics[key]) / len(epoch_metrics[key])
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            logger.warning("Validation data loader not implemented")
            return {}
        
        self.model.eval()
        epoch_metrics = {}
        
        pbar = tqdm(self.val_loader, desc=f"Validation {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(self.device)
            depth_targets = batch["depth"].to(self.device)
            masks = batch.get("masks", None)
            if masks is not None:
                masks = masks.to(self.device)
            
            metrics = self.validation_step(images, depth_targets, masks)
            
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # Average validation metrics
        for key in epoch_metrics:
            epoch_metrics[key] = sum(epoch_metrics[key]) / len(epoch_metrics[key])
        
        return epoch_metrics
    
    def train(self, output_dir: str, num_epochs: Optional[int] = None) -> None:
        """
        Main training loop.
        
        Args:
            output_dir: Directory to save checkpoints and logs
            num_epochs: Number of epochs (defaults to config)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        num_epochs = num_epochs or self.config.train.epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Output directory: {output_dir}")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                       f"RMSE: {train_metrics.get('rmse', 0):.3f}")
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            if val_metrics:
                logger.info(f"Val - Loss: {val_metrics.get('val_loss', 0):.4f}, "
                           f"RMSE: {val_metrics.get('rmse', 0):.3f}")
            
            # Save checkpoint
            if epoch % self.config.train.get("save_every", 5) == 0:
                checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(str(checkpoint_path))
            
            # Save best model
            current_metric = val_metrics.get('rmse', train_metrics.get('rmse', float('inf')))
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                best_path = output_path / "best_model.pt"
                self.save_checkpoint(str(best_path))
                logger.info(f"New best model saved (RMSE: {current_metric:.3f})")
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_step": self.current_step,
            "best_metric": self.best_metric,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_step = checkpoint["current_step"]
        self.best_metric = checkpoint["best_metric"]
        
        logger.info(f"Checkpoint loaded from {path} (step {self.current_step})")


def main():
    """CLI entry point for depth training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SigLIP depth estimation model")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--log_dir", help="Log directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(args.log_dir) / "depth_training.log"
        setup_logging(log_file=str(log_file))
    
    # Load config
    config = load_config(args.config)
    
    # Initialize training pipeline
    trainer = DepthTrainingPipeline(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.train(args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
