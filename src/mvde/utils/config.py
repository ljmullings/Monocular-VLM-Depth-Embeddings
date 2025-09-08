"""Configuration management using OmegaConf."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class Config(BaseModel):
    """Pydantic model for type-safe configuration."""
    
    class Config:
        extra = "allow"  # Allow additional fields


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    local_config: Optional[str] = None,
) -> DictConfig:
    """
    Load configuration with optional overrides and local settings.
    
    Args:
        config_path: Path to the main config file
        overrides: Dictionary of config overrides
        local_config: Path to local configuration file
        
    Returns:
        Merged configuration object
    """
    # Load base config
    cfg = OmegaConf.load(config_path)
    
    # Load local config if exists
    if local_config and Path(local_config).exists():
        local_cfg = OmegaConf.load(local_config)
        cfg = OmegaConf.merge(cfg, local_cfg)
    
    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    # Resolve interpolations
    cfg = OmegaConf.create(cfg)
    OmegaConf.resolve(cfg)
    
    return cfg


def get_default_config() -> DictConfig:
    """Get default configuration."""
    # TODO: Implement default config creation
    return OmegaConf.create({})


def save_config(cfg: DictConfig, path: str) -> None:
    """Save configuration to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(cfg, path)
