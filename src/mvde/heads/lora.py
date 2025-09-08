"""LoRA adapter for distance prediction."""

import torch
import torch.nn as nn
from typing import Optional


class LoRAHead(nn.Module):
    """LoRA (Low-Rank Adaptation) head for distance prediction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize LoRA head.
        
        Args:
            input_dim: Input embedding dimension
            output_dim: Output dimension (1 for scalar distance)
            rank: LoRA rank (bottleneck dimension)
            alpha: LoRA scaling parameter
            dropout: Dropout probability
            bias: Whether to include bias terms
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, output_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        # Initialize A with normal distribution
        nn.init.normal_(self.lora_A.weight, std=1/self.rank)
        
        # Initialize B with zeros (important for LoRA)
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layers.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            Distance predictions [batch_size, output_dim]
        """
        # LoRA forward: B(A(x))
        h = self.lora_A(x)
        h = self.dropout(h)
        output = self.lora_B(h)
        
        # Apply scaling
        output = output * self.scaling
        
        return output
    
    def predict_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict distances from embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Predicted distances (positive values)
        """
        with torch.no_grad():
            distances = self.forward(embeddings)
            
            # Ensure positive distances
            distances = torch.relu(distances)
            
            return distances
    
    def get_lora_parameters(self):
        """Get LoRA-specific parameters for training."""
        return [
            {"params": self.lora_A.parameters()},
            {"params": self.lora_B.parameters()},
        ]
    
    def merge_weights(self) -> torch.Tensor:
        """
        Merge LoRA weights into a single linear transformation.
        
        Returns:
            Merged weight matrix [output_dim, input_dim]
        """
        # Compute W = B @ A
        merged_weight = self.lora_B.weight @ self.lora_A.weight
        return merged_weight * self.scaling
    
    def save_lora_state(self, path: str):
        """Save LoRA-specific state."""
        state = {
            "lora_A": self.lora_A.state_dict(),
            "lora_B": self.lora_B.state_dict(),
            "rank": self.rank,
            "alpha": self.alpha,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        torch.save(state, path)
    
    def load_lora_state(self, path: str):
        """Load LoRA-specific state."""
        state = torch.load(path)
        self.lora_A.load_state_dict(state["lora_A"])
        self.lora_B.load_state_dict(state["lora_B"])
