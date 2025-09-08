"""MLP-based distance prediction head."""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPHead(nn.Module):
    """Multi-layer perceptron for distance prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        """
        Initialize MLP head.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for scalar distance, 3 for vector)
            dropout: Dropout probability
            activation: Activation function
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            Distance predictions [batch_size, output_dim]
        """
        return self.layers(x)
    
    def predict_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict distances from embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Predicted distances
        """
        with torch.no_grad():
            distances = self.forward(embeddings)
            
            # Apply activation for distance (ensure positive)
            if self.output_dim == 1:
                distances = torch.relu(distances)  # Scalar distance
            
            return distances
    
    def get_feature_importance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance scores using gradients.
        
        Args:
            embeddings: Input embeddings (requires_grad=True)
            
        Returns:
            Feature importance scores
        """
        embeddings.requires_grad_(True)
        output = self.forward(embeddings)
        
        # Compute gradients
        output.backward(torch.ones_like(output))
        importance = embeddings.grad.abs().mean(dim=0)
        
        return importance
