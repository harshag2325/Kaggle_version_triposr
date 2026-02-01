from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from ..utils import BaseModule


class TriplaneUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int
        out_channels: int

    cfg: Config

    def configure(self) -> None:
        self.upsample = nn.ConvTranspose2d(
            self.cfg.in_channels, 
            self.cfg.out_channels, 
            kernel_size=2, 
            stride=2
        )

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        # Merge batch and plane dimensions for efficient processing
        B, Np = triplanes.shape[0], 3
        triplanes_flat = rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=Np)
        upsampled = self.upsample(triplanes_flat)
        return rearrange(upsampled, "(B Np) Co Hp Wp -> B Np Co Hp Wp", Np=Np)


class NeRFMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int
        n_neurons: int
        n_hidden_layers: int
        activation: str = "relu"
        bias: bool = True
        weight_init: Optional[str] = "kaiming_uniform"
        bias_init: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        # Build layers more efficiently
        layers = [
            self._make_linear(self.cfg.in_channels, self.cfg.n_neurons),
            self._make_activation(),
        ]
        
        # Add hidden layers
        layers.extend([
            layer
            for _ in range(self.cfg.n_hidden_layers - 1)
            for layer in (self._make_linear(self.cfg.n_neurons, self.cfg.n_neurons), 
                         self._make_activation())
        ])
        
        # Output layer (density 1 + features 3)
        layers.append(self._make_linear(self.cfg.n_neurons, 4))
        
        self.layers = nn.Sequential(*layers)
        
        # Cache activation type for reuse
        self._activation_fn = self._get_activation_fn()

    def _make_linear(self, dim_in: int, dim_out: int) -> nn.Linear:
        """Create and initialize a linear layer."""
        layer = nn.Linear(dim_in, dim_out, bias=self.cfg.bias)
        
        # Weight initialization
        if self.cfg.weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif self.cfg.weight_init is not None:
            raise ValueError(f"Unknown weight_init: {self.cfg.weight_init}")
        
        # Bias initialization
        if self.cfg.bias:
            if self.cfg.bias_init == "zero":
                nn.init.zeros_(layer.bias)
            elif self.cfg.bias_init is not None:
                raise ValueError(f"Unknown bias_init: {self.cfg.bias_init}")
        
        return layer

    def _get_activation_fn(self) -> nn.Module:
        """Get activation function class."""
        activations = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if self.cfg.activation not in activations:
            raise ValueError(f"Unknown activation: {self.cfg.activation}")
        return activations[self.cfg.activation]

    def _make_activation(self) -> nn.Module:
        """Create activation layer."""
        return self._get_activation_fn()(inplace=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_channels)
            
        Returns:
            Dictionary with 'density' (..., 1) and 'features' (..., 3)
        """
        inp_shape = x.shape[:-1]
        
        # Flatten all dims except last, process, then reshape
        x_flat = x.view(-1, x.shape[-1])
        features = self.layers(x_flat)
        features = features.view(*inp_shape, 4)
        
        # Split into density and features (more efficient than slicing twice)
        return {
            "density": features[..., :1],
            "features": features[..., 1:]
        }