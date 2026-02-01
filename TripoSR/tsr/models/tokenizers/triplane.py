import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...utils import BaseModule


class Triplane1DTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int
        num_channels: int

    cfg: Config

    def configure(self) -> None:
        # Pre-compute constants
        self.plane_size = self.cfg.plane_size
        self.num_channels = self.cfg.num_channels
        self.num_planes = 3
        self.total_tokens = self.num_planes * self.plane_size * self.plane_size
        
        # Initialize learnable triplane embeddings with proper scaling
        scale = 1.0 / math.sqrt(self.num_channels)
        self.embeddings = nn.Parameter(
            torch.randn(
                self.num_planes,
                self.num_channels,
                self.plane_size,
                self.plane_size,
                dtype=torch.float32,
            ) * scale
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate triplane tokens for a batch.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Flattened triplane tokens of shape (B, C, num_planes * H * W)
        """
        # Efficient batching: expand and reshape in one go
        return rearrange(
            self.embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1, -1),
            "B Np C H W -> B C (Np H W)",
        )

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert flattened tokens back to triplane representation.
        
        Args:
            tokens: Flattened tokens of shape (B, C, num_tokens)
            
        Returns:
            Triplane representation of shape (B, num_planes, C, H, W)
        """
        batch_size, num_channels, num_tokens = tokens.shape
        
        # Validate input dimensions
        assert num_tokens == self.total_tokens, (
            f"Expected {self.total_tokens} tokens "
            f"({self.num_planes} planes × {self.plane_size}²), got {num_tokens}"
        )
        assert num_channels == self.num_channels, (
            f"Expected {self.num_channels} channels, got {num_channels}"
        )
        
        return rearrange(
            tokens,
            "B C (Np H W) -> B Np C H W",
            Np=self.num_planes,
            H=self.plane_size,
            W=self.plane_size,
        )

    def get_plane(self, tokens: torch.Tensor, plane_idx: int) -> torch.Tensor:
        """
        Extract a specific plane from triplane tokens.
        
        Args:
            tokens: Flattened tokens of shape (B, C, num_tokens)
            plane_idx: Index of plane to extract (0, 1, or 2)
            
        Returns:
            Single plane of shape (B, C, H, W)
        """
        assert 0 <= plane_idx < self.num_planes, (
            f"plane_idx must be in [0, {self.num_planes}), got {plane_idx}"
        )
        
        triplanes = self.detokenize(tokens)
        return triplanes[:, plane_idx]

    def reset_parameters(self, scale: Optional[float] = None) -> None:
        """
        Reinitialize embeddings with optional custom scale.
        
        Args:
            scale: Initialization scale. If None, uses 1/sqrt(num_channels)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.num_channels)
        
        nn.init.normal_(self.embeddings, mean=0.0, std=scale)

    @property
    def embedding_shape(self) -> tuple[int, int, int, int]:
        """Get the shape of a single triplane embedding."""
        return (self.num_planes, self.num_channels, self.plane_size, self.plane_size)

    @property
    def flattened_shape(self) -> tuple[int, int]:
        """Get the shape of flattened tokens (per sample)."""
        return (self.num_channels, self.total_tokens)


class AdaptiveTriplane1DTokenizer(Triplane1DTokenizer):
    """
    Triplane tokenizer with adaptive resolution support.
    Allows interpolation to different plane sizes at runtime.
    """
    
    def forward(
        self, 
        batch_size: int,
        target_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate triplane tokens with optional resolution adaptation.
        
        Args:
            batch_size: Number of samples in the batch
            target_size: Target plane size. If None, uses configured size.
            
        Returns:
            Flattened triplane tokens
        """
        if target_size is None or target_size == self.plane_size:
            return super().forward(batch_size)
        
        # Interpolate to target size
        embeddings = torch.nn.functional.interpolate(
            rearrange(self.embeddings, "Np C H W -> (Np) C H W"),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        embeddings = rearrange(
            embeddings, "(Np) C H W -> Np C H W", Np=self.num_planes
        )
        
        return rearrange(
            embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1, -1),
            "B Np C H W -> B C (Np H W)",
        )

    def detokenize(
        self, 
        tokens: torch.Tensor,
        plane_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Detokenize with automatic plane size inference.
        
        Args:
            tokens: Flattened tokens
            plane_size: Expected plane size. If None, inferred from token count.
            
        Returns:
            Triplane representation
        """
        batch_size, num_channels, num_tokens = tokens.shape
        
        # Infer plane size if not provided
        if plane_size is None:
            # num_tokens = num_planes * H * W
            plane_size = int(math.sqrt(num_tokens / self.num_planes))
            
            # Validate that it's a perfect square
            if plane_size * plane_size * self.num_planes != num_tokens:
                raise ValueError(
                    f"Cannot infer plane_size from {num_tokens} tokens "
                    f"(not divisible by {self.num_planes} into a perfect square)"
                )
        
        return rearrange(
            tokens,
            "B C (Np H W) -> B Np C H W",
            Np=self.num_planes,
            H=plane_size,
            W=plane_size,
        )