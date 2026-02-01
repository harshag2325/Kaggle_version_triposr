from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...utils import BaseModule
from .basic_transformer_block import BasicTransformerBlock


class Transformer1D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 88
        in_channels: Optional[int] = None
        out_channels: Optional[int] = None
        num_layers: int = 1
        dropout: float = 0.0
        norm_num_groups: int = 32
        cross_attention_dim: Optional[int] = None
        attention_bias: bool = False
        activation_fn: str = "geglu"
        only_cross_attention: bool = False
        double_self_attention: bool = False
        upcast_attention: bool = False
        norm_type: str = "layer_norm"
        norm_elementwise_affine: bool = True
        gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        # Cache dimensions
        self.num_attention_heads = self.cfg.num_attention_heads
        self.attention_head_dim = self.cfg.attention_head_dim
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.in_channels = self.cfg.in_channels
        self.out_channels = self.cfg.out_channels or self.cfg.in_channels
        self.gradient_checkpointing = self.cfg.gradient_checkpointing

        # Input normalization and projection
        self.norm = nn.GroupNorm(
            num_groups=self.cfg.norm_num_groups,
            num_channels=self.cfg.in_channels,
            eps=1e-6,
            affine=True,
        )
        self.proj_in = nn.Linear(self.cfg.in_channels, self.inner_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                self.inner_dim,
                self.num_attention_heads,
                self.attention_head_dim,
                dropout=self.cfg.dropout,
                cross_attention_dim=self.cfg.cross_attention_dim,
                activation_fn=self.cfg.activation_fn,
                attention_bias=self.cfg.attention_bias,
                only_cross_attention=self.cfg.only_cross_attention,
                double_self_attention=self.cfg.double_self_attention,
                upcast_attention=self.cfg.upcast_attention,
                norm_type=self.cfg.norm_type,
                norm_elementwise_affine=self.cfg.norm_elementwise_affine,
            )
            for _ in range(self.cfg.num_layers)
        ])

        # Output projection
        self.proj_out = nn.Linear(self.inner_dim, self.cfg.in_channels)

    def _convert_mask_to_bias(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Convert binary mask to attention bias.
        
        Args:
            mask: Binary mask of shape (batch, seq_len) where 1 = keep, 0 = discard
            dtype: Target dtype for the bias
            
        Returns:
            Attention bias of shape (batch, 1, seq_len)
        """
        return ((1 - mask.to(dtype)) * -10000.0).unsqueeze(1)

    def _prepare_attention_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare attention masks by converting them to biases if needed."""
        # Convert attention_mask
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = self._convert_mask_to_bias(attention_mask, dtype)

        # Convert encoder_attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = self._convert_mask_to_bias(encoder_attention_mask, dtype)

        return attention_mask, encoder_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer1D model.

        Args:
            hidden_states: Input tensor of shape (batch, channels, seq_len)
            encoder_hidden_states: Conditional embeddings of shape (batch, seq_len, embed_dim), optional
            attention_mask: Attention mask of shape (batch, key_tokens), optional
            encoder_attention_mask: Cross-attention mask, optional

        Returns:
            Output tensor of shape (batch, channels, seq_len)
        """
        # Prepare attention masks
        attention_mask, encoder_attention_mask = self._prepare_attention_masks(
            attention_mask, encoder_attention_mask, hidden_states.dtype
        )

        # 1. Input processing
        batch, _, seq_len = hidden_states.shape
        residual = hidden_states

        # Normalize, reshape, and project
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).reshape(batch, seq_len, self.inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # 2. Transformer blocks
        if self.training and self.gradient_checkpointing:
            # Use gradient checkpointing for memory efficiency
            for block in self.transformer_blocks:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_reentrant=False,
                )
        else:
            # Standard forward pass
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

        # 3. Output processing
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, seq_len, self.inner_dim).permute(0, 2, 1)

        return hidden_states + residual