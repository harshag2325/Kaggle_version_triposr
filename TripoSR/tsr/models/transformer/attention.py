from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    r"""
    A cross attention layer with support for multiple attention processors.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*): The number of channels in the encoder_hidden_states.
        heads (`int`, *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`, *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False): Set to `True` for query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False): Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False): Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*): The type of normalization for cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32): The number of groups for group norm in cross attention.
        added_kv_proj_dim (`int`, *optional*): The number of channels for added key and value projections.
        norm_num_groups (`int`, *optional*): The number of groups for group norm in attention.
        out_bias (`bool`, *optional*, defaults to `True`): Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`): Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`): Set to `True` to only use cross attention.
        eps (`float`, *optional*, defaults to 1e-5): An additional value added to the denominator in group normalization.
        rescale_output_factor (`float`, *optional*, defaults to 1.0): A factor to rescale the output.
        residual_connection (`bool`, *optional*, defaults to `False`): Set to `True` to add the residual connection.
        out_dim (`int`, *optional*): Output dimension. If None, defaults to query_dim.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # Core dimensions
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim or query_dim
        self.out_dim = out_dim or query_dim
        self.inner_dim = out_dim or (dim_head * heads)
        self.heads = self.out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        
        # Attention settings
        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if scale_qk else 1.0
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        
        # Additional KV projections
        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention
        
        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None."
            )
        
        # Normalization layers
        self.group_norm = (
            nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
            if norm_num_groups is not None
            else None
        )
        self.spatial_norm = None
        self.norm_cross = self._create_cross_norm(
            cross_attention_norm, cross_attention_norm_num_groups, eps, added_kv_proj_dim
        )
        
        # Projection layers
        self.linear_cls = nn.Linear
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        
        if not self.only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None
        
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Dropout(dropout)
        )
        
        # Set attention processor
        if processor is None:
            processor = (
                AttnProcessor2_0()
                if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
                else AttnProcessor()
            )
        self.set_processor(processor)

    def _create_cross_norm(
        self,
        cross_attention_norm: Optional[str],
        num_groups: int,
        eps: float,
        added_kv_proj_dim: Optional[int],
    ) -> Optional[nn.Module]:
        """Create cross attention normalization layer."""
        if cross_attention_norm is None:
            return None
        
        if cross_attention_norm == "layer_norm":
            return nn.LayerNorm(self.cross_attention_dim)
        
        if cross_attention_norm == "group_norm":
            norm_cross_num_channels = (
                added_kv_proj_dim if added_kv_proj_dim is not None 
                else self.cross_attention_dim
            )
            return nn.GroupNorm(
                num_channels=norm_cross_num_channels,
                num_groups=num_groups,
                eps=eps,
                affine=True,
            )
        
        raise ValueError(
            f"unknown cross_attention_norm: {cross_attention_norm}. "
            "Should be None, 'layer_norm' or 'group_norm'"
        )

    def set_processor(self, processor: "AttnProcessor") -> None:
        """Set the attention processor."""
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """Forward pass through attention layer."""
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape from [batch_size, seq_len, dim] to [batch_size // heads, seq_len, dim * heads]."""
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        return (
            tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size // head_size, seq_len, dim * head_size)
        )

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        """Reshape from [batch_size, seq_len, dim] to [batch_size, seq_len, heads, dim // heads]."""
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size).permute(0, 2, 1, 3)
        
        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        
        return tensor

    def get_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention scores."""
        dtype = query.dtype
        
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        
        # Prepare input for baddbmm
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1],
                dtype=query.dtype, device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1
        
        # Compute attention scores
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        
        attention_probs = attention_scores.softmax(dim=-1).to(dtype)
        
        return attention_probs

    def prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> Optional[torch.Tensor]:
        """Prepare attention mask for attention computation."""
        if attention_mask is None:
            return None
        
        head_size = self.heads
        current_length = attention_mask.shape[-1]
        
        # Pad if necessary
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # MPS workaround: manual padding construction
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        
        # Adjust dimensions
        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1).repeat_interleave(head_size, dim=1)
        
        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize encoder hidden states."""
        assert self.norm_cross is not None, (
            "self.norm_cross must be defined to call self.norm_encoder_hidden_states"
        )
        
        if isinstance(self.norm_cross, nn.LayerNorm):
            return self.norm_cross(encoder_hidden_states)
        
        if isinstance(self.norm_cross, nn.GroupNorm):
            # GroupNorm expects (N, C, *), so transpose for (batch, seq, hidden) -> (batch, hidden, seq)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            return encoder_hidden_states.transpose(1, 2)
        
        raise ValueError(f"Unsupported norm type: {type(self.norm_cross)}")

    @torch.no_grad()
    def fuse_projections(self, fuse: bool = True) -> None:
        """Fuse Q, K, V projections for efficiency."""
        is_cross_attention = self.cross_attention_dim != self.query_dim
        device = self.to_q.weight.device
        dtype = self.to_q.weight.dtype
        
        if not is_cross_attention:
            # Fuse Q, K, V for self-attention
            concatenated_weights = torch.cat([
                self.to_q.weight.data,
                self.to_k.weight.data,
                self.to_v.weight.data
            ])
            in_features, out_features = concatenated_weights.shape[1], concatenated_weights.shape[0]
            
            self.to_qkv = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
        else:
            # Fuse K, V for cross-attention
            concatenated_weights = torch.cat([
                self.to_k.weight.data,
                self.to_v.weight.data
            ])
            in_features, out_features = concatenated_weights.shape[1], concatenated_weights.shape[0]
            
            self.to_kv = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
        
        self.fused_projections = fuse


class AttnProcessor:
    r"""Default processor for performing attention-related computations."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        
        # Handle 4D input (image-like)
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # Determine sequence length
        batch_size, sequence_length, _ = (
            encoder_hidden_states.shape if encoder_hidden_states is not None 
            else hidden_states.shape
        )
        
        # Prepare attention mask
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        # Apply group norm if present
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Compute attention
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Output projection and dropout
        hidden_states = attn.to_out(hidden_states)
        
        # Reshape back to 4D if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        # Add residual and rescale
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class AttnProcessor2_0:
    r"""Processor for implementing scaled dot-product attention (PyTorch 2.0+)."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0. Please upgrade PyTorch to 2.0 or later."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        
        # Handle 4D input
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        # Determine sequence length
        batch_size, sequence_length, _ = (
            encoder_hidden_states.shape if encoder_hidden_states is not None 
            else hidden_states.shape
        )
        
        # Prepare attention mask for SDPA (needs 4D shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        
        # Apply group norm if present
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Reshape for multi-head attention (SDPA format)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attention_mask, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        # Reshape back
        hidden_states = (
            hidden_states.transpose(1, 2)
            .reshape(batch_size, -1, attn.heads * head_dim)
            .to(query.dtype)
        )
        
        # Output projection and dropout
        hidden_states = attn.to_out(hidden_states)
        
        # Reshape back to 4D if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        # Add residual and rescale
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states