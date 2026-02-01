from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .attention import Attention


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (`bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*): Whether to use only cross-attention layers.
        double_self_attention (`bool`, *optional*): Whether to use two self-attention layers.
        upcast_attention (`bool`, *optional*): Whether to upcast the attention computation to float32.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`): Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`): The normalization layer to use.
        final_dropout (`bool` *optional*, defaults to False): Whether to apply a final dropout after the last feed-forward layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self._chunk_size = None
        self._chunk_dim = 0

        assert norm_type == "layer_norm", f"Only 'layer_norm' is supported, got {norm_type}"

        # 1. Self-Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attention (conditional)
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0) -> None:
        """Sets chunk feed-forward parameters."""
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # 1. Self-Attention
        attn_output = self.attn1(
            self.norm1(hidden_states),
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        if self.attn2 is not None:
            attn_output = self.attn2(
                self.norm2(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        ff_output = self._apply_feed_forward(self.norm3(hidden_states))
        hidden_states = hidden_states + ff_output

        return hidden_states

    def _apply_feed_forward(self, norm_hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward with optional chunking."""
        if self._chunk_size is None:
            return self.ff(norm_hidden_states)

        # Chunked feed-forward for memory efficiency
        dim_size = norm_hidden_states.shape[self._chunk_dim]
        if dim_size % self._chunk_size != 0:
            raise ValueError(
                f"Hidden states dimension {dim_size} must be divisible by chunk size {self._chunk_size}. "
                f"Adjust chunk_size in `unet.enable_forward_chunking`."
            )

        num_chunks = dim_size // self._chunk_size
        return torch.cat(
            [self.ff(chunk) for chunk in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
            dim=self._chunk_dim,
        )


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Create activation layer
        activation_map = {
            "gelu": lambda: GELU(dim, inner_dim),
            "gelu-approximate": lambda: GELU(dim, inner_dim, approximate="tanh"),
            "geglu": lambda: GEGLU(dim, inner_dim),
            "geglu-approximate": lambda: ApproximateGELU(dim, inner_dim),
        }
        
        if activation_fn not in activation_map:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")
        
        act_fn = activation_map[activation_fn]()

        # Build network as Sequential for efficiency
        layers = [
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        ]
        
        if final_dropout:
            layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        
        # MPS doesn't support gelu for float16
        if hidden_states.device.type == "mps" and hidden_states.dtype == torch.float16:
            return F.gelu(
                hidden_states.to(torch.float32), 
                approximate=self.approximate
            ).to(hidden_states.dtype)
        
        return F.gelu(hidden_states, approximate=self.approximate)


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        
        # MPS doesn't support gelu for float16
        if gate.device.type == "mps" and gate.dtype == torch.float16:
            gate = F.gelu(gate.to(torch.float32)).to(gate.dtype)
        else:
            gate = F.gelu(gate)
        
        return hidden_states * gate


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of Gaussian Error Linear Unit (GELU).
    For more details, see section 2: https://arxiv.org/abs/1606.08415.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.register_buffer("scale_factor", torch.tensor(1.702))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(self.scale_factor * x)