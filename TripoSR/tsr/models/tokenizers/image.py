from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.vit.modeling_vit import ViTModel, ViTConfig

from ...utils import BaseModule


class DINOSingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = r'/kaggle/input/models-weight-config-files/other/default/1/pytorch_model.bin'
        enable_gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        # Use Path for better path handling
        model_dir = Path(r"/kaggle/input/models-weight-config-files/other/default/1")
        
        # Load config and initialize model
        config = ViTConfig.from_json_file(str(model_dir / "v16config.json"))
        self.model = ViTModel(config)

        # Load pretrained weights efficiently
        self._load_pretrained_weights(model_dir / "pytorch_model.bin")

        # Enable gradient checkpointing if needed
        if self.cfg.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Register normalization constants (ImageNet stats)
        self._register_normalization_buffers()

    def _load_pretrained_weights(self, weights_path: Path) -> None:
        """Load pretrained weights with automatic filtering of incompatible keys."""
        # Determine device automatically
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        state_dict = torch.load(
            str(weights_path),
            map_location=device,
            weights_only=True  # Security: only load weights, not arbitrary objects
        )
        
        # Filter out pooler and other incompatible keys
        model_keys = set(self.model.state_dict().keys())
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_keys
        }
        
        # Load with informative logging
        missing_keys, unexpected_keys = self.model.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"Missing keys (will use random initialization): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {len(unexpected_keys)}")

    def _register_normalization_buffers(self) -> None:
        """Register ImageNet normalization statistics as buffers."""
        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        
        # Reshape for broadcasting: (1, 1, 3, 1, 1)
        self.register_buffer(
            "image_mean",
            mean.view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            std.view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self, 
        images: torch.FloatTensor,
        return_global_features: bool = False,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Forward pass through DINO tokenizer.
        
        Args:
            images: Input images of shape (B, C, H, W) or (B, N, C, H, W)
            return_global_features: If True, also return global features (pooler output)
            **kwargs: Additional arguments passed to the model
            
        Returns:
            Local features of shape (B, N, C, num_tokens) or (B, C, num_tokens) if packed
            If return_global_features=True, returns tuple (local_features, global_features)
        """
        # Handle 4D input (single view per batch item)
        packed = images.ndim == 4
        if packed:
            images = images.unsqueeze(1)

        batch_size, n_views = images.shape[:2]
        
        # Normalize images
        images = (images - self.image_mean) / self.image_std
        
        # Process through ViT
        images_flat = rearrange(images, "B N C H W -> (B N) C H W")
        out = self.model(images_flat, interpolate_pos_encoding=True)
        
        # Extract and reshape local features
        local_features = out.last_hidden_state.transpose(1, 2)  # (B*N, C, num_tokens)
        local_features = rearrange(
            local_features, "(B N) C T -> B N C T", B=batch_size, N=n_views
        )
        
        if packed:
            local_features = local_features.squeeze(1)
        
        if return_global_features:
            global_features = rearrange(
                out.pooler_output, "(B N) C -> B N C", B=batch_size, N=n_views
            )
            if packed:
                global_features = global_features.squeeze(1)
            return local_features, global_features
        
        return local_features

    @torch.no_grad()
    def extract_features(
        self, 
        images: torch.FloatTensor,
        layer_idx: Optional[int] = None
    ) -> torch.FloatTensor:
        """
        Extract features from a specific layer (for analysis/visualization).
        
        Args:
            images: Input images
            layer_idx: Layer index to extract from. If None, uses final layer.
            
        Returns:
            Features from specified layer
        """
        # Handle 4D input
        if images.ndim == 4:
            images = images.unsqueeze(1)

        batch_size = images.shape[0]
        
        # Normalize
        images = (images - self.image_mean) / self.image_std
        images_flat = rearrange(images, "B N C H W -> (B N) C H W")
        
        # Extract from specific layer
        if layer_idx is not None:
            out = self.model(
                images_flat, 
                interpolate_pos_encoding=True,
                output_hidden_states=True
            )
            features = out.hidden_states[layer_idx]
        else:
            out = self.model(images_flat, interpolate_pos_encoding=True)
            features = out.last_hidden_state
        
        return features

    def freeze_encoder(self, freeze: bool = True) -> None:
        """Freeze/unfreeze the encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = not freeze

    def detokenize(self, *args, **kwargs):
        """Not implemented for DINO tokenizer (encoder-only)."""
        raise NotImplementedError(
            "DINO is an encoder-only model and does not support detokenization. "
            "Use a decoder model for reconstruction tasks."
        )


class DINOMultiViewTokenizer(DINOSingleImageTokenizer):
    """
    Specialized tokenizer for multi-view inputs.
    Ensures input always has view dimension.
    """
    
    def forward(
        self,
        images: torch.FloatTensor,
        return_global_features: bool = False,
        **kwargs
    ) -> torch.FloatTensor:
        """Forward pass with required multi-view input."""
        if images.ndim != 5:
            raise ValueError(
                f"DINOMultiViewTokenizer expects 5D input (B, N, C, H, W), "
                f"got {images.ndim}D input with shape {images.shape}"
            )
        
        return super().forward(images, return_global_features, **kwargs)
