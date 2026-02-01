from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from ..utils import (
    BaseModule,
    chunk_batch,
    get_activation,
    rays_intersect_bbox,
    scale_tensor,
)


class TriplaneNeRFRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float
        feature_reduction: str = "concat"
        density_activation: str = "trunc_exp"
        density_bias: float = -1.0
        color_activation: str = "sigmoid"
        num_samples_per_ray: int = 128
        randomized: bool = False

    cfg: Config

    def configure(self) -> None:
        assert self.cfg.feature_reduction in ["concat", "mean"]
        self.chunk_size = 0
        
        # Pre-compute and cache activation functions
        self._density_activation = get_activation(self.cfg.density_activation)
        self._color_activation = get_activation(self.cfg.color_activation)
        
        # Pre-compute scale factors for position normalization
        self._scale_min = -self.cfg.radius
        self._scale_max = self.cfg.radius

    def set_chunk_size(self, chunk_size: int) -> None:
        assert chunk_size >= 0, "chunk_size must be a non-negative integer (0 for no chunking)."
        self.chunk_size = chunk_size

    def query_triplane(
        self,
        decoder: torch.nn.Module,
        positions: torch.Tensor,
        triplane: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        input_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)

        # Normalize positions from (-radius, radius) to (-1, 1)
        positions = scale_tensor(positions, (self._scale_min, self._scale_max), (-1, 1))

        def _query_chunk(x: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Create 2D indices for three orthogonal planes (XY, XZ, YZ)
            indices2D = torch.stack(
                [x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]],
                dim=-3,
            )
            
            # Sample from triplane - single grid_sample call
            out = F.grid_sample(
                triplane,  # Already in shape (Np, Cp, Hp, Wp)
                rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
                align_corners=False,
                mode="bilinear",
            )
            
            # Feature reduction
            if self.cfg.feature_reduction == "concat":
                out = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
            else:  # mean
                out = reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")

            return decoder(out)

        # Process with or without chunking
        net_out = (chunk_batch(_query_chunk, self.chunk_size, positions) 
                   if self.chunk_size > 0 
                   else _query_chunk(positions))

        # Apply activations (cached functions)
        net_out["density_act"] = self._density_activation(
            net_out["density"] + self.cfg.density_bias
        )
        net_out["color"] = self._color_activation(net_out["features"])

        # Reshape outputs
        return {k: v.view(*input_shape, -1) for k, v in net_out.items()}

    def _forward(
        self,
        decoder: torch.nn.Module,
        triplane: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> torch.Tensor:
        rays_shape = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        n_rays = rays_o.shape[0]

        # Compute ray-bbox intersections
        t_near, t_far, rays_valid = rays_intersect_bbox(rays_o, rays_d, self.cfg.radius)
        
        # Early exit if no valid rays
        if not rays_valid.any():
            return torch.ones(n_rays, 3, dtype=rays_o.dtype, device=rays_o.device).view(*rays_shape, 3)
        
        t_near, t_far = t_near[rays_valid], t_far[rays_valid]

        # Generate sampling points along rays
        t_vals = torch.linspace(
            0, 1, self.cfg.num_samples_per_ray + 1, device=triplane.device, dtype=rays_o.dtype
        )
        t_mid = (t_vals[:-1] + t_vals[1:]) * 0.5
        z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]

        # Compute 3D sample positions
        xyz = rays_o[rays_valid, None, :] + z_vals[..., None] * rays_d[rays_valid, None, :]

        # Query triplane
        mlp_out = self.query_triplane(decoder, xyz, triplane)

        # Volume rendering
        deltas = t_vals[1:] - t_vals[:-1]
        alpha = 1 - torch.exp(-deltas * mlp_out["density_act"][..., 0])
        
        # Compute transmittance weights
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=-1),
            dim=-1
        )
        weights = alpha * transmittance

        # Composite color and opacity
        comp_rgb_ = (weights[..., None] * mlp_out["color"]).sum(dim=-2)
        opacity_ = weights.sum(dim=-1)

        # Fill in results for all rays
        comp_rgb = torch.ones(n_rays, 3, dtype=comp_rgb_.dtype, device=comp_rgb_.device)
        comp_rgb[rays_valid] = comp_rgb_ + (1 - opacity_[..., None])

        return comp_rgb.view(*rays_shape, 3)

    def forward(
        self,
        decoder: torch.nn.Module,
        triplane: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> torch.Tensor:
        if triplane.ndim == 4:
            return self._forward(decoder, triplane, rays_o, rays_d)
        
        # Batch processing
        return torch.stack(
            [self._forward(decoder, triplane[i], rays_o[i], rays_d[i])
             for i in range(triplane.shape[0])],
            dim=0,
        )

    def train(self, mode: bool = True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()