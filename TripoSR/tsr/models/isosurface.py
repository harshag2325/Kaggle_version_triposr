from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .marching_cubes_2 import mcubes_gpu
from .marching_cubes import mcubes_cpu


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int, use_gpu: bool = True) -> None:
        super().__init__()
        self.resolution = resolution
        self.use_gpu = use_gpu
        self.mc_func_gpu: Callable = mcubes_gpu
        self.mc_func_cpu: Callable = mcubes_cpu
        
        # Pre-compute and cache grid vertices
        self._grid_vertices: Optional[torch.FloatTensor] = None
        self._normalization_factor = 1.0 / (resolution - 1.0)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # Generate grid coordinates efficiently
            coords = torch.linspace(
                *self.points_range, 
                self.resolution, 
                dtype=torch.float32
            )
            
            # Use stack instead of cat for better memory efficiency
            grid = torch.stack(
                torch.meshgrid(coords, coords, coords, indexing="ij"), 
                dim=-1
            )
            
            self._grid_vertices = grid.reshape(-1, 3)
        
        return self._grid_vertices

    def _run_marching_cubes_gpu(
        self, 
        level: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run marching cubes on GPU."""
        return self.mc_func_gpu(level, 0.0)

    def _run_marching_cubes_cpu(
        self, 
        level: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run marching cubes on CPU as fallback."""
        # Convert to numpy
        level_np = level.detach().cpu().numpy()
        v_pos_np, t_pos_idx_np = self.mc_func_cpu(level_np, np.float32(0.0))
        
        # Convert back to torch tensors on original device
        v_pos = torch.from_numpy(v_pos_np).to(
            device=level.device,
            dtype=level.dtype
        )
        t_pos_idx = torch.from_numpy(t_pos_idx_np).to(
            device=level.device,
            dtype=torch.long
        )
        
        return v_pos, t_pos_idx

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Extract isosurface using marching cubes.
        
        Args:
            level: Scalar field values, shape (resolution^3,)
            
        Returns:
            v_pos: Vertex positions normalized to [0, 1], shape (N, 3)
            t_pos_idx: Triangle indices, shape (M, 3)
        """
        # Reshape to 3D grid and negate (single operation)
        level = -level.view(self.resolution, self.resolution, self.resolution)

        # Try GPU first, fallback to CPU if needed
        if self.use_gpu and level.is_cuda:
            try:
                v_pos, t_pos_idx = self._run_marching_cubes_gpu(level)
            except Exception as e:
                print(f"Marching cubes GPU failed, using CPU fallback: {e}")
                
                # Reset CUDA state
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                v_pos, t_pos_idx = self._run_marching_cubes_cpu(level)
        else:
            v_pos, t_pos_idx = self._run_marching_cubes_cpu(level)

        # Normalize vertices to [0, 1] range (cached normalization factor)
        v_pos = v_pos * self._normalization_factor

        return v_pos, t_pos_idx

    def set_resolution(self, resolution: int) -> None:
        """Update resolution and invalidate cached grid vertices."""
        if resolution != self.resolution:
            self.resolution = resolution
            self._grid_vertices = None
            self._normalization_factor = 1.0 / (resolution - 1.0)

    def clear_cache(self) -> None:
        """Clear cached grid vertices to free memory."""
        self._grid_vertices = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()