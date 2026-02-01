import importlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import PIL.Image
import rembg
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from omegaconf import DictConfig, OmegaConf
from PIL import Image


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    """Parse and merge structured configuration."""
    return OmegaConf.merge(OmegaConf.structured(fields), cfg)

def find_class(cls_string: str):
    module_string, cls_name = cls_string.rsplit(".", 1)

    # auto-fix relative tsr imports
    if module_string.startswith("tsr."):
        module_string = "TripoSR." + module_string

    module = importlib.import_module(module_string)
    return getattr(module, cls_name)

def get_intrinsic_from_fov(
    fov: float, 
    H: int, 
    W: int, 
    bs: int = -1
) -> torch.Tensor:
    """
    Compute camera intrinsic matrix from field of view.
    
    Args:
        fov: Field of view in radians
        H: Image height
        W: Image width
        bs: Batch size. If > 0, returns batched intrinsics
        
    Returns:
        Intrinsic matrix of shape (3, 3) or (bs, 3, 3)
    """
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    
    # Build intrinsic matrix efficiently
    intrinsic = np.array([
        [focal_length, 0, W / 2.0],
        [0, focal_length, H / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)

    if bs > 0:
        intrinsic = np.tile(intrinsic[None], (bs, 1, 1))

    return torch.from_numpy(intrinsic)


class BaseModule(nn.Module):
    """Base module with structured configuration support."""
    
    @dataclass
    class Config:
        pass

    cfg: Config

    def __init__(
        self, 
        cfg: Optional[Union[dict, DictConfig]] = None, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        raise NotImplementedError


class ImagePreprocessor:
    """Preprocessing utility for images with automatic format conversion and resizing."""
    
    @staticmethod
    def _to_tensor(image: Union[PIL.Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert various image formats to torch tensor."""
        if isinstance(image, torch.Tensor):
            return image
        
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        
        # Convert to float tensor
        if image.dtype == np.uint8:
            return torch.from_numpy(image.astype(np.float32) / 255.0)
        return torch.from_numpy(image)

    def convert_and_resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        size: int,
    ) -> torch.Tensor:
        """Convert image to tensor and resize to square."""
        image = self._to_tensor(image)
        
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        
        # Resize with antialiasing
        image = F.interpolate(
            image.permute(0, 3, 1, 2),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1)
        
        return image if batched else image.squeeze(0)

    def __call__(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        size: int,
    ) -> torch.Tensor:
        """Preprocess image(s) to uniform tensor format."""
        # Handle batched input
        if isinstance(image, (np.ndarray, torch.FloatTensor)) and image.ndim == 4:
            return self.convert_and_resize(image, size)
        
        # Handle list or single image
        if not isinstance(image, list):
            image = [image]
        
        return torch.stack([self.convert_and_resize(im, size) for im in image], dim=0)


def rays_intersect_bbox(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    radius: Union[float, torch.Tensor],
    near: float = 0.0,
    valid_thresh: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ray-bbox intersections for volume rendering.
    
    Args:
        rays_o: Ray origins of shape (..., 3)
        rays_d: Ray directions of shape (..., 3)
        radius: Bounding box radius (scalar or tensor)
        near: Minimum intersection distance
        valid_thresh: Threshold for valid intersections
        
    Returns:
        t_near: Near intersection distances
        t_far: Far intersection distances
        rays_valid: Boolean mask of valid intersections
    """
    input_shape = rays_o.shape[:-1]
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    
    # Avoid division by zero
    rays_d_safe = torch.where(
        rays_d.abs() < 1e-6, 
        torch.full_like(rays_d, 1e-6), 
        rays_d
    )
    
    # Setup bounding box
    if isinstance(radius, (int, float)):
        radius = torch.tensor(
            [[-radius, radius], [-radius, radius], [-radius, radius]],
            dtype=torch.float32,
            device=rays_o.device
        )
    
    # Slightly tighten radius to ensure intersection points are inside
    radius = radius * (1.0 - 1e-3)
    
    # Compute intersections
    interx0 = (radius[:, 1] - rays_o) / rays_d_safe
    interx1 = (radius[:, 0] - rays_o) / rays_d_safe
    
    t_near = torch.minimum(interx0, interx1).amax(dim=-1).clamp_min(near)
    t_far = torch.maximum(interx0, interx1).amin(dim=-1)
    
    # Check valid intersections
    rays_valid = (t_far - t_near) > valid_thresh
    
    # Zero out invalid rays
    t_near = torch.where(rays_valid, t_near, torch.zeros_like(t_near))
    t_far = torch.where(rays_valid, t_far, torch.zeros_like(t_far))
    
    # Reshape to original input shape
    return (
        t_near.view(*input_shape, 1),
        t_far.view(*input_shape, 1),
        rays_valid.view(*input_shape)
    )


def chunk_batch(
    func: Callable, 
    chunk_size: int, 
    *args, 
    **kwargs
) -> Any:
    """
    Execute function on tensor inputs in chunks to save memory.
    
    Args:
        func: Function to execute
        chunk_size: Size of each chunk. If <= 0, no chunking
        *args: Positional arguments (tensors will be chunked)
        **kwargs: Keyword arguments (tensors will be chunked)
        
    Returns:
        Merged output from all chunks
    """
    if chunk_size <= 0:
        return func(*args, **kwargs)
    
    # Find batch size
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    
    assert B is not None, "No tensor found in args/kwargs, cannot determine batch size."
    
    # Process chunks
    out = defaultdict(list)
    out_type = None
    
    for i in range(0, max(1, B), chunk_size):
        # Chunk tensors, keep non-tensors as-is
        chunk_args = [
            arg[i:i + chunk_size] if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        chunk_kwargs = {
            k: arg[i:i + chunk_size] if isinstance(arg, torch.Tensor) else arg
            for k, arg in kwargs.items()
        }
        
        out_chunk = func(*chunk_args, **chunk_kwargs)
        
        if out_chunk is None:
            continue
        
        out_type = type(out_chunk)
        
        # Normalize output to dict format
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, (tuple, list)):
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif not isinstance(out_chunk, dict):
            raise TypeError(
                f"Return value must be torch.Tensor, list, tuple, or dict. Got {type(out_chunk)}"
            )
        
        # Collect chunks
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)
    
    if out_type is None:
        return None
    
    # Merge chunks
    out_merged = {}
    for k, v in out.items():
        if all(vv is None for vv in v):
            out_merged[k] = None
        elif all(isinstance(vv, torch.Tensor) for vv in v):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value: "
                f"{[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )
    
    # Return in original format
    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(len(out_merged))])
    return out_merged


ValidScale = Union[Tuple[float, float], torch.FloatTensor]


def scale_tensor(
    dat: torch.FloatTensor, 
    inp_scale: ValidScale, 
    tgt_scale: ValidScale
) -> torch.FloatTensor:
    """
    Linearly scale tensor from input range to target range.
    
    Args:
        dat: Input tensor
        inp_scale: Input range (min, max)
        tgt_scale: Target range (min, max)
        
    Returns:
        Scaled tensor
    """
    inp_scale = inp_scale or (0, 1)
    tgt_scale = tgt_scale or (0, 1)
    
    if isinstance(tgt_scale, torch.FloatTensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    
    # Normalize to [0, 1] then scale to target
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    return dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]


# Cache for activation functions
_ACTIVATION_CACHE = {}


def get_activation(name: Optional[str]) -> Callable:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name (e.g., 'relu', 'sigmoid', 'exp')
        
    Returns:
        Activation function
    """
    if name is None or name.lower() == "none":
        return lambda x: x
    
    name = name.lower()
    
    # Check cache first
    if name in _ACTIVATION_CACHE:
        return _ACTIVATION_CACHE[name]
    
    # Special activations
    activation_map = {
        "exp": torch.exp,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "softplus": F.softplus,
    }
    
    if name in activation_map:
        fn = activation_map[name]
    else:
        # Try to get from torch.nn.functional
        try:
            fn = getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")
    
    _ACTIVATION_CACHE[name] = fn
    return fn


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
    normalize: bool = True,
) -> torch.FloatTensor:
    """
    Get ray directions for all pixels in camera coordinate.
    
    Args:
        H: Image height
        W: Image width
        focal: Focal length (scalar or (fx, fy))
        principal: Principal point (cx, cy). If None, uses image center
        use_pixel_centers: Whether to use pixel centers (0.5 offset)
        normalize: Whether to normalize ray directions
        
    Returns:
        Ray directions of shape (H, W, 3)
    """
    pixel_center = 0.5 if use_pixel_centers else 0.0

    # Parse focal length and principal point
    if isinstance(focal, (int, float)):
        fx = fy = float(focal)
        cx, cy = W / 2.0, H / 2.0
    else:
        fx, fy = focal
        assert principal is not None, "principal must be provided when focal is a tuple"
        cx, cy = principal

    # Generate pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    # Compute ray directions
    directions = torch.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        -torch.ones_like(i)
    ], dim=-1)

    if normalize:
        directions = F.normalize(directions, dim=-1)

    return directions


def get_rays(
    directions: torch.Tensor,
    c2w: torch.Tensor,
    keepdim: bool = False,
    normalize: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Transform ray directions from camera to world coordinates.
    
    Args:
        directions: Ray directions in camera space
        c2w: Camera-to-world transformation matrix
        keepdim: Whether to keep original spatial dimensions
        normalize: Whether to normalize ray directions
        
    Returns:
        rays_o: Ray origins
        rays_d: Ray directions in world space
    """
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        c2w = c2w[None] if c2w.ndim == 2 else c2w
        assert c2w.ndim == 3
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)
        rays_o = c2w[:, :3, 3].expand_as(rays_d)
        
    elif directions.ndim == 3:  # (H, W, 3)
        if c2w.ndim == 2:
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(-1)
            rays_o = c2w[None, None, :3, 3].expand_as(rays_d)
        else:  # c2w.ndim == 3
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)
            rays_o = c2w[:, None, None, :3, 3].expand_as(rays_d)
            
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)
        rays_o = c2w[:, None, None, :3, 3].expand_as(rays_d)

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
        
    if not keepdim:
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_spherical_cameras(
    n_views: int,
    elevation_deg: float,
    camera_distance: float,
    fovy_deg: float,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays for spherical camera arrangement.
    
    Args:
        n_views: Number of views
        elevation_deg: Camera elevation in degrees
        camera_distance: Distance from origin
        fovy_deg: Vertical field of view in degrees
        height: Image height
        width: Image width
        
    Returns:
        rays_o: Ray origins of shape (n_views, height, width, 3)
        rays_d: Ray directions of shape (n_views, height, width, 3)
    """
    # Generate azimuth angles
    azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
    elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    camera_distances = torch.full_like(elevation_deg, camera_distance)

    # Convert to radians
    elevation = elevation_deg * (math.pi / 180.0)
    azimuth = azimuth_deg * (math.pi / 180.0)

    # Compute camera positions (spherical to cartesian)
    cos_elev = torch.cos(elevation)
    camera_positions = torch.stack([
        camera_distances * cos_elev * torch.cos(azimuth),
        camera_distances * cos_elev * torch.sin(azimuth),
        camera_distances * torch.sin(elevation),
    ], dim=-1)

    # Camera looks at origin, up is +z
    center = torch.zeros_like(camera_positions)
    up = torch.tensor([[0, 0, 1]], dtype=torch.float32).expand(n_views, -1)

    # Compute camera-to-world matrix
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    
    c2w3x4 = torch.cat([
        torch.stack([right, up, -lookat], dim=-1),
        camera_positions.unsqueeze(-1)
    ], dim=-1)
    
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0

    # Compute ray directions
    fovy = fovy_deg * (math.pi / 180.0)
    focal_length = 0.5 * height / torch.tan(0.5 * fovy)
    
    directions_unit = get_ray_directions(H=height, W=width, focal=1.0)
    directions = directions_unit.unsqueeze(0).expand(n_views, -1, -1, -1).clone()
    directions[:, :, :, :2] /= focal_length[:, None, None, None]
    
    return get_rays(directions, c2w, keepdim=True, normalize=True)


def remove_background(
    image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    """
    Remove background from image using rembg.
    
    Args:
        image: Input image
        rembg_session: Rembg session for reuse
        force: Force background removal even if alpha channel exists
        **rembg_kwargs: Additional arguments for rembg
        
    Returns:
        Image with background removed
    """
    # Check if image already has transparency
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    
    if do_remove or force:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    
    return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    """
    Resize foreground object to specified ratio of image size.
    
    Args:
        image: RGBA image with transparency
        ratio: Target ratio of foreground to image size
        
    Returns:
        Resized image with foreground centered
    """
    image = np.array(image)
    assert image.shape[-1] == 4, "Image must have alpha channel"
    
    # Find foreground bounding box
    alpha_mask = image[..., 3] > 0
    y_coords, x_coords = np.where(alpha_mask)
    y1, y2 = y_coords.min(), y_coords.max()
    x1, x2 = x_coords.min(), x_coords.max()
    
    # Crop foreground
    fg = image[y1:y2 + 1, x1:x2 + 1]
    
    # Pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0 = (size - fg.shape[0]) // 2
    pw0 = (size - fg.shape[1]) // 2
    ph1 = size - fg.shape[0] - ph0
    pw1 = size - fg.shape[1] - pw0
    
    new_image = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")
    
    # Add padding based on ratio
    new_size = int(size / ratio)
    ph0 = (new_size - size) // 2
    pw0 = (new_size - size) // 2
    ph1 = new_size - size - ph0
    pw1 = new_size - size - pw0
    
    new_image = np.pad(new_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")
    
    return PIL.Image.fromarray(new_image)
