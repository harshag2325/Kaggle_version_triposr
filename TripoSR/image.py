# image.py (optimized version)
from flask import Flask, render_template, request, jsonify
from rembg import remove
from PIL import Image
import os
import uuid
import subprocess
import sys
import base64
import threading
import traceback
from pathlib import Path
from typing import Tuple, Optional
from functools import lru_cache
from contextlib import contextmanager

from diffusers import StableDiffusionPipeline
import torch


# Configuration
class Config:
    """Centralized configuration"""
    BASE_DIR = Path(__file__).parent.absolute()
    STATIC_DIR = BASE_DIR / "static"
    UPLOAD_DIR = STATIC_DIR / "uploads"
    TRIPOSR_OUT_DIR = STATIC_DIR / "triposr"

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # Model settings
    LOCAL_MODEL_ID = "Manojb/stable-diffusion-2-1-base"
    DEFAULT_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 7.0
    MIN_INFERENCE_STEPS = 30
    MAX_INFERENCE_STEPS = 50
    MIN_GUIDANCE_SCALE = 1.0
    MAX_GUIDANCE_SCALE = 20.0

    # Canvas constraints
    MAX_CANVAS_WIDTH = 4096
    MAX_CANVAS_HEIGHT = 4096
    DEFAULT_CANVAS_WIDTH = 900
    DEFAULT_CANVAS_HEIGHT = 550


# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE

# Create directories
Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Config.TRIPOSR_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Global model management
_local_pipe = None
_pipe_lock = threading.Lock()


class ModelManager:
    """Singleton for managing the SD pipeline"""

    @staticmethod
    def get_pipeline():
        """Lazily load and return the StableDiffusion pipeline"""
        global _local_pipe

        if _local_pipe is not None:
            return _local_pipe

        with _pipe_lock:
            # Double-check pattern
            if _local_pipe is not None:
                return _local_pipe

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[model] Loading {Config.LOCAL_MODEL_ID} on {device}...")

                pipe = StableDiffusionPipeline.from_pretrained(
                    Config.LOCAL_MODEL_ID,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    safety_checker=None
                )

                pipe.to(device)

                # Enable optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()

                if device == "cuda" and hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass

                _local_pipe = pipe
                print(f"[model] Model loaded successfully on {device}")
                return _local_pipe

            except Exception as e:
                print(f"[model] Failed to load: {e}")
                traceback.print_exc()
                raise


class ImageValidator:
    """Validates uploaded images"""

    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @staticmethod
    def validate_image(file) -> Tuple[bool, Optional[str]]:
        """Validate image file"""
        if not file:
            return False, "No file provided"

        if not ImageValidator.allowed_file(file.filename):
            return False, "Unsupported file format"

        try:
            # Try to open as image to verify it's valid
            Image.open(file.stream)
            file.stream.seek(0)  # Reset stream
            return True, None
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"


class ImageProcessor:
    """Handles image processing operations"""

    @staticmethod
    def remove_background(image_path: Path) -> Image.Image:
        """Remove background from image"""
        img = Image.open(image_path).convert("RGBA")
        return remove(img)

    @staticmethod
    def save_image(image: Image.Image, prefix: str = "image") -> Tuple[Path, str]:
        """Save image and return path and URL"""
        uid = uuid.uuid4().hex
        filename = f"{uid}_{prefix}.png"
        file_path = Config.UPLOAD_DIR / filename

        image.save(file_path, "PNG")
        url = f"/static/uploads/{filename}"
        return file_path, url

    @staticmethod
    def blend_images(bg_path: Path, fg_path: Path, x: int, y: int,
                     w: int, h: int, canvas_w: int, canvas_h: int) -> Path:
        """Blend foreground onto background with positioning"""
        bg = Image.open(bg_path).convert("RGBA")
        fg = Image.open(fg_path).convert("RGBA")

        # Create canvas
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        # Scale and center background
        scale = min(canvas_w / bg.width, canvas_h / bg.height)
        bg_width = int(bg.width * scale)
        bg_height = int(bg.height * scale)
        bg_resized = bg.resize((bg_width, bg_height), Image.LANCZOS)

        bg_x = (canvas_w - bg_width) // 2
        bg_y = (canvas_h - bg_height) // 2
        canvas.paste(bg_resized, (bg_x, bg_y), bg_resized)

        # Resize and position foreground
        fg_resized = fg.resize((w, h), Image.LANCZOS)

        # Clamp position to canvas bounds
        x = max(0, min(x, canvas_w - w))
        y = max(0, min(y, canvas_h - h))

        canvas.paste(fg_resized, (x, y), fg_resized)

        # Save result
        out_path = Config.UPLOAD_DIR / f"final_{uuid.uuid4().hex}.png"
        canvas.convert("RGB").save(out_path, "PNG")

        return out_path


class TripoSRRunner:
    """Handles TripoSR 3D generation"""

    @staticmethod
    def run(image_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Run TripoSR on image
        Returns: (model_url, render_url, texture_url)
        """
        run_id = uuid.uuid4().hex[:8]
        out_dir = Config.TRIPOSR_OUT_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "run.py",
            str(image_path),
            "--output-dir",
            str(out_dir),
        ]

        try:
            subprocess.run(cmd, cwd=Config.BASE_DIR, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            print("[TripoSR] Process timeout")
            return None, None, None
        except Exception as e:
            print(f"[TripoSR] Error: {e}")
            traceback.print_exc()
            return None, None, None

        # Find output files
        model_path = None
        render_path = None
        texture_path = None

        for file_path in out_dir.rglob("*"):
            if not file_path.is_file():
                continue

            name_lower = file_path.name.lower()

            if model_path is None and file_path.suffix.lower() in {'.obj', '.ply', '.glb', '.gltf'}:
                model_path = file_path

            if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                if 'input' in name_lower and render_path is None:
                    render_path = file_path
                elif any(k in name_lower for k in ['tex', 'texture', 'albedo', 'baked']) and texture_path is None:
                    texture_path = file_path
                elif render_path is None:
                    render_path = file_path

        # Fallback texture to render
        if texture_path is None:
            texture_path = render_path

        def to_url(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            rel_path = path.relative_to(Config.STATIC_DIR).as_posix()
            return f"/static/{rel_path}"

        return to_url(model_path), to_url(render_path), to_url(texture_path)


# Utility functions
def url_to_path(url: str) -> Path:
    """Convert URL to file system path"""
    if url.startswith("/static/"):
        rel = url[len("/static/"):]
    else:
        rel = url.lstrip("/")
    return Config.STATIC_DIR / rel


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload background and foreground images"""
    img1 = request.files.get("image1")  # background
    img2 = request.files.get("image2")  # foreground

    if not img1 or not img2:
        return jsonify({"error": "Missing files"}), 400

    # Validate images
    valid1, error1 = ImageValidator.validate_image(img1)
    valid2, error2 = ImageValidator.validate_image(img2)

    if not valid1:
        return jsonify({"error": f"Background image: {error1}"}), 400
    if not valid2:
        return jsonify({"error": f"Foreground image: {error2}"}), 400

    try:
        # Process background
        bg_img = Image.open(img1.stream).convert("RGBA")
        bg_path, bg_url = ImageProcessor.save_image(bg_img, "bg")

        # Process foreground - save raw and remove background
        fg_raw = Image.open(img2.stream).convert("RGBA")
        fg_raw_path, _ = ImageProcessor.save_image(fg_raw, "raw")

        fg_nobg = ImageProcessor.remove_background(fg_raw_path)
        _, fg_url = ImageProcessor.save_image(fg_nobg, "nobg")

        return jsonify({"image1": bg_url, "image2_nobg": fg_url})

    except Exception as e:
        print(f"[upload] Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/prompt_fg", methods=["POST"])
def prompt_fg():
    """Generate foreground from text prompt using local SD model"""
    prompt = request.form.get("prompt", "").strip()
    keep_shadows = request.form.get("keep_shadows", "0")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    # Parse parameters with validation
    try:
        steps = int(request.form.get("steps", Config.DEFAULT_INFERENCE_STEPS))
        steps = clamp(steps, Config.MIN_INFERENCE_STEPS, Config.MAX_INFERENCE_STEPS)

        guidance = float(request.form.get("guidance_scale", Config.DEFAULT_GUIDANCE_SCALE))
        guidance = clamp(guidance, Config.MIN_GUIDANCE_SCALE, Config.MAX_GUIDANCE_SCALE)
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    # Build prompt
    prompt_template = prompt
    if keep_shadows == "0":
        prompt_template += ", isolated object, plain background, minimal shadows, studio lighting"

    # Load model
    try:
        pipe = ModelManager.get_pipeline()
    except Exception as e:
        return jsonify({"error": f"Model initialization failed: {str(e)}"}), 500

    # Generate image
    try:
        generation = pipe(
            prompt_template,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        image = generation.images[0]
    except Exception as e:
        print(f"[generation] Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

    # Save and process
    try:
        raw_path, _ = ImageProcessor.save_image(image, "raw_local")
        nobg_img = ImageProcessor.remove_background(raw_path)
        _, fg_url = ImageProcessor.save_image(nobg_img, "nobg_local")

        return jsonify({"fg": fg_url})

    except Exception as e:
        print(f"[prompt_fg] Processing error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Post-processing failed: {str(e)}"}), 500


@app.route("/blend", methods=["POST"])
def blend():
    """Blend foreground onto background"""
    bg_url = request.form.get("bg")
    fg_url = request.form.get("fg")

    if not bg_url or not fg_url:
        return jsonify({"error": "Missing image URLs"}), 400

    try:
        # Parse parameters
        x = int(float(request.form.get("x", 0)))
        y = int(float(request.form.get("y", 0)))
        w = int(float(request.form.get("w", 100)))
        h = int(float(request.form.get("h", 100)))

        canvas_w = int(float(request.form.get("canvas_w", Config.DEFAULT_CANVAS_WIDTH)))
        canvas_h = int(float(request.form.get("canvas_h", Config.DEFAULT_CANVAS_HEIGHT)))

        # Validate canvas size
        canvas_w = clamp(canvas_w, 1, Config.MAX_CANVAS_WIDTH)
        canvas_h = clamp(canvas_h, 1, Config.MAX_CANVAS_HEIGHT)

    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    # Convert URLs to paths
    bg_path = url_to_path(bg_url)
    fg_path = url_to_path(fg_url)

    if not bg_path.exists() or not fg_path.exists():
        return jsonify({"error": "Image files not found"}), 404

    try:
        out_path = ImageProcessor.blend_images(
            bg_path, fg_path, x, y, w, h, canvas_w, canvas_h
        )

        final_url = f"/static/uploads/{out_path.name}"
        return jsonify({"final": final_url})

    except Exception as e:
        print(f"[blend] Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Blending failed: {str(e)}"}), 500


@app.route("/triposr", methods=["POST"])
def triposr_route():
    """Generate 3D model from foreground image"""
    fg_url = request.form.get("fg")

    if not fg_url:
        return jsonify({"error": "Missing foreground URL"}), 400

    fg_path = url_to_path(fg_url)

    if not fg_path.exists():
        return jsonify({"error": "Foreground image not found"}), 404

    try:
        model_url, render_url, texture_url = TripoSRRunner.run(fg_path)

        if not model_url and not render_url:
            return jsonify({"error": "TripoSR failed to generate output"}), 500

        return jsonify({
            "model": model_url,
            "render": render_url,
            "texture": texture_url
        })

    except Exception as e:
        print(f"[triposr] Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"3D generation failed: {str(e)}"}), 500


@app.route("/upload_3d_view", methods=["POST"])
def upload_3d_view():
    """Upload a captured 3D view as new foreground"""
    file = request.files.get("image_data")

    try:
        if file:
            binary = file.read()
        else:
            data_url = request.form.get("image_data")
            if not data_url:
                return jsonify({"error": "Missing image data"}), 400

            # Parse data URL
            if "," in data_url:
                _, b64 = data_url.split(",", 1)
            else:
                b64 = data_url

            binary = base64.b64decode(b64)

        # Save image
        uid = uuid.uuid4().hex
        out_path = Config.UPLOAD_DIR / f"{uid}_3dview.png"
        out_path.write_bytes(binary)

        url = f"/static/uploads/{out_path.name}"
        return jsonify({"fg": url})

    except Exception as e:
        print(f"[upload_3d_view] Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({"error": "File size exceeds maximum limit"}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    print(f"[500] Internal error: {error}")
    traceback.print_exc()
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    app.run(debug=True)