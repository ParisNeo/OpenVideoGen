import logging
import uuid
import time
import os
import shutil
import argparse
import platform
import subprocess
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from pathlib import Path
import torch
# Updated imports for new models and utilities
from diffusers import (
    CogVideoXPipeline,
    StableVideoDiffusionPipeline,
    MochiPipeline,
    AutoencoderKLWan, # Added for Wan model
    WanPipeline        # Added for Wan model
)
from diffusers.utils import export_to_video
import toml
from pathlib import Path

# --- Dependency Management using pipmaster ---
try:
    import pipmaster as pm
except ImportError:
    print("pipmaster not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pipmaster"])
    import pipmaster as pm

# Function to check and install dependencies
def install_dependencies():
    logger.info("Checking and installing dependencies...")

    # OpenCV check (often needed for video/image processing)
    if not pm.is_installed("opencv-python"):
        logger.info("Installing opencv-python...")
        pm.install("opencv-python")

    # PyTorch check (ensure compatible version if needed, Wan requires >= 2.4.0, but let installer handle for now)
    # Note: Wan documentation mentions torch >= 2.4.0. The current installer might get a lower version.
    # If issues arise specifically with Wan, a version constraint might be needed here.
    if not pm.is_installed("torch"):
        logger.info("Installing torch, torchvision, torchaudio...")
        # Using cu121 as a common recent CUDA version, adjust if needed (e.g., cu124)
        # Check https://pytorch.org/ for the correct URL for your CUDA version
        pytorch_index_url = "https://download.pytorch.org/whl/cu121" # Example, adjust CUDA version if necessary
        try:
            pm.install_multiple(["torch", "torchvision", "torchaudio"], index_url=pytorch_index_url)
        except Exception as e:
            logger.warning(f"Failed to install PyTorch with index {pytorch_index_url}: {e}. Trying default index.")
            pm.install_multiple(["torch", "torchvision", "torchaudio"])

    # Core ML/Diffusers packages
    packages_to_install = [
        "diffusers", "transformers", "accelerate",
        "imageio", "imageio-ffmpeg", "sentencepiece", "toml"
    ]
    for package in packages_to_install:
        if not pm.is_installed(package):
            logger.info(f"Installing {package}...")
            pm.install(package)

    # Accelerate version check (example)
    # if not pm.is_version_higher("accelerate", "0.26.0"):
    #     logger.info("Updating accelerate...")
    #     pm.install("accelerate")

    # Wan model specific dependencies (usually included with diffusers, but good practice)
    # No separate packages explicitly listed for WanPipeline/AutoencoderKLWan in docs

    logger.info("Dependency check/installation finished.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("openvideogen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OpenVideoGen")

# --- Run Dependency Installation ---
install_dependencies()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="OpenVideoGen API",
    description="Open source video generation API using various diffusion models.",
    version="0.2.0" # Incremented version for Wan support
)

# --- Configuration Loading ---

# Determine config file search paths based on OS
def get_config_search_paths():
    system = platform.system()
    search_paths = []
    app_name = "openvideogen"

    if system == "Linux":
        search_paths.extend([
            Path(f"/etc/{app_name}/config.toml"),
            Path(f"/usr/local/etc/{app_name}/config.toml"),
            Path.home() / f".config/{app_name}/config.toml",
            Path.cwd() / "config.toml"
        ])
    elif system == "Windows":
        search_paths.extend([
            Path(os.getenv("APPDATA", Path.home() / "AppData/Roaming")) / f"{app_name}/config.toml",
            Path(os.getenv("PROGRAMDATA", "C:/ProgramData")) / f"{app_name}/config.toml",
            Path.cwd() / "config.toml"
        ])
    elif system == "Darwin":  # macOS
        search_paths.extend([
            Path.home() / f"Library/Application Support/{app_name}/config.toml",
            Path(f"/Library/Application Support/{app_name}/config.toml"),
            Path(f"/usr/local/etc/{app_name}/config.toml"),
            Path.cwd() / "config.toml"
        ])
    else: # Fallback for other systems
        search_paths.append(Path.cwd() / "config.toml")

    # Add script directory as a fallback search location
    script_dir = Path(__file__).parent
    if script_dir not in [p.parent for p in search_paths]:
         search_paths.append(script_dir / "config.toml")

    # Remove duplicates while preserving order
    unique_paths = []
    for path in search_paths:
        if path not in unique_paths:
            unique_paths.append(path)

    return unique_paths

# Load configuration with priority: command-line arg > env var > search paths > default
def load_config():
    DEFAULT_CONFIG = {
        "models": {
            "cogvideox_2b": {"name": "THUDM/CogVideoX-2b", "type": "cogvideox"},
            # "cogvideox_5b": {"name": "THUDM/CogVideoX-5b", "type": "cogvideox"}, # Can be large
            "stable_video": {"name": "stabilityai/stable-video-diffusion-img2vid", "type": "stablevideo"},
            "mochi": {"name": "genmo/mochi-1-preview", "type": "mochi"},
            "wan_1_3b": {"name": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "type": "wan"} # Added Wan model
        },
        "settings": {
            "default_model": "wan_1_3b", # Changed default to the new model
            "force_gpu": False,
            "use_gpu": True,
            "dtype": "bfloat16", # Changed default dtype, Wan prefers bfloat16
            "output_folder": "./outputs",
            "model_cache_dir": "./model_cache",
            "port": 8088,
            "host": "0.0.0.0",
            "file_retention_time": 3600 * 24 # 24 hours in seconds
        },
        "generation": {
            "default_guidance_scale": 6.0,
            "default_num_inference_steps": 50,
            # Wan specific defaults (can be overridden by request)
            "wan_default_height": 480,
            "wan_default_width": 832,
            "wan_default_num_frames": 81,
            "wan_default_fps": 15,
            "wan_default_guidance_scale": 5.0
        }
    }

    parser = argparse.ArgumentParser(description="OpenVideoGen API Server")
    parser.add_argument("--config", type=str, help="Path to the config.toml file")
    # Allow overriding host and port via CLI
    parser.add_argument("--host", type=str, help="Host to bind the server to")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    args = parser.parse_args()

    loaded_config = {}

    # 1. Command-line argument
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            logger.info(f"Loading config from command-line argument: {config_path}")
            loaded_config = toml.load(config_path)
        else:
            logger.error(f"Config file specified via --config not found: {config_path}. Exiting.")
            sys.exit(1) # Exit if specified config not found

    # 2. Environment variable
    elif env_config_path := os.getenv("OPENVIDEOGEN_CONFIG"):
        config_path = Path(env_config_path)
        if config_path.exists():
            logger.info(f"Loading config from environment variable OPENVIDEOGEN_CONFIG: {config_path}")
            loaded_config = toml.load(config_path)
        else:
            logger.error(f"Config file specified via OPENVIDEOGEN_CONFIG not found: {config_path}. Exiting.")
            sys.exit(1) # Exit if specified config not found

    # 3. Search paths
    else:
        search_paths = get_config_search_paths()
        for path in search_paths:
            if path.exists():
                logger.info(f"Loading config from search path: {path}")
                loaded_config = toml.load(path)
                break # Stop searching once found

    # Merge loaded config with defaults (loaded values take precedence)
    def merge_dicts(source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                merge_dicts(value, node)
            else:
                destination[key] = value
        return destination

    final_config = merge_dicts(DEFAULT_CONFIG, loaded_config) # Start with default, overlay loaded

     # Override host/port from CLI if provided
    if args.host:
        final_config["settings"]["host"] = args.host
        logger.info(f"Overriding host from CLI: {args.host}")
    if args.port:
        final_config["settings"]["port"] = args.port
        logger.info(f"Overriding port from CLI: {args.port}")


    # Create default config if none was found
    if not loaded_config: # If no file was loaded from any source
        default_path = Path.cwd() / "config.toml"
        try:
            with open(default_path, "w") as f:
                toml.dump(DEFAULT_CONFIG, f)
            logger.info(f"No config file found. Created default config.toml at: {default_path}")
            final_config = DEFAULT_CONFIG # Use the pure default if creating
        except Exception as e:
            logger.error(f"Could not create default config file at {default_path}: {e}")
            logger.warning("Proceeding with default settings in memory.")
            final_config = DEFAULT_CONFIG

    # Validate crucial settings
    if not final_config.get("models"):
        logger.error("Config error: 'models' section is missing. Using default models.")
        final_config["models"] = DEFAULT_CONFIG["models"]
    if "settings" not in final_config:
        logger.error("Config error: 'settings' section is missing. Using default settings.")
        final_config["settings"] = DEFAULT_CONFIG["settings"]
    if "generation" not in final_config:
        logger.error("Config error: 'generation' section is missing. Using default generation settings.")
        final_config["generation"] = DEFAULT_CONFIG["generation"]


    # Ensure output and cache directories exist
    try:
        Path(final_config["settings"]["output_folder"]).mkdir(parents=True, exist_ok=True)
        Path(final_config["settings"]["model_cache_dir"]).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output or cache directories: {e}")
        logger.error("Please check permissions and paths in config. Exiting.")
        sys.exit(1)

    return final_config

config = load_config()

# --- Pydantic Models ---
class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(None, description="Optional negative prompt.")
    model_name: Optional[str] = Field(None, description="Model to use (overrides default). E.g., 'wan_1_3b', 'cogvideox_2b'.")
    # Provide defaults, but allow None to signal using model-specific defaults later
    height: Optional[int] = Field(None, description="Video height. If None, uses model default.")
    width: Optional[int] = Field(None, description="Video width. If None, uses model default.")
    steps: Optional[int] = Field(None, description="Number of inference steps (overrides config default).")
    guidance_scale: Optional[float] = Field(None, description="Guidance scale (overrides config default).")
    seed: int = Field(-1, description="Seed for generation (-1 for random).")
    nb_frames: Optional[int] = Field(None, description="Number of frames to generate. If None, uses model default.")
    fps: Optional[int] = Field(None, description="Frames per second for output video. If None, uses model default.")

class MultiPromptVideoRequest(BaseModel):
    prompts: List[str]
    frames: List[int] # Specifies duration for each prompt
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    # Allow specifying dimensions for multi-prompt
    height: Optional[int] = Field(None, description="Video height. If None, uses model default.")
    width: Optional[int] = Field(None, description="Video width. If None, uses model default.")
    fps: Optional[int] = Field(None, description="Frames per second. If None, uses model default.")
    num_inference_steps: Optional[int] = None # Override num_inference_steps
    guidance_scale: Optional[float] = None # Override guidance_scale
    seed: Optional[int] = None

class JobStatus(BaseModel):
    job_id: str
    status: str # "pending", "processing", "completed", "failed"
    progress: int = 0
    message: Optional[str] = None
    video_url: Optional[str] = None
    created_at: float
    expires_at: float

# In-memory job storage
jobs: Dict[str, JobStatus] = {}

# --- Video Generation Service ---
class VideoGenService:
    def __init__(self):
        self.models_config = config["models"]
        self.settings = config["settings"]
        self.generation_defaults = config["generation"]

        self.default_model = self.settings.get("default_model", list(self.models_config.keys())[0])
        self.use_gpu = self.settings.get("use_gpu", True) and torch.cuda.is_available()
        self.force_gpu = self.settings.get("force_gpu", False)

        # Determine dtype, default to float16 if bfloat16 is not supported or configured
        configured_dtype = self.settings.get("dtype", "bfloat16")
        if configured_dtype == "bfloat16" and (not hasattr(torch, 'bfloat16') or not self.use_gpu):
            logger.warning("BFloat16 configured but not supported or no GPU available. Falling back to Float16.")
            self.dtype = torch.float16
            self.dtype_str = "float16"
        elif configured_dtype == "bfloat16":
            self.dtype = torch.bfloat16
            self.dtype_str = "bfloat16"
        else:
            self.dtype = torch.float16
            self.dtype_str = "float16"
        logger.info(f"Using dtype: {self.dtype_str}")


        self.output_folder = Path(self.settings.get("output_folder", "./outputs"))
        self.model_cache_dir = Path(self.settings.get("model_cache_dir", "./model_cache"))
        self.file_retention_time = self.settings.get("file_retention_time", 3600 * 24)

        self.default_guidance_scale = self.generation_defaults.get("default_guidance_scale", 6.0)
        self.default_num_inference_steps = self.generation_defaults.get("default_num_inference_steps", 50)

        # Wan-specific defaults from config
        self.wan_defaults = {
            "height": self.generation_defaults.get("wan_default_height", 480),
            "width": self.generation_defaults.get("wan_default_width", 832),
            "num_frames": self.generation_defaults.get("wan_default_num_frames", 81),
            "fps": self.generation_defaults.get("wan_default_fps", 15),
            "guidance_scale": self.generation_defaults.get("wan_default_guidance_scale", 5.0),
        }
        # Placeholder defaults for other models (can be refined)
        self.other_defaults = {
            "height": 480,
            "width": 720,
            "num_frames": 49,
            "fps": 8,
            "guidance_scale": self.default_guidance_scale,
         }


        self.pipelines = {}
        self.load_pipelines()

    def load_pipelines(self):
        if self.force_gpu and not torch.cuda.is_available():
            logger.error("force_gpu is set to True, but no GPU is available. Check CUDA setup.")
            raise RuntimeError("force_gpu is set to True, but no GPU is available.")

        device = "cuda" if self.use_gpu else "cpu"
        logger.info(f"Attempting to load models to device: {device}")

        for model_key, model_info in self.models_config.items():
            model_name = model_info["name"]
            model_type = model_info["type"]
            logger.info(f"Loading model: {model_key} ({model_name}, type: {model_type})...")

            try:
                pipeline = None
                load_kwargs = {
                    "torch_dtype": self.dtype,
                    "cache_dir": self.model_cache_dir
                }

                if model_type == "cogvideox":
                    pipeline = CogVideoXPipeline.from_pretrained(model_name, **load_kwargs)
                elif model_type == "stablevideo":
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(model_name, **load_kwargs)
                elif model_type == "mochi":
                    pipeline = MochiPipeline.from_pretrained(model_name, **load_kwargs)
                    pipeline.enable_vae_tiling() # Mochi specific optimization
                elif model_type == "wan":
                    # Wan requires specific loading: VAE first (float32), then pipeline
                    logger.info(f"Loading Wan VAE (using float32)...")
                    vae = AutoencoderKLWan.from_pretrained(
                        model_name,
                        subfolder="vae",
                        torch_dtype=torch.float32, # Wan VAE requires float32 as per docs/example
                        cache_dir=self.model_cache_dir
                    )
                    logger.info(f"Loading Wan Pipeline (using {self.dtype_str})...")
                    pipeline = WanPipeline.from_pretrained(
                        model_name,
                        vae=vae,
                        **load_kwargs # Uses self.dtype (float16/bfloat16)
                    )
                else:
                    logger.warning(f"Unsupported model type '{model_type}' for model key '{model_key}'. Skipping.")
                    continue

                # Move to device and enable offloading if applicable
                if self.use_gpu:
                    try:
                        pipeline.to(device)
                        pipeline.enable_model_cpu_offload() # Helps manage VRAM
                        logger.info(f"Moved {model_key} to CUDA and enabled CPU offloading.")
                    except AttributeError:
                         logger.warning(f"Model {model_key} does not support enable_model_cpu_offload(). Trying direct .to(device).")
                         pipeline.to(device)
                    except Exception as move_err:
                         logger.error(f"Error moving {model_key} to {device} or enabling offload: {move_err}. Model might not work correctly.")
                         # Optionally skip adding this pipeline if moving failed critically
                         # continue
                elif not self.force_gpu:
                     pipeline.to("cpu") # Ensure it's on CPU if not using GPU
                     logger.info(f"Model {model_key} loaded to CPU.")
                else:
                    # This case should be caught earlier, but as a safeguard:
                    logger.error(f"Configuration error: force_gpu=True but GPU not used for {model_key}.")
                    continue # Don't add the pipeline if it couldn't be placed correctly

                self.pipelines[model_key] = pipeline
                logger.info(f"Successfully loaded model: {model_key}")

            except ImportError as e:
                 logger.error(f"Failed to load model {model_key} due to missing dependency: {e}. Please ensure all required libraries (like diffusers, transformers) are installed.")
            except Exception as e:
                logger.error(f"Failed to load model {model_key} ({model_name}): {e}")
                logger.error(f"Check model name, internet connection, cache permissions ({self.model_cache_dir}), and available VRAM/RAM.")

        if not self.pipelines:
             logger.error("No models were loaded successfully. The API will not be able to generate videos.")
             # Optionally exit if no models loaded: sys.exit(1)

    def _get_model_defaults(self, model_type: str) -> dict:
        """Returns default generation parameters based on model type."""
        if model_type == "wan":
            return self.wan_defaults.copy()
        else:
            # Use general defaults for other known types or as a fallback
            return self.other_defaults.copy()

    def generate_video(self, job_id: str, request: VideoGenerationRequest) -> None:
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 0
        start_time = time.time()

        model_key = request.model_name or self.default_model
        if model_key not in self.pipelines:
            error_msg = f"Model '{model_key}' not found or not loaded successfully."
            logger.error(f"Job {job_id}: {error_msg}")
            jobs[job_id].status = "failed"
            jobs[job_id].message = error_msg
            return

        pipeline = self.pipelines[model_key]
        model_type = self.models_config[model_key]["type"]
        model_defaults = self._get_model_defaults(model_type)

        # Determine final generation parameters, prioritizing request values
        height = request.height if request.height is not None else model_defaults["height"]
        width = request.width if request.width is not None else model_defaults["width"]
        nb_frames = request.nb_frames if request.nb_frames is not None else model_defaults["num_frames"]
        fps = request.fps if request.fps is not None else model_defaults["fps"]
        guidance_scale = request.guidance_scale if request.guidance_scale is not None else model_defaults["guidance_scale"]
        steps = request.steps if request.steps is not None else self.default_num_inference_steps
        negative_prompt = request.negative_prompt # Can be None

        # Prepare generator for seeding
        generator = None
        device = "cuda" if self.use_gpu else "cpu"
        if request.seed != -1:
            generator = torch.Generator(device=device).manual_seed(request.seed)
            logger.info(f"Job {job_id}: Using seed {request.seed}")
        else:
             logger.info(f"Job {job_id}: Using random seed")


        gen_params = {
            "prompt": request.prompt,
            "negative_prompt": negative_prompt,
            "num_frames": nb_frames,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "generator": generator,
        }

        logger.info(f"Job {job_id}: Starting generation with {model_key} ({model_type}).")
        logger.info(f"Job {job_id}: Params: { {k: v for k, v in gen_params.items() if k != 'generator'} }") # Log params except generator object


        try:
            # Define the callback function for progress tracking
            def step_callback(step: int, timestep: int, latents: torch.Tensor):
                 # Note: Diffusers callback signature can vary slightly. This is a common one.
                 # Adjust if pipeline expects different args.
                 total_steps = gen_params["num_inference_steps"]
                 progress = int(((step + 1) / total_steps) * 95) # Cap progress from steps at 95%
                 jobs[job_id].progress = min(progress, 95)
                 # logger.debug(f"Job {job_id} progress: {jobs[job_id].progress}% at step {step + 1}/{total_steps}") # Too verbose for INFO

            # Add callback if the pipeline supports it (check documentation or wrap call)
            # Not all pipelines guarantee 'callback_steps' or 'callback' argument.
            # We'll add it, but wrap the generation call in case it's not supported by a specific pipeline.
            gen_params_with_callback = gen_params.copy()
            # Using callback_steps=1 means call after each step
            if hasattr(pipeline, '__call__') and 'callback_steps' in pipeline.__call__.__code__.co_varnames:
                 gen_params_with_callback["callback_steps"] = 1
                 gen_params_with_callback["callback"] = step_callback
                 logger.info(f"Job {job_id}: Callback enabled.")
            else:
                 logger.warning(f"Job {job_id}: Pipeline {model_key} might not support progress callback.")


            # --- Perform Inference ---
            video_frames = None
            autocast_context = torch.autocast(
                 device_type=device.split(":")[0], # 'cuda' or 'cpu'
                 dtype=self.dtype,
                 enabled=self.use_gpu # Only use autocast on GPU
                 )

            with autocast_context:
                if hasattr(pipeline, '__call__') and 'callback' in pipeline.__call__.__code__.co_varnames:
                     output = pipeline(**gen_params_with_callback)
                else:
                    # Fallback if callback args aren't directly supported in __call__
                     output = pipeline(**gen_params) # Generate without callback args

                # Extract frames - structure might vary slightly (e.g., .frames vs .video)
                if hasattr(output, "frames"):
                    video_frames = output.frames[0] if isinstance(output.frames, list) else output.frames
                elif hasattr(output, "video"): # Some pipelines might use 'video'
                     video_frames = output.video[0] if isinstance(output.video, list) else output.video
                else:
                    raise ValueError("Could not find 'frames' or 'video' in pipeline output.")


            if video_frames is None:
                 raise RuntimeError("Video frame generation resulted in None.")

            jobs[job_id].progress = 95 # Mark generation as done

            # --- Export Video ---
            output_filename = self.output_folder / f"video_{job_id}.mp4"
            logger.info(f"Job {job_id}: Exporting video to {output_filename} with FPS={fps}...")
            export_to_video(video_frames, str(output_filename), fps=fps)
            jobs[job_id].progress = 100 # Mark export as done

            elapsed_time = time.time() - start_time
            logger.info(f"Job {job_id}: Video generated successfully in {elapsed_time:.2f}s.")
            jobs[job_id].status = "completed"
            jobs[job_id].video_url = f"/download/{job_id}"
            jobs[job_id].message = "Video generated successfully."

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.exception(f"Job {job_id}: Video generation failed after {elapsed_time:.2f}s: {str(e)}")
            jobs[job_id].status = "failed"
            jobs[job_id].message = f"Video generation failed: {str(e)}"
            jobs[job_id].progress = 0 # Reset progress on failure

    def generate_video_by_frames(self, job_id: str, request: MultiPromptVideoRequest) -> None:
        """
        Handles multi-prompt requests by essentially concatenating prompts
        and generating one video. A more sophisticated implementation might involve
        interpolation or generating segments, but this provides basic functionality.
        """
        if not request.prompts or not request.frames or len(request.prompts) != len(request.frames):
            jobs[job_id].status = "failed"
            jobs[job_id].message = "Prompts and frames lists must exist and have the same length."
            logger.error(f"Job {job_id}: Invalid multi-prompt request structure.")
            return

        # Combine prompts (simple concatenation for now)
        # Consider adding separators or context if needed by models
        combined_prompt = " ".join(request.prompts)
        total_frames = sum(request.frames) # Total length

        # Determine model and defaults
        model_key = request.model_name or self.default_model
        model_type = self.models_config.get(model_key, {}).get("type")
        if not model_type:
             jobs[job_id].status = "failed"
             jobs[job_id].message = f"Model '{model_key}' configuration not found."
             return
        model_defaults = self._get_model_defaults(model_type)


        # Map MultiPromptVideoRequest to VideoGenerationRequest
        # Use provided values or fall back to model defaults
        video_request = VideoGenerationRequest(
            prompt=combined_prompt,
            negative_prompt=request.negative_prompt,
            model_name=model_key,
            height=request.height if request.height is not None else model_defaults["height"],
            width=request.width if request.width is not None else model_defaults["width"],
            steps=request.num_inference_steps, # Pass through if provided
            guidance_scale=request.guidance_scale, # Pass through if provided
            seed=request.seed if request.seed is not None else -1,
            nb_frames=total_frames, # Use calculated total frames
            fps=request.fps if request.fps is not None else model_defaults["fps"]
        )

        # Delegate to the main generation function
        self.generate_video(job_id, video_request)

    def cleanup_expired_files(self):
        """Remove expired files and job records."""
        current_time = time.time()
        expired_job_ids = []
        # Iterate over a copy of keys to allow deletion during iteration
        for job_id in list(jobs.keys()):
            job = jobs.get(job_id) # Check if job still exists (might be deleted concurrently)
            if job and current_time > job.expires_at:
                 expired_job_ids.append(job_id)
                 video_path = self.output_folder / f"video_{job_id}.mp4"
                 if video_path.exists():
                     try:
                         video_path.unlink()
                         logger.info(f"Deleted expired file for job {job_id}: {video_path}")
                     except OSError as e: # Catch file system errors
                         logger.error(f"Failed to delete expired file {video_path} for job {job_id}: {e}")
                 elif job.status == "completed":
                     # Log if a completed job's file is missing upon expiry
                     logger.warning(f"Expired completed job {job_id} video file not found at {video_path}.")


        if expired_job_ids:
             logger.info(f"Cleaning up {len(expired_job_ids)} expired job(s)...")
             for job_id in expired_job_ids:
                 if job_id in jobs:
                     del jobs[job_id]
                     logger.info(f"Removed expired job record: {job_id}")
        # else:
             # logger.debug("No expired jobs found during cleanup.") # Optional: log if nothing to clean

# --- Initialize Service ---
try:
    service = VideoGenService()
except Exception as service_init_error:
     logger.exception(f"Failed to initialize VideoGenService: {service_init_error}")
     logger.error("API cannot start. Please check logs and configuration.")
     sys.exit(1)

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Check service status, configuration, and loaded models."""
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    if gpu_available:
        try:
            gpu_info = {
                "count": torch.cuda.device_count(),
                "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            }
        except Exception as e:
            gpu_info = {"error": f"Could not query GPU details: {e}"}

    return {
        "status": "healthy" if service.pipelines else "degraded",
        "message": "Service running." if service.pipelines else "Service running, but no models loaded successfully.",
        "default_model": service.default_model,
        "loaded_models": list(service.pipelines.keys()),
        "available_models_in_config": list(service.models_config.keys()),
        "settings": {
            "use_gpu": service.use_gpu,
            "force_gpu": service.force_gpu,
            "dtype": service.dtype_str,
            "output_folder": str(service.output_folder),
            "model_cache_dir": str(service.model_cache_dir),
            "file_retention_time_seconds": service.file_retention_time,
        },
        "gpu_details": {
             "available": gpu_available,
             **gpu_info
        }
    }

@app.get("/models")
async def get_models():
    """Get models available according to the configuration file."""
    model_details = []
    for key, info in service.models_config.items():
         is_loaded = key in service.pipelines
         model_details.append({
              "id": key,
              "name": info.get("name", "N/A"),
              "type": info.get("type", "N/A"),
              "loaded": is_loaded,
              "status": "Loaded" if is_loaded else ("Configured" if key in service.models_config else "Unknown")
         })

    return {"models": model_details}


@app.post("/submit", status_code=202) # Use 202 Accepted for async tasks
async def submit_job(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """Submit a video generation job."""
    if not service.pipelines:
         raise HTTPException(status_code=503, detail="Service Unavailable: No models loaded.")

    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + service.file_retention_time

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=created_at,
        expires_at=expires_at,
        progress=0
    )

    logger.info(f"Received job submission {job_id} for model '{request.model_name or service.default_model}'.")
    background_tasks.add_task(service.generate_video, job_id, request)
    # Add cleanup task less frequently, maybe triggered elsewhere or periodically
    # background_tasks.add_task(service.cleanup_expired_files) # Running cleanup on every request might be too much

    return {"job_id": job_id, "message": "Job submitted successfully."}

@app.post("/submit_multi", status_code=202)
async def submit_multi_job(request: MultiPromptVideoRequest, background_tasks: BackgroundTasks):
    """Submit a multi-prompt video generation job."""
    if not service.pipelines:
         raise HTTPException(status_code=503, detail="Service Unavailable: No models loaded.")

    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + service.file_retention_time

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=created_at,
        expires_at=expires_at,
        progress=0
    )

    logger.info(f"Received multi-prompt job submission {job_id} for model '{request.model_name or service.default_model}'.")
    background_tasks.add_task(service.generate_video_by_frames, job_id, request)
    # background_tasks.add_task(service.cleanup_expired_files)

    return {"job_id": job_id, "message": "Multi-prompt job submitted successfully."}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status and progress of a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")

    # Periodically run cleanup when status is checked (simple background cleanup trigger)
    # Consider a more robust periodic task runner for production
    if time.time() % 60 < 5: # Run roughly every minute when status is checked
         logger.debug("Running periodic cleanup via status check...")
         service.cleanup_expired_files() # Run synchronously for simplicity here

    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the generated video for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")

    job = jobs[job_id]
    if job.status == "pending" or job.status == "processing":
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is still {job.status}. Please wait.")
    if job.status == "failed":
         raise HTTPException(status_code=400, detail=f"Job '{job_id}' failed: {job.message}")
    if job.status != "completed" or not job.video_url:
        # This case should ideally be covered by the above, but as a fallback:
        raise HTTPException(status_code=400, detail="Video not ready or generation failed.")


    video_path = service.output_folder / f"video_{job_id}.mp4"
    if not video_path.exists():
        logger.error(f"Completed job {job_id} video file not found at expected path: {video_path}")
        raise HTTPException(status_code=404, detail="Video file not found on server. It might have expired or been cleaned up.")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"openvideogen_{job_id}.mp4" # More descriptive filename
    )


# --- Static Files and Web UI ---
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    logger.warning(f"Static directory not found at {static_dir}. Web UI will not be available.")
else:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Serving static files from: {static_dir}")

@app.get("/", response_class=HTMLResponse, include_in_schema=False) # Exclude from OpenAPI docs
@app.get("/webui", response_class=HTMLResponse)
async def serve_webui():
    """Serves the main Web UI HTML file."""
    webui_html_path = static_dir / "webui.html"
    if not webui_html_path.exists():
         logger.error(f"Web UI file not found: {webui_html_path}")
         raise HTTPException(status_code=404, detail="Web UI file (webui.html) not found in static directory.")
    try:
        with open(webui_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except Exception as e:
         logger.exception(f"Error reading Web UI file: {e}")
         raise HTTPException(status_code=500, detail="Could not load Web UI.")

# --- Application Startup ---
if __name__ == "__main__":
    import uvicorn

    # Ensure service is initialized before starting server
    if not service or not hasattr(service, 'settings'):
         logger.critical("VideoGenService failed to initialize properly. Cannot start Uvicorn.")
         sys.exit(1)

    host = service.settings.get("host", "0.0.0.0")
    port = service.settings.get("port", 8088)

    logger.info(f"Starting OpenVideoGen API server on {host}:{port}")
    logger.info(f"Default model: {service.default_model}")
    logger.info(f"Loaded models: {list(service.pipelines.keys())}")
    logger.info(f"GPU Usage Enabled: {service.use_gpu}")
    logger.info(f"Using dtype: {service.dtype_str}")


    # Schedule periodic cleanup (alternative to triggering via status checks)
    # This requires an async scheduler like `apscheduler` or running cleanup in a separate thread/process
    # Simple example: Run cleanup in background task on startup (runs only once)
    # More robust solution needed for continuous cleanup without external triggers.
    # background_scheduler = BackgroundTasks()
    # background_scheduler.add_task(schedule_periodic_cleanup, service.cleanup_expired_files, service.file_retention_time / 2) # Example: clean every half retention time

    uvicorn.run(app, host=host, port=port)
