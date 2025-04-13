#
# File: openvideogen/main.py
# Author: parisneo
# Description: Main FastAPI application file for OpenVideoGen API with on-demand, VRAM-aware model loading.
# Date: 10/04/2025
#
import logging
import uuid
import time
import os
import shutil
import argparse
import platform
import subprocess
import sys
import io
import asyncio
import gc # Garbage Collector
from enum import Enum
from collections import defaultdict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from PIL import Image

# Diffusers imports
from diffusers import (
    CogVideoXPipeline, StableVideoDiffusionPipeline, MochiPipeline,
    AutoencoderKLWan, WanPipeline, HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel, LTXPipeline, AnimateDiffPipeline,
    MotionAdapter, DDIMScheduler
)
from diffusers.utils import export_to_video
import toml

# --- Dependency Management ---
try:
    import pipmaster as pm
except ImportError:
    print("pipmaster not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pipmaster"])
    import pipmaster as pm

def install_dependencies():
    logger.info("Checking and installing dependencies...")
    packages_to_install = [
        "diffusers", "transformers", "accelerate", "imageio",
        "imageio-ffmpeg", "sentencepiece", "toml", "Pillow",
        "python-multipart", "opencv-python"
    ]
    for package in packages_to_install:
        if not pm.is_installed(package):
            logger.info(f"Installing {package}...")
            pm.install(package)
    if not pm.is_installed("torch"):
        logger.info("Installing torch, torchvision, torchaudio...")
        pytorch_index_url = "https://download.pytorch.org/whl/cu121" # Adjust CUDA version if needed
        try:
            pm.install_multiple(["torch", "torchvision", "torchaudio"], index_url=pytorch_index_url)
        except Exception as e:
            logger.warning(f"Failed PyTorch install with index {pytorch_index_url}: {e}. Trying default.")
            pm.install_multiple(["torch", "torchvision", "torchaudio"])
    logger.info("Dependency check/installation finished.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("openvideogen.log"), logging.StreamHandler()]
)
logger = logging.getLogger("OpenVideoGen")

# --- Run Dependency Installation ---
install_dependencies()

# --- Job Status Enum ---
class JobStatusEnum(str, Enum):
    PENDING = "pending"
    WAITING_FOR_RESOURCES = "waiting for resources"
    LOADING_MODEL = "loading model"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="OpenVideoGen API",
    description="Open source video generation API with on-demand, VRAM-aware model loading.",
    version="0.5.0" # Incremented version
)

# --- Configuration Loading ---
# (get_config_search_paths and load_config functions remain the same as in the previous version)
def get_config_search_paths():
    system = platform.system()
    search_paths = []
    app_name = "openvideogen"
    if system == "Linux":
        search_paths.extend([
            Path(f"/etc/{app_name}/config.toml"),
            Path(f"/usr/local/etc/{app_name}/config.toml"),
            Path.home() / f".config/{app_name}/config.toml",
            Path.cwd() / "config.toml" ])
    elif system == "Windows":
        search_paths.extend([
            Path(os.getenv("APPDATA", Path.home() / "AppData/Roaming")) / f"{app_name}/config.toml",
            Path(os.getenv("PROGRAMDATA", "C:/ProgramData")) / f"{app_name}/config.toml",
            Path.cwd() / "config.toml"])
    elif system == "Darwin":
        search_paths.extend([
            Path.home() / f"Library/Application Support/{app_name}/config.toml",
            Path(f"/Library/Application Support/{app_name}/config.toml"),
            Path(f"/usr/local/etc/{app_name}/config.toml"),
            Path.cwd() / "config.toml"])
    else: search_paths.append(Path.cwd() / "config.toml")
    script_dir = Path(__file__).parent
    project_root_dir = script_dir.parent
    if project_root_dir.resolve() not in [p.parent.resolve() for p in search_paths]:
        search_paths.append(project_root_dir / "config.toml")
    if script_dir.resolve() not in [p.parent.resolve() for p in search_paths]:
        search_paths.append(script_dir / "config.toml")
    unique_paths = []
    for path in search_paths:
        abs_path = path.resolve()
        if abs_path not in unique_paths: unique_paths.append(abs_path)
    return unique_paths

def load_config():
    DEFAULT_CONFIG = {
        "models": {"wan_1_3b": {"name": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "type": "wan"}, "stable_video_xt_1_1": {"name": "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "type": "img2vid"}},
        "settings": {"default_t2v_model": "wan_1_3b", "default_i2v_model": "stable_video_xt_1_1", "force_gpu": False, "use_gpu": True, "dtype": "bfloat16", "output_folder": "./outputs", "model_cache_dir": "./model_cache", "port": 8088, "host": "0.0.0.0", "file_retention_time": 86400, "model_inactivity_timeout": 3600, "unload_check_interval": 600},
        "generation": {"default_guidance_scale": 6.0, "default_num_inference_steps": 50, "wan_default_height": 480, "wan_default_width": 832, "wan_default_num_frames": 81, "wan_default_fps": 15, "wan_default_guidance_scale": 5.0, "mochi_default_height": 768, "mochi_default_width": 768, "mochi_default_num_frames": 84, "mochi_default_fps": 30, "hunyuan_default_height": 576, "hunyuan_default_width": 1024, "hunyuan_default_num_frames": 61, "hunyuan_default_fps": 15, "ltx_default_height": 480, "ltx_default_width": 704, "ltx_default_num_frames": 161, "ltx_default_fps": 24, "animatediff_default_height": 512, "animatediff_default_width": 512, "animatediff_default_num_frames": 16, "animatediff_default_fps": 8, "animatediff_guidance_scale": 7.5, "img2vid_default_height": 576, "img2vid_default_width": 1024, "img2vid_default_num_frames": 25, "img2vid_default_fps": 7, "img2vid_motion_bucket_id": 127, "img2vid_noise_aug_strength": 0.02, "img2vid_decode_chunk_size": 8}
    }
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Path to config.toml")
    parser.add_argument("--host", type=str, help="Host to bind")
    parser.add_argument("--port", type=int, help="Port to run on")
    args, unknown = parser.parse_known_args()
    loaded_config = {}; config_path_used = None
    if args.config:
        config_path = Path(args.config);
        if config_path.exists(): loaded_config = toml.load(config_path); config_path_used = config_path; logger.info(f"Loading config from CLI: {config_path}")
        else: logger.error(f"--config file not found: {config_path}. Exiting."); sys.exit(1)
    elif env_config_path := os.getenv("OPENVIDEOGEN_CONFIG"):
        config_path = Path(env_config_path)
        if config_path.exists(): loaded_config = toml.load(config_path); config_path_used = config_path; logger.info(f"Loading config from ENV: {config_path}")
        else: logger.error(f"ENV config file not found: {config_path}. Exiting."); sys.exit(1)
    else:
        search_paths = get_config_search_paths(); logger.info(f"Searching config in: {[str(p) for p in search_paths]}")
        for path in search_paths:
            if path.exists(): loaded_config = toml.load(path); config_path_used = path; logger.info(f"Loading config from search path: {path}"); break
    def merge_dicts(source, destination):
        for key, value in source.items(): destination[key] = merge_dicts(value, destination.setdefault(key, {})) if isinstance(value, dict) else value
        return destination
    final_config = DEFAULT_CONFIG.copy();
    if loaded_config: final_config = merge_dicts(loaded_config, final_config)
    if args.host: final_config["settings"]["host"] = args.host; logger.info(f"Overriding host from CLI: {args.host}")
    if args.port: final_config["settings"]["port"] = args.port; logger.info(f"Overriding port from CLI: {args.port}")
    if not config_path_used:
        default_path = Path.cwd() / "config.toml"
        try:
            if not default_path.exists():
                 with open(default_path, "w") as f: toml.dump(DEFAULT_CONFIG, f)
                 logger.info(f"Created default config.toml at: {default_path}")
            else:
                 logger.info(f"Using existing config.toml in CWD: {default_path}")
                 loaded_config = toml.load(default_path)
                 final_config = merge_dicts(loaded_config, DEFAULT_CONFIG.copy())
            if not final_config.get("models"): final_config = DEFAULT_CONFIG
        except Exception as e: logger.error(f"Could not create/load default config at {default_path}: {e}"); logger.warning("Using built-in defaults."); final_config = DEFAULT_CONFIG
    if not final_config.get("models"): final_config["models"] = DEFAULT_CONFIG["models"]
    if "settings" not in final_config: final_config["settings"] = DEFAULT_CONFIG["settings"]
    if "generation" not in final_config: final_config["generation"] = DEFAULT_CONFIG["generation"]
    try:
        Path(final_config["settings"]["output_folder"]).mkdir(parents=True, exist_ok=True)
        Path(final_config["settings"]["model_cache_dir"]).mkdir(parents=True, exist_ok=True)
    except Exception as e: logger.error(f"Failed to create output/cache dirs: {e}. Exiting."); sys.exit(1)
    return final_config

config = load_config()

# --- Pydantic Models ---
class TextVideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(None)
    model_name: Optional[str] = Field(None)
    height: Optional[int] = Field(None)
    width: Optional[int] = Field(None)
    steps: Optional[int] = Field(None)
    guidance_scale: Optional[float] = Field(None)
    seed: int = Field(-1)
    nb_frames: Optional[int] = Field(None)
    fps: Optional[int] = Field(None)

class ImageVideoGenerationRequest(BaseModel):
    model_name: Optional[str] = Field(None)
    height: Optional[int] = Field(None)
    width: Optional[int] = Field(None)
    fps: Optional[int] = Field(None)
    motion_bucket_id: Optional[int] = Field(None)
    noise_aug_strength: Optional[float] = Field(None)
    seed: int = Field(-1)
    decode_chunk_size: Optional[int] = Field(None)
    num_inference_steps: Optional[int] = Field(None)
    prompt: Optional[str] = Field(None)

class MultiPromptVideoRequest(BaseModel):
    prompts: List[str]
    frames: List[int]
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    height: Optional[int] = Field(None)
    width: Optional[int] = Field(None)
    fps: Optional[int] = Field(None)
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None

class JobStatus(BaseModel):
    job_id: str
    status: JobStatusEnum # Use the Enum
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
        self.generation_defaults_config = config["generation"]

        self.default_t2v_model_key = self.settings.get("default_t2v_model")
        self.default_i2v_model_key = self.settings.get("default_i2v_model")
        self.use_gpu = self.settings.get("use_gpu", True) and torch.cuda.is_available()
        self.force_gpu = self.settings.get("force_gpu", False)
        self.device = "cuda" if self.use_gpu else "cpu"

        self.dtype, self.dtype_str = self._get_torch_dtype()
        logger.info(f"Using default dtype: {self.dtype_str}")

        self.output_folder = Path(self.settings.get("output_folder", "./outputs"))
        self.model_cache_dir = Path(self.settings.get("model_cache_dir", "./model_cache"))
        self.file_retention_time = self.settings.get("file_retention_time", 86400)
        self.model_inactivity_timeout = self.settings.get("model_inactivity_timeout", 3600)
        self.unload_check_interval = self.settings.get("unload_check_interval", 600)

        self.default_guidance_scale = self.generation_defaults_config.get("default_guidance_scale", 6.0)
        self.default_num_inference_steps = self.generation_defaults_config.get("default_num_inference_steps", 50)

        # State for loaded models and GPU access control
        self.pipelines: Dict[str, Any] = {} # Stores currently loaded pipeline objects
        self.last_used_times: Dict[str, float] = {} # Tracks last access time for loaded models
        self.gpu_state_lock = asyncio.Lock() # Lock to control GPU access (loading & inference)
        # self.loading_locks = defaultdict(asyncio.Lock) # No longer needed with global gpu lock

        logger.info("VideoGenService initialized. No models loaded at startup.")

    def _get_torch_dtype(self):
        configured_dtype = self.settings.get("dtype", "bfloat16").lower()
        dtype = torch.float16; dtype_str = "float16"
        if configured_dtype == "bfloat16":
            if hasattr(torch, 'bfloat16') and self.use_gpu and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16; dtype_str = "bfloat16"
            else: logger.warning("BFloat16 configured but unsupported/unavailable. Falling back to Float16.")
        elif configured_dtype != "float16": logger.warning(f"Unsupported dtype '{configured_dtype}'. Falling back to Float16.")
        return dtype, dtype_str

    def _get_model_specific_dtype(self, model_type: str, model_name: str) -> torch.dtype:
        if model_type == "hunyuan": return self.dtype # Handled internally during load
        if model_type == "wan" and "vae" in model_name.lower(): return torch.float32
        return self.dtype

    def _load_single_pipeline(self, model_key: str) -> Any:
        """Loads a single pipeline based on its key. Runs synchronously."""
        # (Implementation is the same as previous version, handling different model types)
        if model_key not in self.models_config: raise ValueError(f"Model '{model_key}' not found in config.")
        model_info = self.models_config[model_key]; model_name = model_info["name"]
        model_type = model_info.get("type", "unknown"); model_variant = model_info.get("variant"); adapter_name = model_info.get("adapter")
        logger.info(f"Attempting to load pipeline: {model_key} (HF: {model_name}, Type: {model_type})")
        if self.force_gpu and not torch.cuda.is_available(): raise RuntimeError("force_gpu requires available GPU.")
        current_dtype = self._get_model_specific_dtype(model_type, model_name)
        load_kwargs = {"torch_dtype": current_dtype, "cache_dir": self.model_cache_dir}
        if model_variant: load_kwargs["variant"] = model_variant
        pipeline = None
        try:
            if model_type == "cogvideox": pipeline = CogVideoXPipeline.from_pretrained(model_name, **load_kwargs)
            elif model_type in ["stablevideo", "img2vid"]: pipeline = StableVideoDiffusionPipeline.from_pretrained(model_name, **load_kwargs)
            elif model_type == "mochi": pipeline = MochiPipeline.from_pretrained(model_name, **load_kwargs)
            elif model_type == "wan":
                vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32, cache_dir=self.model_cache_dir)
                pipeline = WanPipeline.from_pretrained(model_name, vae=vae, torch_dtype=self.dtype, cache_dir=self.model_cache_dir)
            elif model_type == "hunyuan":
                transformer_dtype = torch.bfloat16 if hasattr(torch,'bfloat16') and self.use_gpu and torch.cuda.is_bf16_supported() else self.dtype
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=transformer_dtype, cache_dir=self.model_cache_dir)
                pipeline = HunyuanVideoPipeline.from_pretrained(model_name, transformer=transformer, torch_dtype=self.dtype, cache_dir=self.model_cache_dir)
            elif model_type == "ltx": pipeline = LTXPipeline.from_pretrained(model_name, **load_kwargs)
            elif model_type == "animatediff":
                 if not adapter_name: raise ValueError(f"AnimateDiff '{model_key}' needs 'adapter'.")
                 adapter = MotionAdapter.from_pretrained(adapter_name, torch_dtype=self.dtype, cache_dir=self.model_cache_dir)
                 pipeline = AnimateDiffPipeline.from_pretrained(model_name, motion_adapter=adapter, torch_dtype=self.dtype, cache_dir=self.model_cache_dir)
                 try:
                     scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler", torch_dtype=self.dtype, cache_dir=self.model_cache_dir, clip_sample=False, timestep_spacing="linspace", beta_schedule="linear", steps_offset=1)
                     pipeline.scheduler = scheduler
                 except Exception as e: logger.warning(f"Could not set DDIM scheduler for {model_key}: {e}")
            else: raise ValueError(f"Unsupported model type '{model_type}'.")

            pipeline.to(self.device)
            if self.use_gpu: # Apply optimizations
                optimizations = []
                if hasattr(pipeline, "enable_model_cpu_offload"): pipeline.enable_model_cpu_offload(); optimizations.append("CPU offload")
                if hasattr(pipeline, "enable_vae_slicing"): pipeline.enable_vae_slicing(); optimizations.append("VAE slicing")
                if hasattr(pipeline, "enable_vae_tiling"): pipeline.enable_vae_tiling(); optimizations.append("VAE tiling")
                if optimizations: logger.info(f"Applied optimizations for {model_key}: {', '.join(optimizations)}")
            elif not self.force_gpu: logger.info(f"Model {model_key} loaded to CPU.")
            else: raise RuntimeError("force_gpu requires GPU.")
            logger.info(f"Successfully loaded pipeline for {model_key}.")
            return pipeline
        except Exception as e:
            logger.exception(f"Error loading pipeline {model_key}: {e}")
            # Clean up potential partial load? Difficult. Rely on GC.
            if pipeline is not None: del pipeline; gc.collect(); torch.cuda.empty_cache() if self.use_gpu else None
            raise # Re-raise

    async def unload_inactive_models(self):
        """Unloads the currently loaded model if it's inactive."""
        if self.model_inactivity_timeout <= 0: return

        async with self.gpu_state_lock: # Lock needed to safely check/modify pipelines
            if not self.pipelines: # No models loaded
                 # logger.debug("Unload check: No models loaded.") # Reduce noise
                 return

            # Since only one model can be loaded at a time (due to VRAM constraint logic)
            current_model_key = list(self.pipelines.keys())[0]
            last_used = self.last_used_times.get(current_model_key, 0)
            idle_time = time.time() - last_used

            logger.debug(f"Unload check: Model '{current_model_key}' idle for {idle_time:.0f}s (Timeout: {self.model_inactivity_timeout}s)")

            if idle_time > self.model_inactivity_timeout:
                logger.info(f"Unloading inactive model: {current_model_key} (idle for {idle_time:.0f}s)")
                pipeline_to_delete = self.pipelines.pop(current_model_key, None)
                self.last_used_times.pop(current_model_key, None)

                if pipeline_to_delete:
                    del pipeline_to_delete
                    logger.info(f"Removed '{current_model_key}' from active pipelines.")
                    # Try to clear GPU memory
                    if self.use_gpu:
                        try:
                            gc.collect()
                            torch.cuda.empty_cache()
                            logger.info("Cleared CUDA cache after unloading model.")
                        except Exception as e:
                            logger.warning(f"Could not clear CUDA cache: {e}")
            # else: # Reduce noise
                # logger.debug(f"Model '{current_model_key}' is active, keeping loaded.")

    def _get_model_defaults(self, model_type: str) -> dict:
        # (Implementation remains the same)
        defaults = {}
        prefix = f"{model_type}_default_"
        for key, value in self.generation_defaults_config.items():
            if key.startswith(prefix): defaults[key.replace(prefix, "")] = value
        if "guidance_scale" not in defaults: defaults["guidance_scale"] = self.default_guidance_scale
        if "num_inference_steps" not in defaults: defaults["num_inference_steps"] = self.default_num_inference_steps
        if model_type == "img2vid":
             i2v_prefix = "img2vid_default_"
             for key, value in self.generation_defaults_config.items():
                 if key.startswith(i2v_prefix) and key.replace(i2v_prefix, "") not in defaults:
                      defaults[key.replace(i2v_prefix, "")] = value
        if not defaults:
            defaults = {"height": 512, "width": 512, "num_frames": 16, "fps": 8, "guidance_scale": self.default_guidance_scale, "num_inference_steps": self.default_num_inference_steps, "motion_bucket_id": 127, "noise_aug_strength": 0.02, "decode_chunk_size": 8}
            logger.warning(f"No defaults for '{model_type}', using basic fallback.")
        return defaults

    def _prepare_generator(self, seed: int) -> Optional[torch.Generator]:
        # (Implementation remains the same)
        if seed == -1: logger.debug("Using random seed."); return None
        else: logger.debug(f"Using seed {seed}."); return torch.Generator(device=self.device).manual_seed(seed)

    def _run_inference(self, job_id: str, pipeline: Any, gen_params: Dict[str, Any], model_key: str, model_type: str):
        """Runs the model inference step. Synchronous part."""
        # (Implementation remains the same as previous version)
        video_frames = None
        def step_callback(pipe, step: int, timestep: int, callback_kwargs: dict):
            total_steps = gen_params.get("num_inference_steps", self.default_num_inference_steps)
            if total_steps > 0:
                 progress = int(((step + 1) / total_steps) * 90);
                 if job_id in jobs: jobs[job_id].progress = min(progress, 90)
                 # logger.debug(f"Job {job_id} progress: {jobs[job_id].progress if job_id in jobs else 'N/A'}% (Step: {step+1}/{total_steps})") # Reduce noise
            return callback_kwargs
        callback_supported = False; call_signature = None
        if hasattr(pipeline, '__call__'):
            try:
                import inspect; call_signature = inspect.signature(pipeline.__call__)
                cb_kwargs = {}
                if 'callback_steps' in call_signature.parameters and 'callback' in call_signature.parameters:
                    cb_kwargs = {"callback_steps": 1, "callback": step_callback}; callback_supported = True
                elif 'callback_on_step_end' in call_signature.parameters:
                     cb_kwargs = {"callback_on_step_end": step_callback, "callback_on_step_end_tensor_inputs": ["latents"]}; callback_supported = True # Adjust inputs if needed
                if callback_supported: gen_params.update(cb_kwargs); logger.debug(f"Job {job_id}: Callback enabled.")
            except Exception as e: logger.warning(f"Job {job_id}: Sig inspection failed: {e}")
        if not callback_supported: logger.debug(f"Job {job_id}: No callback support.") # Reduce noise

        autocast_context = torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype if model_type != "wan" else torch.float16, enabled=self.use_gpu)
        logger.info(f"Job {job_id}: Running inference for {model_key}...")
        cleaned_gen_params = {k: v for k, v in gen_params.items() if v is not None or k == "negative_prompt"}
        with autocast_context: output = pipeline(**cleaned_gen_params)
        if hasattr(output, "frames"): video_frames = output.frames[0] if isinstance(output.frames, list) else output.frames
        elif hasattr(output, "video"): video_frames = output.video[0] if isinstance(output.video, list) else output.video
        else: raise ValueError("No 'frames' or 'video' in pipeline output.")
        if video_frames is None: raise RuntimeError("Frame generation resulted in None.")
        if job_id in jobs: jobs[job_id].progress = 95
        return video_frames

    # --- Generation Task Logic (Refactored for Locking and Status) ---

    async def _generation_task_wrapper(self, job_id: str, model_key: str, generation_func: callable, *args, **kwargs):
        """Handles locking, loading, running generation, and status updates."""
        if job_id not in jobs: logger.error(f"Job {job_id} disappeared before task start."); return
        pipeline = None
        acquired_lock = False
        max_wait_retries = 120 # Wait up to 10 minutes (120 * 5s)
        retry_count = 0

        try:
            while retry_count < max_wait_retries:
                logger.debug(f"Job {job_id}: Attempting to acquire GPU lock (Retry {retry_count})...")
                async with self.gpu_state_lock:
                    acquired_lock = True
                    logger.debug(f"Job {job_id}: Acquired GPU lock.")
                    current_gpu_model = list(self.pipelines.keys())[0] if self.pipelines else None

                    if current_gpu_model == model_key:
                        logger.info(f"Job {job_id}: Required model '{model_key}' already loaded. Proceeding.")
                        pipeline = self.pipelines[model_key]
                        self.last_used_times[model_key] = time.time()
                        break # Exit retry loop, proceed with generation

                    elif current_gpu_model is None:
                        logger.info(f"Job {job_id}: GPU free. Loading required model '{model_key}'.")
                        if job_id in jobs: jobs[job_id].status = JobStatusEnum.LOADING_MODEL
                        try:
                            loop = asyncio.get_running_loop()
                            # Load synchronously in executor while holding lock
                            pipeline = await loop.run_in_executor(None, self._load_single_pipeline, model_key)
                            self.pipelines[model_key] = pipeline
                            self.last_used_times[model_key] = time.time()
                            logger.info(f"Job {job_id}: Model '{model_key}' loaded successfully.")
                            break # Exit retry loop, proceed with generation
                        except Exception as load_err:
                             logger.error(f"Job {job_id}: Failed to load model '{model_key}': {load_err}")
                             raise  # Re-raise to be caught by outer try-except

                    else: # GPU busy with a different model
                        logger.info(f"Job {job_id}: GPU busy with model '{current_gpu_model}'. Waiting for resources...")
                        if job_id in jobs: jobs[job_id].status = JobStatusEnum.WAITING_FOR_RESOURCES; jobs[job_id].message = f"Waiting for GPU (currently used by {current_model_key})"
                        # Release lock and retry later
                        acquired_lock = False # Mark lock as released for the finally block
                        # Exit the 'with' block here to release the lock before sleeping
                # Lock is released here if GPU was busy
                if jobs[job_id].status == JobStatusEnum.WAITING_FOR_RESOURCES:
                     retry_count += 1
                     await asyncio.sleep(5) # Wait before retrying
                     if job_id not in jobs: logger.warning(f"Job {job_id} cancelled while waiting."); return # Check if job was cancelled
                     continue # Go back to acquire lock
                 # If we loaded successfully or found the model, the loop was broken

            # Check if we timed out waiting
            if retry_count >= max_wait_retries:
                 raise TimeoutError(f"Job {job_id} timed out waiting for GPU resources.")

            # --- Proceed with Generation (Lock is held if we didn't wait) ---
            if job_id in jobs: jobs[job_id].status = JobStatusEnum.PROCESSING; jobs[job_id].message = "Generating video..."
            loop = asyncio.get_running_loop()

            # Run the specific generation logic (T2V or I2V)
            await generation_func(job_id, pipeline, loop, *args, **kwargs)

            # Final status update (already done in generation_func on success)

        except Exception as e:
            elapsed = time.time() - jobs[job_id].created_at # Use creation time for total duration
            logger.exception(f"Job {job_id}: Generation task failed after {elapsed:.2f}s: {str(e)}")
            if job_id in jobs:
                jobs[job_id].status = JobStatusEnum.FAILED
                jobs[job_id].message = f"Task failed: {str(e)}"
                jobs[job_id].progress = 0
        finally:
            # Ensure lock is released if it was acquired and held
            if acquired_lock and self.gpu_state_lock.locked():
                 self.gpu_state_lock.release()
                 logger.debug(f"Job {job_id}: Released GPU lock.")
            # Pipeline object reference is lost when function exits, GC handles it.
            # Unloading is handled by the periodic task.

    async def _execute_t2v_inference(self, job_id: str, pipeline: Any, loop: asyncio.AbstractEventLoop, request: TextVideoGenerationRequest):
        """Contains the actual T2V inference and export logic."""
        model_key = request.model_name or self.default_t2v_model_key
        model_info = self.models_config.get(model_key, {})
        model_type = model_info.get("type", "unknown")
        model_defaults = self._get_model_defaults(model_type)

        height = request.height or model_defaults.get("height", 512)
        width = request.width or model_defaults.get("width", 512)
        nb_frames = request.nb_frames or model_defaults.get("num_frames", 16)
        fps = request.fps or model_defaults.get("fps", 8)
        guidance_scale = request.guidance_scale if request.guidance_scale is not None else model_defaults.get("guidance_scale", self.default_guidance_scale)
        steps = request.steps or model_defaults.get("num_inference_steps", self.default_num_inference_steps)
        negative_prompt = request.negative_prompt
        generator = self._prepare_generator(request.seed)

        gen_params = {
            "prompt": request.prompt, "negative_prompt": negative_prompt, "num_frames": nb_frames,
            "num_inference_steps": steps, "guidance_scale": guidance_scale, "height": height,
            "width": width, "generator": generator,
        }
        if model_type == "mochi": gen_params.pop("guidance_scale", None); gen_params.pop("negative_prompt", None)
        if model_type == "hunyuan": gen_params.pop("guidance_scale", None)

        logger.info(f"Job {job_id}: Running T2V inference for {model_key}...")
        video_frames = await loop.run_in_executor(
            None, self._run_inference, job_id, pipeline, gen_params, model_key, model_type
        )

        output_filename = self.output_folder / f"video_{job_id}.mp4"
        logger.info(f"Job {job_id}: Exporting T2V video to {output_filename} with FPS={fps}...")
        await loop.run_in_executor(None, export_to_video, video_frames, str(output_filename), fps)

        if job_id in jobs:
             jobs[job_id].progress = 100
             elapsed = time.time() - jobs[job_id].created_at
             logger.info(f"Job {job_id}: T2V generated successfully in {elapsed:.2f}s.")
             jobs[job_id].status = JobStatusEnum.COMPLETED
             jobs[job_id].video_url = f"/download/{job_id}"
             jobs[job_id].message = "Text-to-Video generated successfully."

    async def _execute_i2v_inference(self, job_id: str, pipeline: Any, loop: asyncio.AbstractEventLoop, image: Image.Image, request: ImageVideoGenerationRequest):
        """Contains the actual I2V inference and export logic."""
        model_key = request.model_name or self.default_i2v_model_key
        model_info = self.models_config.get(model_key, {})
        model_type = model_info.get("type", "unknown")
        model_defaults = self._get_model_defaults(model_type)

        height = request.height or model_defaults.get("height", 576)
        width = request.width or model_defaults.get("width", 1024)
        fps = request.fps or model_defaults.get("fps", 7)
        motion_bucket_id = request.motion_bucket_id if request.motion_bucket_id is not None else model_defaults.get("motion_bucket_id", 127)
        noise_aug_strength = request.noise_aug_strength if request.noise_aug_strength is not None else model_defaults.get("noise_aug_strength", 0.02)
        decode_chunk_size = request.decode_chunk_size or model_defaults.get("decode_chunk_size", 8)
        num_inference_steps = request.num_inference_steps or model_defaults.get("num_inference_steps", 25)
        generator = self._prepare_generator(request.seed)

        if image.width != width or image.height != height:
            logger.warning(f"Job {job_id}: Resizing input image from {image.size} to {width}x{height}.")
            image = image.resize((width, height), Image.Resampling.LANCZOS)


        gen_params = {
            "image": image, "height": height, "width": width, "fps": fps,
            "motion_bucket_id": motion_bucket_id, "noise_aug_strength": noise_aug_strength,
            "decode_chunk_size": decode_chunk_size, "generator": generator,
            "prompt": request.prompt if request.prompt else None,
            "num_inference_steps": num_inference_steps,
        }

        logger.info(f"Job {job_id}: Running I2V inference for {model_key}...")
        video_frames = await loop.run_in_executor(
             None, self._run_inference, job_id, pipeline, gen_params, model_key, model_type
        )

        output_filename = self.output_folder / f"video_{job_id}.mp4"
        logger.info(f"Job {job_id}: Exporting I2V video to {output_filename} with FPS={fps}...")
        await loop.run_in_executor(None, export_to_video, video_frames, str(output_filename), fps)

        if job_id in jobs:
             jobs[job_id].progress = 100
             elapsed = time.time() - jobs[job_id].created_at
             logger.info(f"Job {job_id}: I2V generated successfully in {elapsed:.2f}s.")
             jobs[job_id].status = JobStatusEnum.COMPLETED
             jobs[job_id].video_url = f"/download/{job_id}"
             jobs[job_id].message = "Image-to-Video generated successfully."

    async def generate_video(self, job_id: str, request: TextVideoGenerationRequest) -> None:
        """Async task entry point for Text-to-Video."""
        model_key = request.model_name or self.default_t2v_model_key
        if not model_key:
            if job_id in jobs: jobs[job_id].status = JobStatusEnum.FAILED; jobs[job_id].message = "No T2V model specified/configured."
            return
        await self._generation_task_wrapper(job_id, model_key, self._execute_t2v_inference, request)

    async def generate_video_from_image(self, job_id: str, image: Image.Image, request: ImageVideoGenerationRequest) -> None:
        """Async task entry point for Image-to-Video."""
        model_key = request.model_name or self.default_i2v_model_key
        if not model_key:
            if job_id in jobs: jobs[job_id].status = JobStatusEnum.FAILED; jobs[job_id].message = "No I2V model specified/configured."
            return
        await self._generation_task_wrapper(job_id, model_key, self._execute_i2v_inference, image, request)

    async def generate_video_by_frames(self, job_id: str, request: MultiPromptVideoRequest) -> None:
        """Async task for multi-prompt T2V."""
        # Setup remains synchronous, delegation calls the async wrapper
        if job_id not in jobs: logger.error(f"Job {job_id} missing for multi-prompt."); return
        jobs[job_id].status = JobStatusEnum.PROCESSING # Or PENDING? Let wrapper handle status.

        try:
            if not request.prompts or not request.frames or len(request.prompts) != len(request.frames):
                raise ValueError("Prompts/frames mismatch.")

            combined_prompt = " ".join(request.prompts)
            total_frames = sum(request.frames)
            model_key = request.model_name or self.default_t2v_model_key
            if not model_key: raise ValueError("No T2V model specified.")

            model_info = self.models_config.get(model_key)
            if not model_info or model_info.get("type", "unknown") in ["img2vid", "stablevideo"]:
                 raise ValueError(f"Invalid T2V model: {model_key}.")

            model_defaults = self._get_model_defaults(model_info["type"])
            video_request = TextVideoGenerationRequest(
                prompt=combined_prompt, negative_prompt=request.negative_prompt, model_name=model_key,
                height=request.height or model_defaults.get("height", 512),
                width=request.width or model_defaults.get("width", 512),
                steps=request.num_inference_steps, guidance_scale=request.guidance_scale,
                seed=request.seed if request.seed is not None else -1,
                nb_frames=total_frames,
                fps=request.fps or model_defaults.get("fps", 8)
            )
            # Delegate to the main T2V task wrapper
            await self._generation_task_wrapper(job_id, model_key, self._execute_t2v_inference, video_request)

        except Exception as e:
             logger.exception(f"Job {job_id}: Multi-prompt setup failed: {str(e)}")
             if job_id in jobs:
                  jobs[job_id].status = JobStatusEnum.FAILED
                  jobs[job_id].message = f"Multi-prompt failed: {str(e)}"

    def cleanup_expired_files(self):
        # (Implementation remains the same)
        current_time = time.time(); expired_job_ids = []
        for job_id in list(jobs.keys()):
            job = jobs.get(job_id)
            if job and current_time > job.expires_at:
                 expired_job_ids.append(job_id)
                 video_path = self.output_folder / f"video_{job_id}.mp4"
                 if video_path.exists():
                     try: video_path.unlink(); logger.debug(f"Deleted expired file: {video_path}")
                     except OSError as e: logger.error(f"Failed to delete {video_path}: {e}")
                 # elif job.status == JobStatusEnum.COMPLETED: logger.warning(f"Expired completed job {job_id} file missing: {video_path}.") # Reduce noise
        if expired_job_ids:
             logger.info(f"Cleaning {len(expired_job_ids)} expired job records...")
             for job_id in expired_job_ids:
                 if job_id in jobs: del jobs[job_id] # ; logger.debug(f"Removed expired job record: {job_id}") # Reduce noise

# --- Initialize Service ---
try:
    service = VideoGenService()
except Exception as service_init_error:
     logger.exception(f"Failed to initialize VideoGenService: {service_init_error}")
     logger.error("API cannot start."); sys.exit(1)

# --- Periodic Task for Model Unloading ---
async def periodic_model_unload():
    interval = max(1, service.unload_check_interval)
    inactivity_timeout = service.model_inactivity_timeout
    if inactivity_timeout <= 0: logger.info("Periodic model unloading disabled."); return
    logger.info(f"Starting periodic model unload check. Interval: {interval}s, Timeout: {inactivity_timeout}s.")
    while True:
        await asyncio.sleep(interval)
        try: await service.unload_inactive_models()
        except Exception as e: logger.error(f"Error during periodic unload: {e}")

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    loop.create_task(periodic_model_unload())
    # Add file cleanup task if desired
    # loop.create_task(periodic_file_cleanup(service.file_retention_time / 2))

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    if gpu_available:
        try: gpu_info = {"count": torch.cuda.device_count(), "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "bf16_supported": torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else "N/A"}
        except Exception as e: gpu_info = {"error": f"GPU query failed: {e}"}
    # Check pipelines dict size for status
    service_status = "healthy" if service else "uninitialized" # Basic check if service exists
    loaded_models = list(service.pipelines.keys()) if service else []
    return {
        "status": service_status,
        "message": "Service running." if service_status == "healthy" else "Service initialization failed.",
        "default_t2v_model": service.default_t2v_model_key if service else "N/A",
        "default_i2v_model": service.default_i2v_model_key if service else "N/A",
        "currently_loaded_models": loaded_models,
        "available_models_in_config": list(service.models_config.keys()) if service else [],
        "settings": service.settings if service else {}, # Show loaded settings
        "gpu_details": {"available": gpu_available, **gpu_info}
    }

@app.get("/models")
async def get_models():
    model_details = []
    loaded_keys = list(service.pipelines.keys())
    for key, info in service.models_config.items():
         is_loaded = key in loaded_keys
         model_type = info.get("type", "unknown")
         model_details.append({
              "id": key, "name": info.get("name", "N/A"), "type": model_type,
              "task": "Image-to-Video" if model_type in ["img2vid", "stablevideo"] else "Text-to-Video",
              "loaded": is_loaded,
              "status": "Loaded" if is_loaded else ("Configured" if key in service.models_config else "Unknown")
         })
    return {"models": model_details}

# --- Job Submission Endpoints ---

@app.post("/submit", status_code=202, summary="Submit Text-to-Video Job")
async def submit_t2v_job(request: TextVideoGenerationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4()); created_at = time.time()
    expires_at = created_at + service.file_retention_time
    jobs[job_id] = JobStatus(job_id=job_id, status=JobStatusEnum.PENDING, created_at=created_at, expires_at=expires_at)
    model_display = request.model_name or service.default_t2v_model_key or "N/A"
    logger.info(f"Received T2V job {job_id} for model '{model_display}'. Queuing.")
    background_tasks.add_task(service.generate_video, job_id, request)
    return {"job_id": job_id, "message": "Text-to-Video job submitted."}

@app.post("/submit_image_video", status_code=202, summary="Submit Image-to-Video Job")
async def submit_i2v_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    model_name: Optional[str] = Form(None), height: Optional[int] = Form(None), width: Optional[int] = Form(None),
    fps: Optional[int] = Form(None), motion_bucket_id: Optional[int] = Form(None), noise_aug_strength: Optional[float] = Form(None),
    seed: int = Form(-1), decode_chunk_size: Optional[int] = Form(None), num_inference_steps: Optional[int] = Form(None),
    prompt: Optional[str] = Form(None)):
    try:
        image_bytes = await image.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e: raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    finally: await image.close()
    request = ImageVideoGenerationRequest(model_name=model_name, height=height, width=width, fps=fps, motion_bucket_id=motion_bucket_id, noise_aug_strength=noise_aug_strength, seed=seed, decode_chunk_size=decode_chunk_size, num_inference_steps=num_inference_steps, prompt=prompt)
    job_id = str(uuid.uuid4()); created_at = time.time(); expires_at = created_at + service.file_retention_time
    jobs[job_id] = JobStatus(job_id=job_id, status=JobStatusEnum.PENDING, created_at=created_at, expires_at=expires_at)
    model_display = request.model_name or service.default_i2v_model_key or "N/A"
    logger.info(f"Received I2V job {job_id} for model '{model_display}'. Queuing.")
    background_tasks.add_task(service.generate_video_from_image, job_id, input_image, request)
    return {"job_id": job_id, "message": "Image-to-Video job submitted."}

@app.post("/submit_multi", status_code=202, summary="Submit Multi-Prompt T2V Job")
async def submit_multi_job(request: MultiPromptVideoRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4()); created_at = time.time(); expires_at = created_at + service.file_retention_time
    jobs[job_id] = JobStatus(job_id=job_id, status=JobStatusEnum.PENDING, created_at=created_at, expires_at=expires_at)
    model_display = request.model_name or service.default_t2v_model_key or "N/A"
    logger.info(f"Received multi-prompt T2V job {job_id} for model '{model_display}'. Queuing.")
    background_tasks.add_task(service.generate_video_by_frames, job_id, request)
    return {"job_id": job_id, "message": "Multi-prompt job submitted."}

# --- Status and Download Endpoints ---
@app.get("/status/{job_id}", summary="Get Job Status")
async def get_job_status(job_id: str):
    if job_id not in jobs: raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    # Optional: Trigger synchronous file cleanup less often
    if time.time() % 600 < 5: logger.debug("Running file cleanup via status check..."); service.cleanup_expired_files()
    return jobs[job_id]

@app.get("/download/{job_id}", summary="Download Video")
async def download_video(job_id: str):
    if job_id not in jobs: raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    job = jobs[job_id]
    if job.status != JobStatusEnum.COMPLETED:
        status_detail = f"Job '{job_id}' status: {job.status.value}." # Use enum value
        if job.status == JobStatusEnum.FAILED: status_detail += f" Error: {job.message}"
        elif job.status == JobStatusEnum.WAITING_FOR_RESOURCES: status_detail += f" Message: {job.message}"
        raise HTTPException(status_code=400, detail=status_detail)
    if not job.video_url: raise HTTPException(status_code=400, detail="Video URL not available.")
    video_path = service.output_folder / f"video_{job_id}.mp4"
    if not video_path.exists(): logger.error(f"Completed job {job_id} file missing: {video_path}"); raise HTTPException(status_code=404, detail="Video file missing (expired/removed).")
    return FileResponse(path=video_path, media_type="video/mp4", filename=f"openvideogen_{job_id}.mp4")

# --- Static Files and Web UI ---
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Serving static files from: {static_dir.resolve()}")
else: logger.warning(f"Static dir not found: {static_dir}. UI unavailable.")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/webui", response_class=HTMLResponse)
async def serve_webui():
    webui_html_path = static_dir / "webui.html"
    if not webui_html_path.exists(): raise HTTPException(status_code=404, detail="webui.html not found.")
    try:
        with open(webui_html_path, "r", encoding="utf-8") as f: content = f.read()
        return HTMLResponse(content=content)
    except Exception as e: logger.exception(f"Error reading UI: {e}"); raise HTTPException(status_code=500, detail="Could not load UI.")

# --- Application Startup (Main Guard) ---
if __name__ == "__main__":
    import uvicorn
    if not service: logger.critical("Service failed init. Cannot start."); sys.exit(1)
    host = service.settings.get("host", "0.0.0.0"); port = service.settings.get("port", 8088)
    logger.info(f"Starting OpenVideoGen API on http://{host}:{port}")
    logger.info(f"Web UI: http://{host}:{port}/webui | API Docs: http://{host}:{port}/docs")
    logger.info(f"Default T2V: {service.default_t2v_model_key}, Default I2V: {service.default_i2v_model_key}")
    logger.info(f"GPU: {service.use_gpu}, DType: {service.dtype_str}, Inactivity Timeout: {service.model_inactivity_timeout}s")
    # Run via 'openvideogen' command (cli.py) or directly:
    uvicorn.run("openvideogen.main:app", host=host, port=port, reload=False)