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
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import torch
from diffusers import CogVideoXPipeline, StableVideoDiffusionPipeline, MochiPipeline
from diffusers.utils import export_to_video
import toml
import pipmaster as pm
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("openvideogen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OpenVideoGen")

# Install dependencies using subprocess
def install_dependencies():
    logger.info("Checking and installing dependencies...")
    if not pm.is_installed("torch"):
        logger.info(f"Installing torch...")
        pm.install_multiple(["torch", "torchvision", "torchaudio"], "https://download.pytorch.org/whl/cu124")
    for package in ["diffusers", "transformers", "accelerate", "imageio-ffmpeg", "sentencepiece", "toml"]:
        logger.info(f"Installing {package}...")
        if not pm.is_installed(package):
            pm.install(package)
    if not pm.is_version_higher("accelerate", "0.26.0"):
        pm.install("accelerate")
    logger.info("Dependencies installed successfully.")

install_dependencies()

app = FastAPI(
    title="OpenVideoGen API",
    description="Open source video generation API using various diffusion models.",
    version="0.1.0"
)

# Determine config file search paths based on OS
def get_config_search_paths():
    system = platform.system()
    search_paths = []

    if system == "Linux":
        search_paths.extend([
            Path("/etc/openvideogen/config.toml"),
            Path("/usr/local/etc/openvideogen/config.toml"),
            Path.home() / ".config/openvideogen/config.toml",
            Path.cwd() / "config.toml"
        ])
    elif system == "Windows":
        search_paths.extend([
            Path(os.getenv("APPDATA", Path.home() / "AppData/Roaming")) / "openvideogen/config.toml",
            Path.cwd() / "config.toml"
        ])
    elif system == "Darwin":  # macOS
        search_paths.extend([
            Path.home() / "Library/Application Support/openvideogen/config.toml",
            Path("/usr/local/etc/openvideogen/config.toml"),
            Path.cwd() / "config.toml"
        ])
    else:
        search_paths.append(Path.cwd() / "config.toml")

    return search_paths

# Load configuration with priority: command-line arg > env var > search paths
def load_config():
    DEFAULT_CONFIG = {
        "models": {
            "cogvideox_2b": {"name": "THUDM/CogVideoX-2b", "type": "cogvideox"},
            "cogvideox_5b": {"name": "THUDM/CogVideoX-5b", "type": "cogvideox"},
            "stable_video": {"name": "stabilityai/stable-video-diffusion-img2vid", "type": "stablevideo"},
            "mochi": {"name": "genmo/mochi-1-preview", "type": "mochi"}
        },
        "settings": {
            "default_model": "cogvideox_2b",
            "force_gpu": False,
            "use_gpu": True,
            "dtype": "float16",
            "output_folder": "./outputs",
            "model_cache_dir": "./model_cache",  # New setting for model cache directory
            "port": 8088,
            "host": "0.0.0.0",
            "file_retention_time": 3600  # 1 hour in seconds
        },
        "generation": {
            "guidance_scale": 6.0,
            "num_inference_steps": 50
        }
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="OpenVideoGen API Server")
    parser.add_argument("--config", type=str, help="Path to the config.toml file")
    args = parser.parse_args()

    # Check command-line argument
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file specified via --config not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Loading config from command-line argument: {config_path}")
        return toml.load(config_path)

    # Check environment variable
    env_config_path = os.getenv("OPENVIDEOGEN_CONFIG")
    if env_config_path:
        config_path = Path(env_config_path)
        if not config_path.exists():
            logger.error(f"Config file specified via OPENVIDEOGEN_CONFIG not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Loading config from environment variable: {config_path}")
        return toml.load(config_path)

    # Search for config file in standard locations
    search_paths = get_config_search_paths()
    for path in search_paths:
        if path.exists():
            logger.info(f"Loading config from: {path}")
            return toml.load(path)

    # If no config file is found, create a default one in the current directory
    default_path = Path.cwd() / "config.toml"
    with open(default_path, "w") as f:
        toml.dump(DEFAULT_CONFIG, f)
        logger.info(f"Created default config.toml at: {default_path}")
    return DEFAULT_CONFIG

config = load_config()

# Pydantic models for request validation
class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    height: int = 480
    width: int = 720
    steps: Optional[int] = None  # Override num_inference_steps
    guidance_scale: Optional[float] = None  # Override guidance_scale
    seed: int = -1
    nb_frames: int = 49
    fps: int = 8

class MultiPromptVideoRequest(BaseModel):
    prompts: List[str]
    frames: List[int]
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    fps: int = 8
    num_inference_steps: Optional[int] = None  # Override num_inference_steps
    guidance_scale: Optional[float] = None  # Override guidance_scale
    seed: Optional[int] = None

# Job status model
class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: Optional[str] = None
    video_url: Optional[str] = None
    created_at: float
    expires_at: float

# In-memory job storage (use a database like Redis in production)
jobs: Dict[str, JobStatus] = {}

# Video Generation Service
class VideoGenService:
    def __init__(self):
        self.models = config["models"]
        self.default_model = config["settings"].get("default_model", "cogvideox_2b")
        self.use_gpu = config["settings"].get("use_gpu", True) and torch.cuda.is_available()
        self.force_gpu = config["settings"].get("force_gpu", False)
        self.dtype = torch.float16 if config["settings"].get("dtype", "float16") == "float16" else torch.bfloat16
        self.output_folder = Path(config["settings"].get("output_folder", "./outputs"))
        self.model_cache_dir = Path(config["settings"].get("model_cache_dir", "./models"))  # New attribute
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.model_cache_dir.mkdir(exist_ok=True, parents=True)  # Create the directory if it doesn't exist
        self.file_retention_time = config["settings"].get("file_retention_time", 3600 * 5)
        self.default_guidance_scale = config["generation"].get("guidance_scale", 6)
        self.default_num_inference_steps = config["generation"].get("num_inference_steps", 50)
        self.pipelines = {}
        self.load_pipelines()

    def load_pipelines(self):
        if self.force_gpu and not torch.cuda.is_available():
            logger.error("force_gpu is set to True, but no GPU is available.")
            raise RuntimeError("force_gpu is set to True, but no GPU is available.")

        for model_key, model_info in self.models.items():
            try:
                if model_info["type"] == "cogvideox":
                    pipeline = CogVideoXPipeline.from_pretrained(
                        model_info["name"], 
                        torch_dtype=self.dtype,
                        cache_dir=self.model_cache_dir  # Pass the cache directory
                    )
                elif model_info["type"] == "stablevideo":
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(
                        model_info["name"], 
                        torch_dtype=self.dtype,
                        cache_dir=self.model_cache_dir  # Pass the cache directory
                    )
                elif model_info["type"] == "mochi":
                    pipeline = MochiPipeline.from_pretrained(
                        model_info["name"], 
                        torch_dtype=self.dtype,
                        cache_dir=self.model_cache_dir  # Pass the cache directory
                    )
                    pipeline.enable_vae_tiling()
                else:
                    logger.warning(f"Unsupported model type for {model_key}: {model_info['type']}")
                    continue

                if self.force_gpu or (self.use_gpu and torch.cuda.is_available()):
                    pipeline.to("cuda")
                    pipeline.enable_model_cpu_offload()
                self.pipelines[model_key] = pipeline
                logger.info(f"Loaded model: {model_key} into {self.model_cache_dir}")
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {str(e)}")

    def generate_video(self, job_id: str, request: VideoGenerationRequest) -> None:
        jobs[job_id].status = "processing"
        model_key = request.model_name or self.default_model
        if model_key not in self.pipelines:
            jobs[job_id].status = "failed"
            jobs[job_id].message = f"Model {model_key} not found or not loaded."
            return

        pipeline = self.pipelines[model_key]
        output_path = self.output_folder
        output_path.mkdir(exist_ok=True, parents=True)

        gen_params = {
            "prompt": request.prompt,
            "num_frames": request.nb_frames,
            "num_inference_steps": request.steps if request.steps is not None else self.default_num_inference_steps,
            "guidance_scale": request.guidance_scale if request.guidance_scale is not None else self.default_guidance_scale,
            "height": request.height,
            "width": request.width,
        }
        if request.seed != -1:
            gen_params["generator"] = torch.Generator(device="cuda" if (self.force_gpu or self.use_gpu) else "cpu").manual_seed(request.seed)

        try:
            logger.info(f"Generating video for job {job_id} with {model_key}...")
            start_time = time.time()
            
            # Check if the model is Mochi and apply autocast
            if self.models[model_key]["type"] == "mochi":
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    video_frames = pipeline(**gen_params).frames[0]
            else:
                video_frames = pipeline(**gen_params).frames[0]

            output_filename = output_path / f"video_{job_id}.mp4"
            export_to_video(video_frames, str(output_filename), fps=request.fps)
            elapsed_time = time.time() - start_time
            logger.info(f"Video for job {job_id} generated in {elapsed_time:.2f}s at {output_filename}")
            jobs[job_id].status = "completed"
            jobs[job_id].video_url = f"/download/{job_id}"
            jobs[job_id].message = "Video generated successfully"
        except Exception as e:
            logger.error(f"Video generation for job {job_id} failed: {str(e)}")
            jobs[job_id].status = "failed"
            jobs[job_id].message = f"Video generation failed: {str(e)}"

    def generate_video_by_frames(self, job_id: str, request: MultiPromptVideoRequest) -> None:
        if not request.prompts or not request.frames:
            jobs[job_id].status = "failed"
            jobs[job_id].message = "Prompts and frames lists cannot be empty"
            return

        combined_prompt = " ".join(request.prompts)
        total_frames = sum(request.frames)

        video_request = VideoGenerationRequest(
            prompt=combined_prompt,
            model_name=request.model_name,
            nb_frames=total_frames,
            steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed if request.seed is not None else -1,
            output_folder=request.output_folder if request.output_folder else config["settings"]["output_folder"],
            fps=request.fps,
            height=480,  # Default values
            width=720
        )
        self.generate_video(job_id, video_request)

    def cleanup_expired_files(self):
        """Remove expired files and job records."""
        current_time = time.time()
        expired_jobs = []
        for job_id, job in jobs.items():
            if current_time > job.expires_at:
                expired_jobs.append(job_id)
                video_path = self.output_folder / f"video_{job_id}.mp4"
                if video_path.exists():
                    try:
                        video_path.unlink()
                        logger.info(f"Deleted expired file for job {job_id}: {video_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete expired file for job {job_id}: {str(e)}")

        for job_id in expired_jobs:
            del jobs[job_id]
            logger.info(f"Removed expired job record: {job_id}")

service = VideoGenService()

# API Endpoints
@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {
        "status": "healthy",
        "default_model": service.default_model,
        "loaded_models": list(service.pipelines.keys()),
        "gpu_available": torch.cuda.is_available(),
        "force_gpu": service.force_gpu,
        "use_gpu": service.use_gpu
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": list(service.models.keys())}

@app.post("/submit")
async def submit_job(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """Submit a video generation job and return a job ID"""
    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + service.file_retention_time

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=created_at,
        expires_at=expires_at
    )

    background_tasks.add_task(service.generate_video, job_id, request)
    background_tasks.add_task(service.cleanup_expired_files)

    return {"job_id": job_id, "message": "Job submitted successfully"}

@app.post("/submit_multi")
async def submit_multi_job(request: MultiPromptVideoRequest, background_tasks: BackgroundTasks):
    """Submit a multi-prompt video generation job and return a job ID"""
    job_id = str(uuid.uuid4())
    created_at = time.time()
    expires_at = created_at + service.file_retention_time

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=created_at,
        expires_at=expires_at
    )

    background_tasks.add_task(service.generate_video_by_frames, job_id, request)
    background_tasks.add_task(service.cleanup_expired_files)

    return {"job_id": job_id, "message": "Multi-prompt job submitted successfully"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the generated video for a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status != "completed" or not job.video_url:
        raise HTTPException(status_code=400, detail="Video not ready or generation failed")

    video_path = service.output_folder / f"video_{job_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(video_path, media_type="video/mp4", filename=f"video_{job_id}.mp4")

if __name__ == "__main__":
    import uvicorn
    host = config["settings"]["host"]
    port = config["settings"]["port"]
    uvicorn.run(app, host=host, port=port)