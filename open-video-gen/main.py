import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import time
from pathlib import Path
import torch
from diffusers import CogVideoXPipeline, StableVideoDiffusionPipeline
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

# Install dependencies
def install_dependencies():
    logger.info("Checking and installing dependencies...")
    if not pm.is_installed("torch"):
        pm.install_multiple(["torch", "torchvision", "torchaudio"], "https://download.pytorch.org/whl/cu124")
    for package in ["diffusers", "transformers", "accelerate", "imageio-ffmpeg", "sentencepiece", "toml"]:
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

# Load configuration
CONFIG_PATH = Path("config.toml")
DEFAULT_CONFIG = {
    "models": {
        "cogvideox_2b": {"name": "THUDM/CogVideoX-2b", "type": "cogvideox"},
        "cogvideox_5b": {"name": "THUDM/CogVideoX-5b", "type": "cogvideox"},
        "stable_video": {"name": "stabilityai/stable-video-diffusion-img2vid", "type": "stablevideo"}
    },
    "settings": {
        "default_model": "cogvideox_2b",
        "use_gpu": True,
        "dtype": "float16",
        "output_folder": "./outputs"
    }
}

if not CONFIG_PATH.exists():
    with open(CONFIG_PATH, "w") as f:
        toml.dump(DEFAULT_CONFIG, f)
        logger.info("Created default config.toml")

config = toml.load(CONFIG_PATH)

# Pydantic models for request validation
class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    height: int = 480
    width: int = 720
    steps: int = 50
    seed: int = -1
    nb_frames: int = 49
    fps: int = 8
    output_folder: Optional[str] = None

class MultiPromptVideoRequest(BaseModel):
    prompts: List[str]
    frames: List[int]
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    fps: int = 8
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    seed: Optional[int] = None

# Video Generation Service
class VideoGenService:
    def __init__(self):
        self.models = config["models"]
        self.default_model = config["settings"]["default_model"]
        self.use_gpu = config["settings"]["use_gpu"] and torch.cuda.is_available()
        self.dtype = torch.float16 if config["settings"]["dtype"] == "float16" else torch.bfloat16
        self.output_folder = Path(config["settings"]["output_folder"])
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.pipelines = {}
        self.load_pipelines()

    def load_pipelines(self):
        for model_key, model_info in self.models.items():
            try:
                if model_info["type"] == "cogvideox":
                    pipeline = CogVideoXPipeline.from_pretrained(
                        model_info["name"], torch_dtype=self.dtype
                    )
                elif model_info["type"] == "stablevideo":
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(
                        model_info["name"], torch_dtype=self.dtype
                    )
                else:
                    logger.warning(f"Unsupported model type for {model_key}: {model_info['type']}")
                    continue

                if self.use_gpu:
                    pipeline.to("cuda")
                    pipeline.enable_model_cpu_offload()
                self.pipelines[model_key] = pipeline
                logger.info(f"Loaded model: {model_key}")
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {str(e)}")

    def generate_video(self, request: VideoGenerationRequest) -> str:
        model_key = request.model_name or self.default_model
        if model_key not in self.pipelines:
            raise ValueError(f"Model {model_key} not found or not loaded.")

        pipeline = self.pipelines[model_key]
        output_path = Path(request.output_folder) if request.output_folder else self.output_folder
        output_path.mkdir(exist_ok=True, parents=True)

        gen_params = {
            "prompt": request.prompt,
            "num_frames": request.nb_frames,
            "num_inference_steps": request.steps,
            "guidance_scale": 6.0,
            "height": request.height,
            "width": request.width,
        }
        if request.seed != -1:
            gen_params["generator"] = torch.Generator(device="cuda" if self.use_gpu else "cpu").manual_seed(request.seed)

        try:
            logger.info(f"Generating video with {model_key}...")
            start_time = time.time()
            video_frames = pipeline(**gen_params).frames[0]
            output_filename = output_path / f"video_{model_key}_{int(time.time())}.mp4"
            export_to_video(video_frames, str(output_filename), fps=request.fps)
            elapsed_time = time.time() - start_time
            logger.info(f"Video generated in {elapsed_time:.2f}s at {output_filename}")
            return str(output_filename)
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

    def generate_video_by_frames(self, request: MultiPromptVideoRequest) -> str:
        if not request.prompts or not request.frames:
            raise ValueError("Prompts and frames lists cannot be empty")

        combined_prompt = " ".join(request.prompts)
        total_frames = sum(request.frames)

        video_request = VideoGenerationRequest(
            prompt=combined_prompt,
            model_name=request.model_name,
            nb_frames=total_frames,
            steps=request.num_inference_steps,
            fps=request.fps,
            seed=request.seed if request.seed is not None else -1
        )
        return self.generate_video(video_request)

service = VideoGenService()

# API Endpoints
@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {
        "status": "healthy",
        "default_model": service.default_model,
        "loaded_models": list(service.pipelines.keys()),
        "gpu_available": service.use_gpu
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": list(service.models.keys())}

@app.post("/generate")
async def generate_video(request: VideoGenerationRequest):
    """Generate a video from a single prompt"""
    try:
        output_path = service.generate_video(request)
        return {"video_path": output_path, "message": "Video generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_multi")
async def generate_multi_prompt_video(request: MultiPromptVideoRequest):
    """Generate a video from multiple prompts"""
    try:
        output_path = service.generate_video_by_frames(request)
        return {"video_path": output_path, "message": "Video generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
