from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import time
from pathlib import Path
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from ascii_colors import ASCIIColors
import pipmaster as pm

# Install dependencies
def install_dependencies():
    if not pm.is_installed("torch"):
        pm.install_multiple(["torch", "torchvision", "torchaudio"], "https://download.pytorch.org/whl/cu124")
    for package in ["diffusers", "transformers", "accelerate", "imageio-ffmpeg", "sentencepiece"]:
        if not pm.is_installed(package):
            pm.install(package)
    if not pm.is_version_higher("accelerate", "0.26.0"):
        pm.install("accelerate")

install_dependencies()

app = FastAPI(title="CogVideoX API", description="Text-to-Video Generation Service")

# Pydantic models for request validation
class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = "THUDM/CogVideoX-2b"
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
    fps: int = 8
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    seed: Optional[int] = None

# Global pipeline instance
class CogVideoXService:
    def __init__(self):
        self.model_name = "THUDM/CogVideoX-2b"
        self.use_gpu = torch.cuda.is_available()
        self.dtype = torch.float16
        self.output_folder = Path("./outputs")
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.pipeline = self.load_pipeline()

    def load_pipeline(self):
        try:
            pipeline = CogVideoXPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype
            )
            if self.use_gpu:
                pipeline.to("cuda")
                pipeline.enable_model_cpu_offload()
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline: {str(e)}")

    def generate_video(self, request: VideoGenerationRequest) -> str:
        if request.model_name != self.model_name:
            self.model_name = request.model_name
            self.pipeline = self.load_pipeline()

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
            start_time = time.time()
            video_frames = self.pipeline(**gen_params).frames[0]
            output_filename = output_path / f"video_{int(time.time())}.mp4"
            export_to_video(video_frames, str(output_filename), fps=request.fps)
            elapsed_time = time.time() - start_time
            ASCIIColors.success(f"Video generated in {elapsed_time:.2f}s")
            return str(output_filename)
        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")

    def generate_video_by_frames(self, request: MultiPromptVideoRequest) -> str:
        if not request.prompts or not request.frames:
            raise ValueError("Prompts and frames lists cannot be empty")
        
        combined_prompt = " ".join(request.prompts)
        total_frames = sum(request.frames)
        
        video_request = VideoGenerationRequest(
            prompt=combined_prompt,
            nb_frames=total_frames,
            steps=request.num_inference_steps,
            fps=request.fps,
            seed=request.seed if request.seed is not None else -1
        )
        return self.generate_video(video_request)

service = CogVideoXService()

# API Endpoints
@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {"status": "healthy", "model": service.model_name, "gpu_available": service.use_gpu}

@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": ["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"]}

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
