**OpenVideoGen**

Open source video generation API using various diffusion models.

**Features**

- Generate videos from text prompts using state-of-the-art diffusion models
- Support for multiple models (CogVideoX, Stable Video Diffusion, and more)
- Configurable via `config.toml`
- FastAPI-based RESTful API with asynchronous job processing
- Job status checking and file downloading
- Automatic file purging after a configurable time
- Logging for debugging and monitoring
- Easy installation via pip
- Ubuntu systemd service support
- Docker integration

**Installation**

**Prerequisites**

- Python 3.11 or higher
- CUDA-enabled GPU (optional but recommended for faster generation)

**Install via pip**

```bash
pip install openvideogen
```

**Install from source**

```bash
git clone https://github.com/ParisNeo/OpenVideoGen.git
cd OpenVideoGen
pip install .
```

**Usage**

```bash
uvicorn openvideogen.main:app --host 0.0.0.0 --port 8088
```

**API Endpoints**

- `GET /health`: Check service status
- `GET /models`: List available models
- `POST /submit`: Submit a video generation job and get a job ID
- `POST /submit_multi`: Submit a multi-prompt video generation job and get a job ID
- `GET /status/{job_id}`: Check the status of a job
- `GET /download/{job_id}`: Download the generated video for a job

**Example Usage**

**Submit a Job**

```bash
curl -X POST "http://localhost:8088/submit" \
-H "Content-Type: application/json" \
-d '{
    "prompt": "A cat playing with a ball",
    "model_name": "cogvideox_2b",
    "height": 480,
    "width": 720,
    "steps": 50,
    "nb_frames": 49,
    "fps": 8
}'
```

**Check Job Status**

```bash
curl "http://localhost:8088/status/123e4567-e89b-12d3-a456-426614174000"
```

**Download the Video**

```bash
curl "http://localhost:8088/download/123e4567-e89b-12d3-a456-426614174000" --output video.mp4
```

**Configuration**

Edit `config.toml` to add or modify models and settings:

```toml
[models]
cogvideox_2b = {name = "THUDM/CogVideoX-2b", type = "cogvideox"}
cogvideox_5b = {name = "THUDM/CogVideoX-5b", type = "cogvideox"}
stable_video = {name = "stabilityai/stable-video-diffusion-img2vid", type = "stablevideo"}

[settings]
default_model = "cogvideox_2b"
use_gpu = true
dtype = "float16"
output_folder = "./outputs"
port = 8088
host = "0.0.0.0"
file_retention_time = 3600  # Files are deleted after 1 hour (3600 seconds)
```

**Setting Up as an Ubuntu Service**

1. Copy the `openvideogen.service` file to `/etc/systemd/system/`.
2. Edit the service file to update the user, working directory, and `ExecStart` path.
3. Reload systemd, enable the service, and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable openvideogen.service
sudo systemctl start openvideogen.service
```

**Docker Integration**

1. Build the Docker image:

```bash
docker build -t openvideogen:latest .
```

2. Run the Docker container:

```bash
docker run -d \
    -p 8088:8088 \
    -v $(pwd)/outputs:/app/outputs \
    --name openvideogen \
    openvideogen:latest
```

**Supported Models**

- `cogvideox_2b`: CogVideoX 2B model
- `cogvideox_5b`: CogVideoX 5B model
- `stable_video`: Stable Video Diffusion

To add more models, update the `config.toml` file with the model name and type.

**Contributing**

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

**License**

This project is licensed under the Apache 2.0 License - see the `LICENSE` file for details.
