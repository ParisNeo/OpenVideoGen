# OpenVideoGen

#
# File: README.md
# Author: parisneo
# Description: Project Readme file for OpenVideoGen.
# Date: 10/04/2025
#

<div align="center">
  <img src="https://github.com/ParisNeo/OpenVideoGen/blob/main/assets/openvideogen-icon.png" alt="Logo" width="200" height="200"> <!-- Updated path -->
</div>

[![License](https://img.shields.io/github/license/ParisNeo/OpenVideoGen)](https://github.com/ParisNeo/OpenVideoGen/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![GitHub stars](https://img.shields.io/github/stars/ParisNeo/OpenVideoGen?style=social)](https://github.com/ParisNeo/OpenVideoGen/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/ParisNeo/OpenVideoGen)](https://github.com/ParisNeo/OpenVideoGen/issues)
[![Discord](https://img.shields.io/discord/1354804454549094420?color=%237289DA&label=Discord&logo=discord&logoColor=white)](https://discord.com/channels/1354804454549094420) <!-- Updated Discord link -->

Open source video generation API using various diffusion models, supporting both **Text-to-Video** and **Image-to-Video** generation.

## Features

- 🎬 **Text-to-Video (T2V):** Generate videos from text prompts.
- 🖼️ **Image-to-Video (I2V):** Generate videos starting from an input image (e.g., using Stable Video Diffusion).
- 🔄 Support for multiple models (see [Supported Models](#supported-models)).
- ✨ Simple Web UI with tabs for T2V, I2V, and Job Status.
- ⚙️ Configurable via `config.toml` with flexible search paths.
- 🚀 FastAPI-based RESTful API with asynchronous job processing.
- 📊 Job status checking and video downloading/previewing.
- 🧹 Automatic file purging after a configurable time.
- 🔧 Control over GPU usage, data types (`float16`/`bfloat16`), and generation parameters.
- 📝 Logging for debugging and monitoring (`openvideogen.log`).
- 📦 Easy installation via pip.
- 🐧 Ubuntu systemd service support (`openvideogen.service`).
- 🐳 Docker integration (`Dockerfile`).

## Installation

### Prerequisites

- Python 3.11 or higher
- `git` (required for installing `diffusers` main branch sometimes)
- `ffmpeg` installed and available in your system PATH.
    - **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
    - **macOS:** `brew install ffmpeg`
- CUDA-enabled GPU (strongly recommended for performance). Check PyTorch compatibility ([pytorch.org](https://pytorch.org/)).

### Install via pip

```bash
pip install openvideogen
```
*(Note: This might install the latest stable release from PyPI. For the absolute latest features/fixes, install from source.)*

### Install from source (Recommended for latest features)

Clone the repository:
```bash
git clone https://github.com/ParisNeo/OpenVideoGen.git
cd OpenVideoGen
```

Install dependencies (this will also install OpenVideoGen in editable mode):
```bash
pip install -e .
```
*(This uses `setup.py` which reads `requirements.txt` and installs necessary packages, including PyTorch, diffusers, etc.)*

The installation process uses `pipmaster` to check and install dependencies, including PyTorch with CUDA support if available.

## Usage

### Running the API Server

You can run the server using the installed command-line tool:

```bash
openvideogen
```

This command uses the `cli.py` script and is equivalent to running:
```bash
uvicorn openvideogen.main:app --host 0.0.0.0 --port 8088
```

**Command-line options:**

- `--host <ip>`: Host to bind to (default: `0.0.0.0`).
- `--port <number>`: Port to listen on (default: `8088`).
- `--config <path>`: Path to a custom `config.toml` file.
- `--reload`: Enable auto-reload for development (requires `watchfiles`).

**Example with options:**
```bash
openvideogen --port 9000 --config /path/to/my_config.toml --reload
```

Alternatively, set the `OPENVIDEOGEN_CONFIG` environment variable:
```bash
export OPENVIDEOGEN_CONFIG=/path/to/custom_config.toml
openvideogen
```

### Accessing the Web UI

Once the server is running, open your web browser and navigate to: `http://<your-server-ip>:8088/webui` (or the configured host/port).

### Config File Search Paths

The API searches for `config.toml` in the following locations (priority order):
1. Path from `--config` CLI argument.
2. Path from `OPENVIDEOGEN_CONFIG` environment variable.
3. System-specific locations (Linux: `/etc`, `/usr/local/etc`, `~/.config`; Windows: `%APPDATA%`, `%PROGRAMDATA%`; macOS: `~/Library/Application Support`, `/Library/Application Support`, `/usr/local/etc`).
4. Current working directory (`./config.toml`).
5. Project root directory (if installed from source).

If no config file is found, a default `config.toml` is created in the current working directory. An `example_config.toml` with detailed comments is also included in the repository.

## API Endpoints

- `GET /health`: Check service status and configuration.
- `GET /models`: List available models from config and their status.
- `POST /submit`: Submit a **Text-to-Video** job.
- `POST /submit_image_video`: Submit an **Image-to-Video** job (requires image file upload).
- `POST /submit_multi`: Submit a multi-prompt T2V job (simple concatenation).
- `GET /status/{job_id}`: Check the status of a job.
- `GET /download/{job_id}`: Download the generated video for a completed job.

Access the interactive API documentation (Swagger UI) at `/docs`.

## Example API Usage

### Submit a Text-to-Video Job

```bash
curl -X POST "http://localhost:8088/submit" \
-H "Content-Type: application/json" \
-d '{
    "prompt": "Astronaut riding a horse on the moon, cinematic",
    "model_name": "wan_1_3b",
    "negative_prompt": "low quality, blurry",
    "nb_frames": 60,
    "fps": 12,
    "steps": 50,
    "guidance_scale": 7.0
}'
```

### Submit an Image-to-Video Job (e.g., SVD)

```bash
curl -X POST "http://localhost:8088/submit_image_video" \
-F "image=@/path/to/your/input_image.png" \
-F "model_name=stable_video_xt_1_1" \
-F "motion_bucket_id=150" \
-F "noise_aug_strength=0.05" \
-F "fps=10"
```
*(Note: Use `-F` for form data when uploading files.)*

### Check Job Status

```bash
curl "http://localhost:8088/status/<job_id_from_submit>"
```

### Download the Video

```bash
curl "http://localhost:8088/download/<job_id_from_submit>" --output video.mp4
```

## Configuration (`config.toml`)

```toml
[models]
# Define models available to the API
# 'name': Hugging Face repository ID
# 'type': Internal type used for loading logic (e.g., "cogvideox", "wan", "img2vid", "hunyuan", "ltx", "animatediff", "mochi")
# 'variant': (Optional) Specify model variant (e.g., "fp16", "bf16")
# 'adapter': (Optional, required for 'animatediff') HF repo ID of the motion adapter
# --- T2V ---
wan_1_3b = { name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", type = "wan" }
hunyuan = { name = "hunyuanvideo-community/HunyuanVideo", type = "hunyuan" }
# --- I2V ---
stable_video_xt_1_1 = {name = "stabilityai/stable-video-diffusion-img2vid-xt-1-1", type = "img2vid"}

[settings]
default_t2v_model = "wan_1_3b"        # Default for /submit
default_i2v_model = "stable_video_xt_1_1" # Default for /submit_image_video
force_gpu = false                     # Error if no GPU when true
use_gpu = true                        # Use GPU if available
dtype = "bfloat16"                    # "float16" or "bfloat16"
output_folder = "./outputs"           # Where videos are saved
model_cache_dir = "./model_cache"     # HF cache location
port = 8088
host = "0.0.0.0"
file_retention_time = 86400           # Cleanup after 24 hours (seconds)

[generation]
# Default parameters, can be overridden per request or per model type
# T2V Defaults
default_guidance_scale = 6.0
default_num_inference_steps = 50
# Model-specific T2V (e.g., wan_default_height = 480)
# ...
# I2V Defaults (e.g., img2vid_default_fps = 7)
img2vid_default_height = 576
img2vid_default_width = 1024
# ... SVD specific ...
img2vid_motion_bucket_id = 127
img2vid_noise_aug_strength = 0.02
```

## Setting Up as an Ubuntu Service

Use the provided `openvideogen.service` file to run OpenVideoGen as a systemd service.

1.  **Copy & Edit:**
    ```bash
    sudo cp openvideogen.service /etc/systemd/system/
    sudo nano /etc/systemd/system/openvideogen.service
    ```
    Update `User`, `WorkingDirectory`, and `ExecStart` (use the path to `openvideogen` command or `uvicorn` in your venv). Ensure the user has permissions for the working directory, outputs, and model cache.

2.  **Enable & Start:**
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable openvideogen.service
    sudo systemctl start openvideogen.service
    ```

3.  **Check Status & Logs:**
    ```bash
    sudo systemctl status openvideogen.service
    journalctl -u openvideogen.service -f
    ```

## Docker Integration

Use the provided `Dockerfile` to build and run OpenVideoGen in a container.

1.  **Build:**
    ```bash
    docker build -t openvideogen:latest .
    ```

### Run the Docker Container

Run the container, mapping the port and mounting the output directory:

```bash
docker run -d \
    -p 8088:8088 \
    -v $(pwd)/outputs:/app/outputs \
    --name openvideogen \
    openvideogen:latest
```

- `-p 8088:8088`: Maps the container's port 8088 to the host's port 8088 (adjust if you changed the port in config.toml).
- `-v $(pwd)/outputs:/app/outputs`: Mounts the local outputs directory to the container's /app/outputs directory for persistent storage of generated videos.

To use a custom config file, mount it into the container:

```bash
docker run -d \
    -p 8088:8088 \
    -v $(pwd)/outputs:/app/outputs \
    -v /path/to/custom_config.toml:/app/config.toml \
    --name openvideogen \
    openvideogen:latest
```

### Check the Container Logs

```bash
docker logs openvideogen
```

### Stop the Container

```bash
docker stop openvideogen
docker rm openvideogen
```

## Supported Models

- `cogvideox_2b`: CogVideoX 2B model
- `cogvideox_5b`: CogVideoX 5B model
- `stable_video`: Stable Video Diffusion

To add more models, update the `config.toml` file with the model name and type.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ParisNeo/OpenVideoGen&type=Date)](https://star-history.com/#ParisNeo/OpenVideoGen&Date)
