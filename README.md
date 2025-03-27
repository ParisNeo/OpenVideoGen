<img src="assets/icon.png" alt="OpenVideoGen Logo" width="32"/> OpenVideoGen

[![License](https://img.shields.io/github/license/ParisNeo/OpenVideoGen)](https://github.com/ParisNeo/OpenVideoGen/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![GitHub stars](https://img.shields.io/github/stars/ParisNeo/OpenVideoGen?style=social)](https://github.com/ParisNeo/OpenVideoGen/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/ParisNeo/OpenVideoGen)](https://github.com/ParisNeo/OpenVideoGen/issues)
[![Discord](https://img.shields.io/discord/1818856788?color=%237289DA&label=Discord&logo=discord&logoColor=white)](https://discord.gg/yourserverlinkhere)

Open source video generation API using various diffusion models.

## Features

- 🎬 Generate videos from text prompts using state-of-the-art diffusion models
- 🔄 Support for multiple models (CogVideoX, Stable Video Diffusion, and more)
- ⚙️ Configurable via `config.toml` with flexible search paths
- 🚀 FastAPI-based RESTful API with asynchronous job processing
- 📊 Job status checking and file downloading
- 🧹 Automatic file purging after a configurable time
- 🔧 Enhanced control over GPU usage and generation parameters
- 📝 Logging for debugging and monitoring
- 📦 Easy installation via pip
- 🐧 Ubuntu systemd service support
- 🐳 Docker integration

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-enabled GPU (optional but recommended for faster generation)

### Install via pip

```bash
pip install openvideogen
```

### Install from source

Clone the repository:

```bash
git clone https://github.com/ParisNeo/OpenVideoGen.git
cd OpenVideoGen
```

Install dependencies:

```bash
pip install .
```

## Usage

### Run the API

```bash
uvicorn openvideogen.main:app --host 0.0.0.0 --port 8088
```

You can specify a custom config file using the `--config` argument:

```bash
uvicorn openvideogen.main:app --host 0.0.0.0 --port 8088 --config /path/to/custom_config.toml
```

Alternatively, set the `OPENVIDEOGEN_CONFIG` environment variable:

```bash
export OPENVIDEOGEN_CONFIG=/path/to/custom_config.toml
uvicorn openvideogen.main:app --host 0.0.0.0 --port 8088
```

### Config File Search Paths

The API searches for `config.toml` in the following locations, in order of priority:

1. Path specified via the `--config` command-line argument.
2. Path specified via the `OPENVIDEOGEN_CONFIG` environment variable.
3. System-specific locations:

   **Linux:**
   - `/etc/openvideogen/config.toml`
   - `/usr/local/etc/openvideogen/config.toml`
   - `~/.config/openvideogen/config.toml`
   - `./config.toml` (current working directory)

   **Windows:**
   - `%APPDATA%/openvideogen/config.toml`
   - `./config.toml` (current working directory)

   **macOS:**
   - `~/Library/Application Support/openvideogen/config.toml`
   - `/usr/local/etc/openvideogen/config.toml`
   - `./config.toml` (current working directory)

If no config file is found, a default `config.toml` is created in the current working directory.

## API Endpoints

- `GET /health`: Check service status
- `GET /models`: List available models
- `POST /submit`: Submit a video generation job and get a job ID
- `POST /submit_multi`: Submit a multi-prompt video generation job and get a job ID
- `GET /status/{job_id}`: Check the status of a job
- `GET /download/{job_id}`: Download the generated video for a job

## Example Usage

### Submit a Job

Submit a video generation job with custom generation parameters:

```bash
curl -X POST "http://localhost:8088/submit" \
-H "Content-Type: application/json" \
-d '{
    "prompt": "A cat playing with a ball",
    "model_name": "cogvideox_2b",
    "height": 480,
    "width": 720,
    "steps": 75,
    "guidance_scale": 7.5,
    "nb_frames": 49,
    "fps": 8
}'
```

Response:

```json
{
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "message": "Job submitted successfully"
}
```

### Check Job Status

Check the status of the job:

```bash
curl "http://localhost:8088/status/123e4567-e89b-12d3-a456-426614174000"
```

Response (when completed):

```json
{
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "completed",
    "message": "Video generated successfully",
    "video_url": "/download/123e4567-e89b-12d3-a456-426614174000",
    "created_at": 1696112345.123,
    "expires_at": 1696115945.123
}
```

### Download the Video

Download the generated video:

```bash
curl "http://localhost:8088/download/123e4567-e89b-12d3-a456-426614174000" --output video.mp4
```

## Configuration

Edit `config.toml` to add or modify models and settings:

```toml
[models]
cogvideox_2b = {name = "THUDM/CogVideoX-2b", type = "cogvideox"}
cogvideox_5b = {name = "THUDM/CogVideoX-5b", type = "cogvideox"}
stable_video = {name = "stabilityai/stable-video-diffusion-img2vid", type = "stablevideo"}

[settings]
default_model = "cogvideox_2b"
force_gpu = false  # Force GPU usage even if use_gpu is false
use_gpu = true     # Use GPU if available
dtype = "float16"
output_folder = "./outputs"
port = 8088
host = "0.0.0.0"
file_retention_time = 3600  # Files are deleted after 1 hour (3600 seconds)

[generation]
guidance_scale = 6.0
num_inference_steps = 50
```

- `force_gpu`: Set to true to force GPU usage, even if `use_gpu` is false. Will raise an error if no GPU is available.
- `use_gpu`: Set to true to use GPU if available, unless `force_gpu` is true.
- `guidance_scale`: Controls the influence of the prompt on the generation.
- `num_inference_steps`: Number of inference steps for generation.

## Setting Up as an Ubuntu Service

You can run OpenVideoGen as a systemd service on Ubuntu for automatic startup and management.

### Steps

1. **Copy the Service File:**

   Copy the provided `openvideogen.service` file to `/etc/systemd/system/`:

   ```bash
   sudo cp openvideogen.service /etc/systemd/system/openvideogen.service
   ```

2. **Edit the Service File:**

   Open the service file for editing:

   ```bash
   sudo nano /etc/systemd/system/openvideogen.service
   ```

   Update the following fields:
   - `User`: Replace `your-username` with your actual username.
   - `WorkingDirectory`: Replace `/path/to/OpenVideoGen` with the absolute path to your OpenVideoGen directory.
   - `ExecStart`: Replace `/path/to/venv/bin/uvicorn` with the path to your virtual environment's uvicorn binary (e.g., `/home/your-username/venv/bin/uvicorn`).
   - Optionally, update the `--port` value if you changed it in `config.toml`.

   Example:

   ```ini
   [Unit]
   Description=OpenVideoGen FastAPI Service
   After=network.target

   [Service]
   User=your-username
   WorkingDirectory=/home/your-username/OpenVideoGen
   ExecStart=/home/your-username/venv/bin/uvicorn openvideogen.main:app --host 0.0.0.0 --port 8088
   Restart=always
   Environment="PYTHONUNBUFFERED=1"

   [Install]
   WantedBy=multi-user.target
   ```

3. **Reload Systemd and Enable the Service:**

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable openvideogen.service
   sudo systemctl start openvideogen.service
   ```

4. **Check the Service Status:**

   ```bash
   sudo systemctl status openvideogen.service
   ```

5. **View Logs:**

   ```bash
   journalctl -u openvideogen.service -b
   ```

## Docker Integration

You can also run OpenVideoGen in a Docker container.

### Prerequisites

- Docker installed on your system

### Build the Docker Image

Build the image:

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
