**OpenVideoGen**

Open source video generation API using various diffusion models.

## Features
- Generate videos from text prompts using state-of-the-art diffusion models
- Support for multiple models (CogVideoX, Stable Video Diffusion, and more)
- Configurable via `config.toml`
- FastAPI-based RESTful API
- Logging for debugging and monitoring
- Easy installation via pip

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (optional but recommended for faster generation)

### Install via pip
```bash
pip install openvideogen
```

### Install from source
```bash
git clone https://github.com/ParisNeo/OpenVideoGen.git
cd OpenVideoGen
pip install .
```

## Usage
Run the API
```bash
uvicorn openvideogen.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints
- GET `/health`: Check service status
- GET `/models`: List available models
- POST `/generate`: Generate video from a single prompt
- POST `/generate_multi`: Generate video from multiple prompts

## Example Request
Generate a video with a single prompt:
```bash
curl -X POST "http://localhost:8000/generate" \
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

## Configuration
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
