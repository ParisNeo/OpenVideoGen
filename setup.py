import os
import re
from setuptools import setup, find_packages

# Function to extract version from __init__.py
def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py_path = os.path.join(os.path.dirname(__file__), package, '__init__.py')
    if not os.path.exists(init_py_path):
        raise RuntimeError(f"{init_py_path} does not exist")

    with open(init_py_path, "r", encoding="utf-8") as f:
        # Look for __version__ = "x.y.z"
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE)
        if match:
            return match.group(1)
        raise RuntimeError("Unable to find version string.")

# Read the README.md file with UTF-8 encoding
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Open source video generation API using diffusion models" # Fallback

# Package name
PACKAGE_NAME = "openvideogen"

setup(
    name=PACKAGE_NAME,
    version=get_version(PACKAGE_NAME), # Get version dynamically
    description="Open source video generation API using diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ParisNeo",
    author_email="parisneo_ai@gmail.com",
    url="https://github.com/ParisNeo/OpenVideoGen",
    # Find packages automatically in the 'openvideogen' directory
    packages=find_packages(include=[PACKAGE_NAME, f"{PACKAGE_NAME}.*"]),
    # Include static files (web UI, assets)
    include_package_data=False, # Explicitly set to False, we use package_data
    package_data={
        PACKAGE_NAME: [
            'static/*',
            'static/assets/*',
        ],
    },
    install_requires=[
        "fastapi>=0.90.0",
        "uvicorn[standard]>=0.20.0", # Use [standard] for extras
        "pydantic>=1.10.0,<2.0.0", # Example: Pin Pydantic v1 if needed
        # "pydantic>=2.0.0", # Or use Pydantic v2
        # Torch needs careful installation (CPU/GPU specific).
        # Listing it here ensures pip knows about it, but users often
        # need to install it separately following PyTorch.org instructions.
        "torch>=2.0.0",
        "torchvision",
        "torchaudio",
        "diffusers>=0.20.0", # Use a recent version
        "transformers>=4.25.0",
        "accelerate>=0.25.0",
        "imageio>=2.20.0",
        "imageio-ffmpeg>=0.4.5",
        "sentencepiece",
        "toml>=0.10.0",
        "opencv-python>=4.5", # Added from script
        "pipmaster>=0.1.0", # Added from script
        # Add any other dependencies your cli.py or main.py might use
        # For example, if cli.py uses typer:
        # "typer[all]"
    ],
    entry_points={
        # This assumes you have a main function (e.g., called 'main' or 'run_cli')
        # inside your openvideogen/cli.py file that starts the application.
        "console_scripts": [
            "openvideogen=openvideogen.cli:main" # CHANGE 'main' if your function name is different
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
