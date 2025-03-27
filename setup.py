from setuptools import setup, find_packages

# Read the README.md file with UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="openvideogen",
    version="0.1.0",
    description="Open source video generation API using diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ParisNeo",
    author_email="parisneo_ai@gmail.com",
    url="https://github.com/ParisNeo/OpenVideoGen",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "imageio-ffmpeg",
        "sentencepiece",
        "toml",
        "pipmaster"
    ],
    entry_points={
        "console_scripts": [
            "openvideogen=openvideogen.main:app"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)