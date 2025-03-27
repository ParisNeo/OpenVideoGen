from setuptools import setup, find_packages

setup(
    name="openvideogen",
    version="0.1.0",
    description="Open source video generation API using diffusion models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ParisNeo",
    author_email="your-email@example.com",  # Replace with your email
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
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
