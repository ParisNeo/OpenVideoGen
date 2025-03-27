import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="OpenVideoGen API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8088, help="Port to bind the server to (default: 8088)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--config", type=str, help="Path to the config.toml file")
    args = parser.parse_args()

    # If a config file is specified, set the OPENVIDEOGEN_CONFIG environment variable
    if args.config:
        import os
        os.environ["OPENVIDEOGEN_CONFIG"] = args.config

    # Run the Uvicorn server programmatically
    uvicorn.run(
        app="openvideogen.main:app",  # The FastAPI app
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()