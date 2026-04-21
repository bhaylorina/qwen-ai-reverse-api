#!/usr/bin/env python3
"""Start script for Qwen AI API Server"""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen AI OpenAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)
