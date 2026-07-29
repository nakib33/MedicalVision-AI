#!/usr/bin/env python3
"""MedicalVision AI Suite — Application Launcher.

Usage:
    python run.py              # Start server (default: 0.0.0.0:8000)
    python run.py --port 8080  # Custom port
    python run.py --host 127.0.0.1  # Custom host
"""

import argparse
import uvicorn
from shared.config import HOST, PORT


def main():
    parser = argparse.ArgumentParser(
        description="MedicalVision AI Suite — 12 Medical Imaging Projects"
    )
    parser.add_argument("--host", type=str, default=HOST, help="Host address")
    parser.add_argument("--port", type=int, default=PORT, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  MedicalVision AI Suite")
    print(f"  16 CNN-Based Medical Imaging Projects with XAI")
    print(f"  PyTorch + FastAPI + Explainable AI")
    print("=" * 60)
    print()
    print(f"  http://{args.host}:{args.port}")
    print(f"  API docs at http://{args.host}:{args.port}/docs")
    print(f"  Projects loaded: {len([d for d in __import__('os').listdir('projects') if d[0].isdigit()])}")
    print()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
