#!/usr/bin/env python3
"""
Web server entry point.

Usage
-----
Development (with auto-reload):
  python server.py --reload

Production:
  python server.py --host 0.0.0.0 --port 8000

Or via uvicorn directly:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
"""
import argparse

import uvicorn

from app.core.config import settings


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangChain Chatbot — Web Server")
    p.add_argument("--host",   default=settings.host,            help="Bind host")
    p.add_argument("--port",   default=settings.port, type=int,  help="Bind port")
    p.add_argument("--reload", action="store_true",               help="Enable auto-reload (dev only)")
    p.add_argument("--workers", default=1,            type=int,  help="Number of worker processes")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
