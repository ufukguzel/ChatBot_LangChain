"""
Backward-compatibility shim — preserves the original CLI interface.

The application has been refactored into the `app/` package.
This file delegates to the new modules so existing usage still works:

  python app.py                   → CLI chatbot
  python app.py --mock            → CLI mock mode
  python app.py --api             → Web server (FastAPI)
  python app.py --run-tests       → pytest suite
  python app.py --api --host X --port Y

Recommended new commands:
  python cli.py           (CLI)
  python server.py        (Web server)
  pytest tests/ -v        (Tests)
"""
import argparse
import sys
import types


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangChain Chatbot (legacy entry point)")
    p.add_argument("--model",       default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--mock",        action="store_true")
    p.add_argument("--run-tests",   action="store_true")
    p.add_argument("--api",         action="store_true")
    p.add_argument("--host",        default="127.0.0.1")
    p.add_argument("--port",        type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.run_tests:
        import pytest
        sys.exit(pytest.main(["tests/", "-v"]))

    elif args.api:
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")

    else:
        cli_args = types.SimpleNamespace(
            model=args.model,
            temperature=args.temperature,
            mock=args.mock,
        )
        from cli import run_cli
        run_cli(cli_args)
