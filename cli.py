#!/usr/bin/env python3
"""
CLI entry point for LangChain Chatbot.

Usage
-----
  python cli.py                      # interactive mode
  python cli.py --mock               # force mock mode
  python cli.py --model gpt-4o       # choose model
  python cli.py --temperature 0.7
"""
import argparse
import sys

from app.core.config import settings
from app.core.logging_setup import setup_logging
from app.services.chat_engine import ChatEngine, _LC_AVAILABLE

setup_logging(settings.log_level)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangChain Chatbot — CLI")
    p.add_argument("--model",       default=settings.model_name,  help="LLM model identifier")
    p.add_argument("--temperature", default=settings.temperature,  type=float)
    p.add_argument("--mock",        action="store_true",           help="Force mock mode")
    return p.parse_args(argv)


def run_cli(args: argparse.Namespace) -> None:
    api_key = settings.openai_api_key
    engine = ChatEngine(
        model_name=args.model,
        temperature=args.temperature,
        api_key=api_key,
        force_mock=args.mock or settings.force_mock or (api_key is None),
    )

    print("🤖 LangChain Chatbot (CLI) — /exit ile çıkabilirsiniz\n")
    if not _LC_AVAILABLE:
        print("[i] LangChain paketleri bulunamadı — mock modda çalışıyor.")
    elif not api_key and not args.mock:
        print("[i] OPENAI_API_KEY bulunamadı — mock modda çalışıyor.")
    elif args.mock or settings.force_mock:
        print("[i] Zorunlu mock modu etkin.")
    else:
        print(f"[i] LangChain modu: model={args.model}, temperature={args.temperature}")

    while True:
        try:
            user = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor…")
            break

        if user in {"/exit", ":q", "quit", "exit"}:
            print("Görüşürüz! 👋")
            break
        if not user:
            continue

        reply = engine.send(user)
        print(f"🤖 Bot: {reply}\n")


def main() -> None:
    args = parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
