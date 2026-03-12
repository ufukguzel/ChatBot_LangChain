"""Centralised logging configuration."""
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger.  Call once at application startup."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Silence noisy third-party loggers in production
    for noisy in ("httpcore", "httpx", "openai", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
