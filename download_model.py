#!/usr/bin/env python3
"""
Download a sentence-transformer model into this repo's local cache.
"""

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "embeddings_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local sentence-transformer model cache")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Base cache directory (default: ./embeddings_cache)",
    )
    args = parser.parse_args()

    model_dir = Path(args.base_dir).expanduser().resolve() / "model" / args.model_name
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/loading model '{args.model_name}' into: {model_dir}")
    model = SentenceTransformer(args.model_name)
    model.save(str(model_dir))
    print("Model saved locally.")


if __name__ == "__main__":
    main()