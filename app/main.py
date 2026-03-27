"""
Main Entry Point
Demonstrates image-text retrieval with a sample workflow.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from app.image_encoder import encode_image
from app.text_encoder import encode_text
from app.similarity import compute_similarity
from app.utils import validate_image_path


def _list_images(image_dir: Path) -> List[Path]:
    """Return supported image files in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    if not image_dir.exists() or not image_dir.is_dir():
        return []
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MultiScope image-text similarity demo")
    parser.add_argument("--image", type=str, default="data/images/sample.jpg", help="Path to input image")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images for retrieval mode")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K results for retrieval mode")
    parser.add_argument(
        "--text",
        nargs="+",
        default=["a dog running in a park", "a cat sleeping on a couch", "a person coding at a desk"],
        help="One or more text queries",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        # retrieval mode (query vs folder)
        if args.image_dir:
            image_dir = Path(args.image_dir)
            files = _list_images(image_dir)
            if not files:
                raise FileNotFoundError(f"No images found in directory: {image_dir}")

            query = " ".join(args.text).strip()
            if not query:
                raise ValueError("Please provide a text query with --text")

            text_embedding = encode_text(query)
            scores: List[Tuple[Path, float]] = []

            for fp in files:
                img_emb = encode_image(fp)
                score = float(compute_similarity(img_emb, text_embedding))
                scores.append((fp, score))

            ranked = sorted(scores, key=lambda x: x[1], reverse=True)[: max(1, args.top_k)]

            print("MultiScope — Folder Retrieval")
            print("-" * 50)
            print(f"Query: {query}")
            for i, (fp, s) in enumerate(ranked, start=1):
                print(f"{i:>2}. {s: .4f} | {fp.name}")
            return 0

        # existing single-image mode
        image_path = validate_image_path(Path(args.image))
        image_embedding = encode_image(image_path)
        text_embeddings = encode_text(args.text)
        similarities = compute_similarity(image_embedding, text_embeddings)

        print("MultiScope — Image-Text Retrieval")
        print("-" * 50)
        for text, score in zip(args.text, np.atleast_1d(similarities)):
            print(f"{float(score): .4f} | {text}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())