"""
Utility Functions
Helper functions for file validation and preprocessing.
"""

from pathlib import Path
from typing import List, Union


def validate_image_path(image_path: Union[str, Path]) -> Path:
    """
    Validate image file exists and has valid extension.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If not a valid image format
    """
    path = Path(image_path)

    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid image format: {path.suffix}. Allowed: {sorted(valid_extensions)}")

    return path


def sanitize_text(text: str) -> str:
    """Clean text input while preserving casing/semantics."""
    return text.strip()


def parse_text_queries(text_input: str) -> List[str]:
    """Parse multiline text into non-empty cleaned queries."""
    queries = [sanitize_text(t) for t in text_input.splitlines()]
    return [q for q in queries if q]
