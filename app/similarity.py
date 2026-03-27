"""
Similarity Computation
Calculates cosine similarity between image and text embeddings.
"""

import numpy as np
from typing import Union


def compute_similarity(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Compute cosine similarity between image and text embedding(s).

    Returns:
        float for a single text embedding, np.ndarray for batched text embeddings.
        Cosine similarity range is [-1.0, 1.0].
    """
    image_embedding = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
    text_embedding = np.asarray(text_embedding, dtype=np.float32)

    if image_embedding.shape[0] != 512:
        raise ValueError(f"Image embedding must have shape (512,), got {image_embedding.shape}")

    # Defensive normalization (idempotent if already normalized)
    image_norm = np.linalg.norm(image_embedding)
    if image_norm == 0:
        raise ValueError("Image embedding norm is zero")
    image_embedding = image_embedding / image_norm

    if text_embedding.ndim == 1:
        if text_embedding.shape[0] != 512:
            raise ValueError(f"Text embedding must have shape (512,), got {text_embedding.shape}")
        text_norm = np.linalg.norm(text_embedding)
        if text_norm == 0:
            raise ValueError("Text embedding norm is zero")
        text_embedding = text_embedding / text_norm
        return float(np.dot(image_embedding, text_embedding))

    if text_embedding.ndim == 2:
        if text_embedding.shape[1] != 512:
            raise ValueError(f"Text embeddings must have shape (N, 512), got {text_embedding.shape}")
        text_norms = np.linalg.norm(text_embedding, axis=1, keepdims=True)
        if np.any(text_norms == 0):
            raise ValueError("One or more text embeddings have zero norm")
        text_embedding = text_embedding / text_norms
        return np.matmul(text_embedding, image_embedding)

    raise ValueError(f"Text embedding must be 1D or 2D, got {text_embedding.ndim}D")
