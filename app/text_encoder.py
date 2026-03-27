"""
Text Encoder
Converts text prompts to normalized CLIP embeddings.
"""

import torch
import numpy as np
from typing import Union, List
from app.clip_model import get_clip_manager
from app.utils import sanitize_text


def encode_text(text: Union[str, List[str]]) -> np.ndarray:
    """
    Encode text to normalized CLIP embedding vector(s).
    
    Args:
        text: Single text string or list of text strings
    
    Returns:
        Normalized embedding as numpy array.
        Shape (512,) for single text, (N, 512) for multiple texts.
    
    Raises:
        ValueError: If text is empty or invalid type
    """
    if isinstance(text, str):
        cleaned = [sanitize_text(text)]
    elif isinstance(text, list) and all(isinstance(t, str) for t in text):
        cleaned = [sanitize_text(t) for t in text]
    else:
        raise ValueError(f"Text must be str or list[str], got {type(text)}")

    cleaned = [t for t in cleaned if t]
    if not cleaned:
        raise ValueError("Text input cannot be empty")

    manager = get_clip_manager()
    processor = manager.get_processor()
    model = manager.get_model()
    device = manager.get_device()
    
    # Process text
    inputs = processor(
        text=cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # Extract features
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

        # Defensive: handle object outputs from version/runtime mismatch
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            elif hasattr(text_features, "last_hidden_state"):
                text_features = text_features.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported text feature output type: {type(text_features)}")

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Normalize embeddings
    result = text_features.cpu().numpy().astype(np.float32)
    return result.squeeze(0) if len(cleaned) == 1 else result
