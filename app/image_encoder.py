from PIL import Image
import torch
import numpy as np
from PIL import UnidentifiedImageError
from pathlib import Path
from typing import Union
from app.clip_model import get_clip_manager
from app.utils import validate_image_path


def encode_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Encode an image to a normalized CLIP embedding vector.

    Args:
        image_path: Path to image file (str or Path object)

    Returns:
        Normalized embedding as numpy array of shape (512,)

    Raises:
        FileNotFoundError: If image path doesn't exist
        ValueError: If image cannot be opened
    """
    path = validate_image_path(image_path)

    manager = get_clip_manager()
    processor = manager.get_processor()
    model = manager.get_model()
    device = manager.get_device()

    try:
        with Image.open(path) as img:
            image = img.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
    except UnidentifiedImageError as e:
        raise ValueError(f"Cannot decode image: {path}") from e
    except Exception as e:
        raise ValueError(f"Failed to process image {path}: {e}") from e

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

        # Defensive: handle object outputs from version/runtime mismatch
        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            elif hasattr(image_features, "last_hidden_state"):
                image_features = image_features.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported image feature output type: {type(image_features)}")

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).cpu().numpy().astype(np.float32)