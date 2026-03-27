"""
CLIP Model Manager
Handles model and processor initialization with device management.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from typing import Tuple


class CLIPModelManager:
    """Singleton-like manager for CLIP model and processor."""
    
    _instance = None
    _model = None
    _processor = None
    _device = None
    
    MODEL_NAME = "openai/clip-vit-base-patch32"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize CLIP model and processor on first instantiation."""
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(
                self.MODEL_NAME,
                use_fast=False,  # keep legacy/slow processor behavior
            )
            self._model.eval()  # Set to evaluation mode
    
    def get_model(self) -> CLIPModel:
        """Return the CLIP model."""
        return self._model
    
    def get_processor(self) -> CLIPProcessor:
        """Return the CLIP processor."""
        return self._processor
    
    def get_device(self) -> torch.device:
        """Return the device (cuda or cpu)."""
        return self._device
    
    def get_model_info(self) -> dict:
        """Return model configuration info."""
        return {
            "model_name": self.MODEL_NAME,
            "device": str(self._device),
            "dtype": next(self._model.parameters()).dtype
        }


def get_clip_manager() -> CLIPModelManager:
    """Factory function to get or create CLIPModelManager instance."""
    return CLIPModelManager()