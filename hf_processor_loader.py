"""
Helpers for loading Hugging Face vision encoders and their image processors.

The training scripts assume two utilities:
* ``load_auto_model`` – returns a torch.nn.Module encoder given a checkpoint
* ``load_image_processor`` – returns a callable that prepares PIL images

This module keeps the logic in one place and adds a lightweight fallback
processor so the training script can still run when a pretrained processor
is not available (e.g., for custom/timm checkpoints stored locally).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

try:  # Pillow < 10 compatibility
    Resampling = Image.Resampling
except AttributeError:  # pragma: no cover
    Resampling = Image

try:  # The try/except lets us fail gracefully if transformers is missing.
    from transformers import (
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModel,
        AutoProcessor,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'transformers' package is required to run this project. "
        "Install it via `pip install transformers`."
    ) from exc


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """Return a path string that AutoModel can understand."""
    path = Path(checkpoint).expanduser()
    if path.exists():
        return str(path)
    return checkpoint


@lru_cache(maxsize=None)
def load_auto_model(checkpoint: str):
    """
    Load a vision encoder backbone.

    The function memoizes the result so repeated calls do not re-download the
    weights. ``trust_remote_code`` is enabled to support custom HF repos such
    as BiomedCLIP.
    """
    resolved = _resolve_checkpoint_path(checkpoint)
    try:
        return AutoModel.from_pretrained(resolved, trust_remote_code=True)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load encoder from '{checkpoint}'. "
            "Make sure the checkpoint exists locally or on Hugging Face Hub."
        ) from exc


class _BasicImageProcessor:
    """
    Minimal processor used when a pretrained processor config is unavailable.

    It performs resize → tensor conversion → normalization so callers receive
    a dict with ``pixel_values`` similar to HF processors.
    """

    def __init__(self, size: int = 224):
        self.size = size
        # Standard ImageNet normalization used by most ViTs.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _prepare(self, image: Image.Image, size: Optional[Dict[str, int]]):
        width = size.get("width", self.size) if size else self.size
        height = size.get("height", self.size) if size else self.size
        resized = image.resize((width, height), Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __call__(
        self,
        images: Image.Image,
        return_tensors: str = "pt",
        size: Optional[Dict[str, int]] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(images, Image.Image):
            raise TypeError("Expected a PIL.Image input for the basic processor.")
        tensor = self._prepare(images, size)
        if return_tensors == "pt":
            return {"pixel_values": tensor.unsqueeze(0)}
        raise ValueError("Only return_tensors='pt' is supported in the fallback processor.")


def _try_loader(loader: Callable[..., Any], checkpoint: str):
    try:
        return loader(checkpoint, trust_remote_code=True)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=None)
def load_image_processor(checkpoint: str):
    """
    Load the processor responsible for turning PIL images into tensors.

    Falls back to a simple ImageNet-style processor when the checkpoint
    does not ship with its own processor configuration (common for custom
    timm checkpoints stored locally).
    """
    resolved = _resolve_checkpoint_path(checkpoint)

    for loader in (AutoImageProcessor, AutoProcessor, AutoFeatureExtractor):
        processor = _try_loader(loader.from_pretrained, resolved)
        if processor is not None:
            return processor

    # Fall back to a basic processor so downstream code can continue to run.
    return _BasicImageProcessor()
