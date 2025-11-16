"""
Utilities for loading Hugging Face image processors with graceful fallbacks.

Some checkpoints (e.g., BiomedCLIP) ship without ``preprocessor_config.json``,
which makes ``AutoImageProcessor`` fail. This helper tries the standard loader
first and then falls back to ``AutoProcessor`` or ``CLIPImageProcessor`` so
callers always get an object that can turn images into ``pixel_values``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from transformers import AutoImageProcessor, AutoProcessor, CLIPImageProcessor, AutoModel
except ImportError:  # pragma: no cover - handled at runtime on systems w/out transformers
    AutoImageProcessor = None  # type: ignore
    AutoProcessor = None  # type: ignore
    CLIPImageProcessor = None  # type: ignore
    AutoModel = None  # type: ignore

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None  # type: ignore

try:
    import timm
    from timm.layers import SwiGLU, GluMlp
except ImportError:  # pragma: no cover
    timm = None  # type: ignore
    SwiGLU = None  # type: ignore
    GluMlp = None  # type: ignore


def _append_error(errors: List[str], source: str, exc: Exception) -> None:
    errors.append(f"{source}: {exc}")


def load_image_processor(checkpoint: str) -> Any:
    """Return an image processor for ``checkpoint`` with CLIP-friendly fallbacks."""
    if AutoImageProcessor is None and AutoProcessor is None and CLIPImageProcessor is None:
        raise RuntimeError("transformers is required to load Hugging Face encoders.")

    errors: List[str] = []

    if AutoImageProcessor is not None:
        try:
            return AutoImageProcessor.from_pretrained(checkpoint)
        except Exception as exc:  # pragma: no cover - relies on external checkpoints
            _append_error(errors, "AutoImageProcessor", exc)

    if AutoProcessor is not None:
        try:
            return AutoProcessor.from_pretrained(checkpoint)
        except Exception as exc:  # pragma: no cover - relies on external checkpoints
            _append_error(errors, "AutoProcessor", exc)

    if CLIPImageProcessor is not None:
        try:
            return CLIPImageProcessor.from_pretrained(checkpoint)
        except Exception as exc:  # pragma: no cover - relies on external checkpoints
            _append_error(errors, "CLIPImageProcessor", exc)

    raise RuntimeError(
        f"Unable to load an image processor for '{checkpoint}'. "
        f"Tried: {' | '.join(errors) if errors else 'no loaders available.'}"
    )


CUSTOM_WEIGHT_FILES: Dict[str, str] = {
    # Microsoft BiomedCLIP repo exposes `open_clip_pytorch_model.bin` instead of pytorch_model.bin
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": "open_clip_pytorch_model.bin",
    "microsoft/BiomedCLIP-CLIP-ViT-B-16": "open_clip_pytorch_model.bin",
}

CUSTOM_CONFIG_FILES: Dict[str, str] = {
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": "open_clip_config.json",
    "microsoft/BiomedCLIP-CLIP-ViT-B-16": "open_clip_config.json",
}


def load_auto_model(checkpoint: str, **kwargs: Any):
    """Load Hugging Face models with remote-code support enabled by default, or timm models for local paths."""
    # Check if this is a local path with timm-style config
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            import json
            with open(config_file) as f:
                config = json.load(f)
            # Check if this is a timm model (has architecture field or reg_tokens in model_args)
            if "architecture" in config and timm is not None:
                if "model_args" in config and "reg_tokens" in config["model_args"]:
                    # Load with timm for custom architectures like Virchow2
                    print(f"Loading timm model from {checkpoint}")
                    model_name = config["architecture"]
                    model_args = config.get("model_args", {})
                    weights_file = checkpoint_path / "pytorch_model.bin"

                    # Create model with timm - check if SwiGLU is needed based on MLP ratio
                    mlp_ratio = model_args.get("mlp_ratio", 4.0)
                    # SwiGLU uses mlp_layer that gates, so effective ratio is higher
                    # For Virchow2: 6832 / 1280 = 5.3375, and fc2 takes 3416 (half of 6832)
                    # This indicates SwiGLU with a "gated" mlp
                    model_kwargs = {
                        "pretrained": False,
                        "num_classes": model_args.get("num_classes", 0),
                        "img_size": model_args.get("img_size", 224),
                        "init_values": model_args.get("init_values", 1e-5),
                        "reg_tokens": model_args.get("reg_tokens", 0),
                        "mlp_ratio": mlp_ratio,
                        "global_pool": model_args.get("global_pool", ""),
                        "dynamic_img_size": model_args.get("dynamic_img_size", True),
                        "act_layer": "silu",  # SwiGLU uses SiLU activation
                    }

                    # Add GluMlp for models with mlp_ratio > 5 (indicates gated MLP)
                    if mlp_ratio > 5.0 and GluMlp is not None:
                        model_kwargs["mlp_layer"] = GluMlp

                    model = timm.create_model(model_name, **model_kwargs)

                    # Load weights
                    if weights_file.exists():
                        state_dict = torch.load(weights_file, map_location="cpu")
                        model.load_state_dict(state_dict, strict=False)

                    return model

    # Fall back to transformers AutoModel
    if AutoModel is None:
        raise RuntimeError("transformers is required to load Hugging Face encoders.")

    kwargs.setdefault("trust_remote_code", True)
    try:
        return AutoModel.from_pretrained(checkpoint, **kwargs)
    except EnvironmentError as exc:
        custom_file = CUSTOM_WEIGHT_FILES.get(checkpoint)
        if not custom_file:
            raise
        if hf_hub_download is None:
            raise
        resolved_path = hf_hub_download(checkpoint, custom_file)
        snapshot_dir = Path(resolved_path).parent
        target_file = snapshot_dir / "pytorch_model.bin"
        if not target_file.exists():
            shutil.copy(resolved_path, target_file)
        custom_cfg = CUSTOM_CONFIG_FILES.get(checkpoint)
        if custom_cfg:
            cfg_path = hf_hub_download(checkpoint, custom_cfg)
            target_cfg = snapshot_dir / "config.json"
            if not target_cfg.exists():
                shutil.copy(cfg_path, target_cfg)
        return AutoModel.from_pretrained(str(snapshot_dir), **kwargs)
