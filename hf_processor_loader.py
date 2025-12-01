# Utilities for loading Hugging Face image processors with graceful fallbacks.
#
# Studied code from:
#
# https://tia-toolbox.readthedocs.io/en/latest/_modules/tiatoolbox/models/architecture/vanilla.html?utm_source=chatgpt.com
# https://huggingface.co/docs/transformers/en/model_doc/timm_wrapper?utm_source=chatgpt.com

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch
from transformers import AutoImageProcessor, AutoProcessor, AutoModel
import timm
from timm.layers import GluMlp

def _append_error(errors: List[str], source: str, exc: Exception) -> None:
    errors.append(f"{source}: {exc}")

def load_image_processor(checkpoint: str) -> Any:
    """Return an image processor for ``checkpoint`` with AutoProcessor fallback."""
    errors: List[str] = []

    try:
        return AutoImageProcessor.from_pretrained(checkpoint)
    except Exception as exc:
        _append_error(errors, "AutoImageProcessor", exc)

    try:
        return AutoProcessor.from_pretrained(checkpoint)
    except Exception as exc:
        _append_error(errors, "AutoProcessor", exc)

    raise RuntimeError(
        f"Unable to load an image processor for '{checkpoint}'. "
        f"Tried: {' | '.join(errors)}"
    )

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

                    mlp_ratio = model_args.get("mlp_ratio", 4.0)

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
    kwargs.setdefault("trust_remote_code", True)
    return AutoModel.from_pretrained(checkpoint, **kwargs)
