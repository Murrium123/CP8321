#!/usr/bin/env python3

# Code to run models on the SICAPv2 datsets. This is a Python file with a
# number of options so that the same Python file can be used to drive all
# experiments.
#
# References - Studied code from
#
# https://github.com/jusiro/mil_histology for the basic structure of how to
# use PyTorch with this data
# https://medium.com/@lmpo/pytorch-the-backbone-of-modern-deep-learning-3a7b50cb5ba9
# for PyTorch and building the Backbone structure
# https://www.doptsw.com/posts/post_2024-11-06_0bed19 for how to use LoRA with
# PyTorch

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

from hf_processor_loader import load_auto_model, load_image_processor
from peft import LoraConfig, get_peft_model

CLASSES = ["NC", "G3", "G4", "G5"]  # Gleason classes
IMAGE_SIZE = 224  # Common input size for ViT
DEFAULT_BATCH = 16
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Simple logging messages for debugging
def log(msg: str) -> None:
    """Utility print for consistent status updates."""
    print(f"[p4_end2end_explained_simple] {msg}")

# Load files from the split files
# It scans a directory for image files and returns records of pairs of
# image path and each class label (CLASSES has the labels)
def load_split(split_dir: Path) -> List[tuple[Path, int]]:
    records: List[tuple[Path, int]] = []
    for idx, name in enumerate(CLASSES):
        class_dir = split_dir / name
        if not class_dir.exists():
            continue
        for ext in ("*.jpg",):  # SICAPv2 patches are stored as .jpg
            for path in class_dir.glob(ext):
                records.append((path, idx))
    if not records:
        raise FileNotFoundError(f"No images found in {split_dir} !!")
    return records

# PyTorch Dataset for loading and preprocessing histopathology image patches.
# Each sample is loaded from disk, converted to RGB, and transformed using the
# foundation model's image processor. Returns preprocessed pixel tensors and
# their corresponding integer class labels.
#
# Inputs:
#   - samples: List of (image_path, label_index) tuples from load_split
#   - processor: Hugging Face image processor for the chosen encoder
class PatchDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[Path, int]], processor):
        self.samples = list(samples)
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        )

        return inputs["pixel_values"].squeeze(0), label

# This is the Backbone which is the pretrained feature extractor that turns
# images into embeddings vectors. This is the code of the model pipeline.
class Backbone:

# Responsibilities:
# - Selects device automatically to allow optimizing on
# - Loads pretrained weights via load_auto_model
# - Optionally injects LoRA adapters.
# - Implements CLS/mean pooling

    def __init__(
        self,
        checkpoint: str,
        use_lora: bool,
        target_modules: Sequence[str],
        pooling: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        # Just to ensure that if we have a GPU we can use it - Macbooks have a
        # GPU while many Intel PCs do not. MPS is Metal Performance Shaders
        # which is Macbook Apple Silicon GPUs. CUDA is for NVIDIA GPUs.
        # The performance difference has been about 30-40X faster. A run
        # that takes 10 minutes on a GPU can take 6 hours on a CPU-only PC.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.pooling = pooling

        # Auto load the trained model. This is from the hf_processor_loader.py
        # helper script to load the HuggingFace model
        self.model = load_auto_model(checkpoint).to(self.device)

        # Check if this is a timm model (doesn't have .config attribute)
        # There are 2 of these - UNI and Virchow2 and these are stored locally
        self.is_timm_model = not hasattr(self.model, "config")

        # Set whether LoRA will be used. LoRA is a form of fine-tuning
        self.use_lora = bool(use_lora and get_peft_model is not None and LoraConfig is not None)

        if self.use_lora:
            # LoRA inserts small trainable matrices into attention/linear layers.
            # Only these adapters get updated; the original backbone weights stay frozen.
            # First call resolve_target_modules to get the layers to target
            resolved_targets = self._resolve_target_modules(checkpoint, target_modules)

            # Create the configuration of the LoRA adapters
            cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=resolved_targets,
            )

            log("Initializing LoRA adapters on encoder.")

            # Wraps the model with the LoRA adapters - trainable matrices
            self.model = get_peft_model(self.model, cfg)

        if not self.use_lora:
            # Freeze a module
            for p in self.model.parameters():
                p.requires_grad = False

        # Determine hidden size - determine the dimensions of the feature
        # vector to be output
        if self.is_timm_model:
            # For timm models, use num_features or embed_dim
            self.hidden_size = getattr(self.model, "num_features", getattr(self.model, "embed_dim", None))
        else:
            # For Hugging Face models
            self.hidden_size = getattr(
                self.model.config, "hidden_size", getattr(self.model.config, "embed_dim", getattr(self.model.config, "num_features", None))
            )
            if self.hidden_size is None and hasattr(self.model, "num_features"):
                self.hidden_size = self.model.num_features

    # Implement the forward pass with CLS or mean pooling
    # CLS pooling: Uses the first token ([:, 0, :]) which acts as a learned
    #              global summary of the entire image, trained during pre-training
    # Mean pooling: Averages all patch tokens to create a summary representation
    #              that weights all spatial regions equally
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.is_timm_model:
            hidden = self.model.forward_features(pixel_values)
            if self.pooling == "mean":
                return hidden.mean(dim=1)
            return hidden[:, 0, :]
        else:
            # HuggingFace transformers models return a dict-like object;
            # we grab the last hidden states (sequence of patch tokens).
            outputs = self.model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state
            if self.pooling == "mean":
                return hidden.mean(dim=1)
            return hidden[:, 0, :]

# LoRA needs us to select the layers between which the LoRA matrices will be trained
# These differ depending on the model we are using.
    @staticmethod
    def _resolve_target_modules(checkpoint: str, target_modules: Sequence[str]) -> List[str]:
        if target_modules and target_modules != ["auto"]:
            return list(target_modules)
        ckpt = checkpoint.lower()
        if "uni" in ckpt or "virchow" in ckpt or ckpt.endswith("model") or "timm" in ckpt or checkpoint.startswith("/"):
            log("Auto-selecting LoRA targets for timm/UNI/Virchow2 backbone: ['qkv', 'proj']")
            return ["qkv", "proj"]
        log("Auto-selecting LoRA targets for transformer backbone: ['query', 'key', 'value']")
        return ["query", "key", "value"]

# This builds the classifier head we will use. The two models we are using are Linear
# with one hidden layer and MLP - a simple 2 hidden layer neural network. We use ReLU and
# we have a dropout rate of 0.2. This obviously could be tuned and we could introduce more
# complex networks here.
#
# For the MLP, we halve the dimensions (hidden_dim // 2) to create a bottleneck that
# forces the network to learn compressed representations. This is a common pattern in
# classification heads. This dimensionality could be tuned (e.g., // 4, // 8, or
# multiple layers with gradual reduction) depending on complexity needs and overfitting.
def build_head(hidden_dim: int, num_classes: int, kind: str) -> torch.nn.Module:
    if kind == "mlp":
        # Simple two-layer MLP head (common PyTorch Sequential pattern)
        return torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, num_classes),
        )
    return torch.nn.Linear(hidden_dim, num_classes)

# This is a standard PyTorch training loop with gradient descent
# This trains one epoch of the training set.
def train_epoch(backbone: Backbone, head: torch.nn.Module, loader: DataLoader, optimizer, device: torch.device, autocast_device: str):
    head.train()

    # If LoRA is being used the weights in the LoRA matrices will need to be
    # trained else we are just evaluating the frozen matrix
    if backbone.use_lora:
        backbone.model.train()
    else:
        backbone.model.eval()
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()
    use_autocast = torch.cuda.is_available()

    for pixels, labels in loader:
        optimizer.zero_grad()
        pixels = pixels.to(device)
        labels = labels.to(device)
        if use_autocast:
            # Autocast does mixed precision. It lets PyTorch optimize the
            # precision of floats so that it runs better on GPUs
            with torch.autocast(device_type=autocast_device, dtype=torch.float16):
                embeddings = backbone.forward(pixels)
                logits = head(embeddings if backbone.use_lora else embeddings.detach())
                loss = criterion(logits, labels)
        else:
            embeddings = backbone.forward(pixels)
            logits = head(embeddings if backbone.use_lora else embeddings.detach())
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

# Simple function to run inference
@torch.no_grad()
def run_inference(backbone: Backbone, head: torch.nn.Module, loader: DataLoader, device: torch.device):
    backbone.model.eval()
    head.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    for pixels, labels in loader:
        pixels = pixels.to(device)
        logits = head(backbone.forward(pixels))
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(pred.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Collect the embeddings after running a forward pass with the model
# Embeddings are dense vector representations (e.g., 768-dimensional for dinov2-base)
# that capture the semantic features of each image patch. These come from the frozen
# backbone and serve as feature vectors for sklearn classifiers like Random Forest.
@torch.no_grad()
def collect_embeddings(backbone: Backbone, loader: DataLoader, device: torch.device):
    backbone.model.eval()
    emb_list = []
    label_list = []
    for pixels, labels in loader:
        pixels = pixels.to(device)
        embeddings = backbone.forward(pixels)
        emb_list.append(embeddings.cpu().numpy())
        label_list.extend(labels.numpy().tolist())
    return np.concatenate(emb_list, axis=0), np.array(label_list)

# Compute comprehensive classification metrics for Gleason grading.
# Inputs:
#   - y_true: Ground truth labels (integers 0-3 for NC, G3, G4, G5)
#   - y_pred: Predicted labels (integers 0-3)
#   - probs: Prediction probabilities for each class (N x 4 array)
# Returns:
#   Dictionary containing accuracy, precision, recall, F1 (macro averaged),
#   quadratic Cohen's kappa, weighted AUC, and confusion matrix.
# Metrics come from scikit-learn; macro averages treat all classes equally.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa_quadratic": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metrics["auc_weighted"] = roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")

    return metrics

# Entry-point for the script

def main():

    # Argument parsing

    parser = argparse.ArgumentParser(description="SICAPv2 classification with foundation models.")
    parser.add_argument("--data-dir", type=str, default="data", help="Folder containing data")
    parser.add_argument("--test-dir", type=str, default=None, help="Folder containing test")

    # Note that DINO-v2 and Phikon can be run from any machine without
    # downloading models. UNI and Virchow2 require the models to have been
    # downloaded
    parser.add_argument(
        "--encoder",
        type=str,
        default="facebook/dinov2-base",
        choices=[
            "facebook/dinov2-base",
            "owkin/phikon",
            "/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/UNI_model", # Stored locally only
            "/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/Virchow2",  # Stored locally only
        ],
        help="Choose which foundation model checkpoint to use",
    )

    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls", help="Whether to use CLS token or mean pooling")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["auto"],
        help="LoRA target modules",
    )
    # LoRA hyperparameters
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")

    # Classifier choice
    parser.add_argument("--classifier", choices=["linear", "mlp", "rf"], default="linear", help="Downstream classifier type.")

    # Classical model hyperparameters.
    parser.add_argument("--rf-estimators", type=int, default=200, help="Number of trees for RandomForest (when classifier=rf).")

    # Training loop knobs: epochs and patience control how long to train.
    parser.add_argument("--epochs", type=int, default=6, help="Max epochs for neural heads (linear/mlp).")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training based on validation metric",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["accuracy", "f1_macro", "kappa_quadratic"],
        default="accuracy",
        help="Metric used to decide early stopping",
    )

    # Batch size and learning rate
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("--output-json", type=str, default="foundation_results/p4_end2end_metrics.json", help="Where to save metrics JSON.")

    args = parser.parse_args()

    # Make sure that LoRA is only used with neural classifiers
    effective_use_lora = args.use_lora and args.classifier in {"linear", "mlp"}
    if args.use_lora and not effective_use_lora:
        log(" Error: LoRA requires a neural classifier head")

    # Set the data directories
    data_root = Path(args.data_dir)
    train_items = load_split(data_root / "train")
    valid_items = load_split(data_root / "valid")

    # Set the test directory
    test_root = Path(args.test_dir) if args.test_dir else data_root / "test"
    test_items = load_split(test_root)
    log(f"Loaded SICAPv2 splits: {len(train_items)} train / {len(valid_items)} valid / {len(test_items)} test patches.")

    # Load the image processor to be used
    log("Convert whole-slide patches into tensor batches.")
    processor = load_image_processor(args.encoder)

    # Creates the Tensors to be used by the model
    train_ds = PatchDataset(train_items, processor)
    valid_ds = PatchDataset(valid_items, processor)
    test_ds = PatchDataset(test_items, processor)

    # PyTorch DataLoader for each of train, validation and test datasets
    # Note that we shuffle the training samples so that the models sees the
    # sample in different order each epoch
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Define the Backbone - the network architecture
    backbone = Backbone(
        checkpoint=args.encoder,
        use_lora=effective_use_lora,
        target_modules=args.target_modules,
        pooling=args.pooling,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Classifier to train / use
    classifier_type = args.classifier

    if classifier_type in {"linear", "mlp"}:

        # Neural network head - these will be trained
        head = build_head(backbone.hidden_size, len(CLASSES), classifier_type).to(backbone.device)
        params = list(head.parameters()) + (list(backbone.model.parameters()) if backbone.use_lora else [])
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        # autocast is the option to use mixed precision training which is
        # much faster - use 16 bit floats within PyTorch where it can instead
        # of 32 bit floats and this is much faster with the GPU as well
        # Set autocast device type to match the actual device being used
        autocast_device = "cuda" if torch.cuda.is_available() else "mps"

        # Print - for information - whether LoRA is enabled or not
        if backbone.use_lora:
            phase_msg = "LoRA enabled"
        else:
            phase_msg = "No LoRA used"
        log(phase_msg)

        # Initailize the training parameters
        best_metric = -float("inf")
        best_head_state = copy.deepcopy(head.state_dict())
        # Training will happen only with LoRA else the frozen foundation
        # model will be used
        if backbone.use_lora:
            best_backbone_state = copy.deepcopy(backbone.model.state_dict())
        else:
            best_backbone_state = None

        epochs_without_improve = 0
        patience = args.early_stop_patience

        # Run the number of epoch specified
        for epoch in range(args.epochs):
            loss = train_epoch(backbone, head, train_loader, optimizer, backbone.device, autocast_device)
            y_val, p_val, prob_val = run_inference(backbone, head, valid_loader, backbone.device)
            val_metrics = compute_metrics(y_val, p_val, prob_val)

            # Early stopping compares a chosen metric across epochs to see
            # if we are improving or we should stop.
            metric_value = val_metrics.get(args.early_stop_metric, val_metrics.get("accuracy", 0.0))
            # Use small epsilon (1e-6) to avoid floating point comparison issues
            # Only consider it an improvement if the metric increased by at least this amount
            improved = metric_value > best_metric + 1e-6
            if improved:
                best_metric = metric_value
                epochs_without_improve = 0
                best_head_state = copy.deepcopy(head.state_dict())
                if backbone.use_lora:
                    best_backbone_state = copy.deepcopy(backbone.model.state_dict())
                log(
                    f"Epoch {epoch+1}/{args.epochs} - loss {loss:.4f} - val {args.early_stop_metric} improved to {metric_value:.4f}"
                )
            else:
                epochs_without_improve += 1
                log(
                    f"Epoch {epoch+1}/{args.epochs} - loss {loss:.4f} - val {args.early_stop_metric} {metric_value:.4f} (no improvement)"
                )
                if patience and epochs_without_improve >= patience:
                    log(f"Early stopping triggered (no {args.early_stop_metric} improvement for {patience} epoch(s)).")
                    break

        # Set the weights back to the best metric state
        head.load_state_dict(best_head_state)
        if backbone.use_lora and best_backbone_state is not None:
            backbone.model.load_state_dict(best_backbone_state)
        log(f"Best validation {args.early_stop_metric}: {best_metric:.4f}")

        # Run inference based on the best model parameters
        y_test, p_test, prob_test = run_inference(backbone, head, test_loader, backbone.device)
    else:
        # For Random Forest - and could add other classifiers
        log("Extracting foundation-model embeddings for classifier.")
        train_X, train_y = collect_embeddings(backbone, train_loader, backbone.device)
        valid_X, valid_y = collect_embeddings(backbone, valid_loader, backbone.device)
        test_X, test_y = collect_embeddings(backbone, test_loader, backbone.device)

        # RF is the only option but could add others here
        if classifier_type == "rf":
            clf = RandomForestClassifier(
                n_estimators=args.rf_estimators,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown sklearn classifier type: {classifier_type}")

        log(f"[sklearn] Training {classifier_type.upper()} classifier on extracted features.")
        clf.fit(train_X, train_y)
        val_probs = clf.predict_proba(valid_X)
        val_preds = val_probs.argmax(axis=1)
        val_metrics = compute_metrics(valid_y, val_preds, val_probs)
        log(f"[sklearn] Val acc {val_metrics['accuracy']:.4f}")

        test_probs = clf.predict_proba(test_X)
        p_test = test_probs.argmax(axis=1)
        prob_test = test_probs
        y_test = test_y

    # Compute the metrics to be logged
    test_metrics = compute_metrics(y_test, p_test, prob_test)
    log("Evaluating Gleason grading performance.")

    # Print out the metrcs
    log("Test metrics:")
    for key, value in test_metrics.items():
        if key == "confusion_matrix":
            log(f"  {key}: {value}")
        else:
            log(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Write out the JSON file for the run - this has been handy to be able
    # to keep runs and analyze later
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "encoder": args.encoder,
                "use_lora": effective_use_lora,
                "classifier": args.classifier,
                "pooling": args.pooling,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    log(f"Saved metrics to {output_path}")

if __name__ == "__main__":
    # This was added because we saw a warning in training so we turned
    # this parallelism off by default
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
