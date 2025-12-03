#!/usr/bin/env python3

# Run Pre-defined Experiments - as a harness to run against the
# p4_end2end_explained_simple.py script

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# These are the Classifier / Pooling / LoRA combinations to use
CONFIGS: Dict[str, Dict] = {
    "linear_cls": {"classifier": "linear", "pooling": "cls", "use_lora": False},
    "linear_cls_lora": {"classifier": "linear", "pooling": "cls", "use_lora": True},
    "mlp_cls": {"classifier": "mlp", "pooling": "cls", "use_lora": False},
    "mlp_cls_lora": {"classifier": "mlp", "pooling": "cls", "use_lora": True},
    "rf_cls": {"classifier": "rf", "pooling": "cls", "use_lora": False},
}

# These are the list of experiments
EXPERIMENTS: List[Dict] = [
    {
        "id": "A",
        "description": "DINO-v2 baseline",
        "encoder": "facebook/dinov2-base",
        "configs": ["linear_cls", "mlp_cls", "rf_cls"],
        "poolings": ["cls", "mean"],
    },
    {
        "id": "B",
        "description": "Phikon baseline",
        "encoder": "owkin/phikon",
        "configs": ["linear_cls", "mlp_cls"],
        "poolings": ["cls", "mean"],
    },
    {
        "id": "C1",
        "description": "Dinov2 LoRA vs frozen",
        "encoder": "facebook/dinov2-base",
        "configs": ["linear_cls", "linear_cls_lora"],
        "poolings": ["cls"],
        "early_stop_patience": 2,
    },
    {
        "id": "C2",
        "description": "Phikon LoRA vs frozen",
        "encoder": "owkin/phikon",
        "configs": ["linear_cls", "linear_cls_lora"],
        "poolings": ["cls"],
        "early_stop_patience": 2,
    },
    {
        "id": "C3",
        "description": "UNI LoRA vs frozen",
        "encoder": "/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/UNI_model",
        "configs": ["linear_cls", "linear_cls_lora"],
        "poolings": ["cls"],
        "early_stop_patience": 3,
        "lr": 5e-5,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
    {
        "id": "C4",
        "description": "Virchow2 LoRA vs frozen",
        "encoder": "/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/Virchow2",
        "configs": ["linear_cls", "linear_cls_lora", "mlp_cls_lora"],
        "poolings": ["cls"],
        "early_stop_patience": 3,
        "lr": 5e-5,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
]


def run_case(
    fold: int,
    exp: Dict,
    config_name: str,
    pooling_override: str,
    args: argparse.Namespace,
    script_path: Path,
):
    cfg = CONFIGS[config_name]
    pooling = pooling_override or cfg["pooling"]
    classifier = cfg["classifier"]
    use_lora = cfg["use_lora"]

    lr = exp.get("lr", args.lr)
    lora_rank = exp.get("lora_rank", args.lora_rank)
    lora_alpha = exp.get("lora_alpha", args.lora_alpha)
    early_stop_patience = exp.get("early_stop_patience", args.early_stop_patience)

    output_dir = Path(args.output_dir) / f"{exp['id']}_{pooling}" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"fold{fold}_results.json"

# Build the command to call p4_end2end_explained_simple.py

    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir",
        f"data/fold{fold}",
        "--test-dir",
        "data/test",
        "--encoder",
        exp["encoder"],
        "--classifier",
        classifier,
        "--pooling",
        pooling,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(lr),
        "--early-stop-patience",
        str(early_stop_patience),
        "--early-stop-metric",
        args.early_stop_metric,
        "--lora-rank",
        str(lora_rank),
        "--lora-alpha",
        str(lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--rf-estimators",
        str(args.rf_estimators),
        "--output-json",
        str(output_json),
    ]

    if use_lora:
        cmd.append("--use-lora")
    if args.target_modules:
        cmd.extend(["--target-modules", *args.target_modules])

    print(f"\n=== Exp {exp['id']} ({exp['description']}), config {config_name}, pooling {pooling}, fold {fold} ===")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)
    return True


def main():

# There are some hyper-parameters which can be forced here
    parser = argparse.ArgumentParser(description="Direct runner for p4_end2end_explained_simple.py experiments.")
    parser.add_argument("--experiments", type=str, default="all", help="Comma-separated experiment IDs to run (e.g. A or all.")
    parser.add_argument("--folds", type=str, default="1,2,3,4", help="Comma-separated folds to run.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", choices=["accuracy", "f1_macro", "kappa_quadratic"], default="accuracy")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=None)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="foundation_results/p4_driver_runs")
    args = parser.parse_args()

    requested = {e.strip() for e in args.experiments.split(",")} if args.experiments.lower() != "all" else None
    folds = [int(f) for f in args.folds.split(",") if f]
    script_path = Path(__file__).resolve().parent / "p4_end2end_explained_simple.py"

    for exp in EXPERIMENTS:
        if requested and exp["id"] not in requested:
            continue
        for pooling in exp.get("poolings", ["cls"]):
            for config_name in exp["configs"]:
                if config_name not in CONFIGS:
                    print(f"Skipping unknown config {config_name}")
                    continue
                for fold in folds:
                    run_case(fold, exp, config_name, pooling, args, script_path)

if __name__ == "__main__":
    main()
