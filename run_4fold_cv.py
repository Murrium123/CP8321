#!/usr/bin/env python3
"""
Run 4-fold cross-validation experiments with foundation models.

This script runs p4_end2end_explained.py on all 4 validation folds and
aggregates the results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def run_fold(fold_num, encoder, classifier, use_lora, epochs, batch_size, lr, pooling, output_dir):
    """Run training on a single fold."""
    print(f"\n{'='*80}")
    print(f"RUNNING FOLD {fold_num}")
    print(f"{'='*80}\n")

    # Data directory for this fold
    data_dir = f"data/fold{fold_num}"

    # Output file for this fold
    output_file = output_dir / f"fold{fold_num}_results.json"

    # Build command
    cmd = [
        "python", "p4_end2end_explained.py",
        "--data-dir", data_dir,
        "--test-dir", "data/test",  # Shared test set for all folds
        "--encoder", encoder,
        "--classifier", classifier,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--pooling", pooling,
        "--output-json", str(output_file),
    ]

    if use_lora:
        cmd.append("--use-lora")

    print(f"Command: {' '.join(cmd)}\n")

    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Fold {fold_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Fold {fold_num} failed with error code {e.returncode}")
        return False


def aggregate_results(output_dir, num_folds=4):
    """Aggregate results from all folds."""
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS FROM ALL FOLDS")
    print(f"{'='*80}\n")

    all_results = []
    metrics_to_aggregate = [
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "kappa_quadratic", "auc_weighted"
    ]

    # Load results from each fold
    for fold_num in range(1, num_folds + 1):
        result_file = output_dir / f"fold{fold_num}_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
                print(f"✓ Loaded results from fold {fold_num}")
        else:
            print(f"⚠️  Results not found for fold {fold_num}")

    if not all_results:
        print("❌ No results to aggregate!")
        return

    # Calculate mean and std for each metric
    print(f"\n{'='*80}")
    print("AGGREGATED 4-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}\n")

    aggregated = {}
    for metric in metrics_to_aggregate:
        values = [r["test_metrics"][metric] for r in all_results if metric in r["test_metrics"]]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregated[metric] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "values": values
            }
            print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")

    # Calculate confusion matrix average
    print(f"\nAverage Confusion Matrix:")
    conf_matrices = [r["test_metrics"]["confusion_matrix"] for r in all_results]
    avg_conf = np.mean(conf_matrices, axis=0)
    for row in avg_conf:
        print(f"  {[f'{int(x):4d}' for x in row]}")

    # Save aggregated results
    summary_file = output_dir / "4fold_summary.json"
    summary = {
        "encoder": all_results[0]["encoder"],
        "classifier": all_results[0]["classifier"],
        "use_lora": all_results[0]["use_lora"],
        "num_folds": len(all_results),
        "aggregated_metrics": aggregated,
        "individual_folds": all_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Aggregated results saved to {summary_file}")
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Run 4-fold cross-validation with foundation models."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="facebook/dinov2-base",
        help="Foundation model checkpoint to use"
    )
    parser.add_argument(
        "--classifier",
        choices=["linear", "mlp", "rf", "svm"],
        default="linear",
        help="Classifier type"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="foundation_results/4fold_cv",
        help="Output directory for results"
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of folds to run (e.g., '1,2' or '1,2,3,4')"
    )

    args = parser.parse_args()

    # Parse which folds to run
    folds_to_run = [int(f) for f in args.folds.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("4-FOLD CROSS-VALIDATION EXPERIMENT")
    print("="*80)
    print(f"Encoder: {args.encoder}")
    print(f"Classifier: {args.classifier}")
    print(f"LoRA: {args.use_lora}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Pooling: {args.pooling}")
    print(f"Folds to run: {folds_to_run}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Run each fold
    successful_folds = 0
    for fold_num in folds_to_run:
        success = run_fold(
            fold_num=fold_num,
            encoder=args.encoder,
            classifier=args.classifier,
            use_lora=args.use_lora,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            pooling=args.pooling,
            output_dir=output_dir
        )
        if success:
            successful_folds += 1

    # Aggregate results
    if successful_folds > 0:
        aggregate_results(output_dir, num_folds=len(folds_to_run))
        print(f"\n✅ Successfully completed {successful_folds}/{len(folds_to_run)} folds")
    else:
        print("\n❌ No folds completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
