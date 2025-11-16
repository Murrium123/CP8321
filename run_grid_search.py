#!/usr/bin/env python3
"""
Grid search script for Dinov2-base with 4-fold cross-validation.

Tests all combinations of:
- Classifiers: linear, mlp, rf, svm
- Pooling: cls only (mean pooling skipped)
- LoRA: enabled/disabled (for neural classifiers only)

Total: 6 configurations × 4 folds = 24 runs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Grid search configurations
CONFIGURATIONS = [
    # Linear classifier
    {"classifier": "linear", "pooling": "cls", "use_lora": False, "name": "linear_cls"},
    {"classifier": "linear", "pooling": "cls", "use_lora": True, "name": "linear_cls_lora"},

    # MLP classifier
    {"classifier": "mlp", "pooling": "cls", "use_lora": False, "name": "mlp_cls"},
    {"classifier": "mlp", "pooling": "cls", "use_lora": True, "name": "mlp_cls_lora"},

    # Random Forest (no LoRA)
    {"classifier": "rf", "pooling": "cls", "use_lora": False, "name": "rf_cls"},

    # SVM (no LoRA)
    {"classifier": "svm", "pooling": "cls", "use_lora": False, "name": "svm_cls"},
]


def run_configuration(config, encoder, epochs, batch_size, lr, output_dir, folds_to_run):
    """Run 4-fold CV for a single configuration."""
    config_name = config["name"]
    config_output_dir = output_dir / config_name

    print(f"\n{'='*80}")
    print(f"RUNNING CONFIGURATION: {config_name}")
    print(f"{'='*80}")
    print(f"  Classifier: {config['classifier']}")
    print(f"  Pooling: {config['pooling']}")
    print(f"  LoRA: {config['use_lora']}")
    print(f"  Folds: {folds_to_run}")
    print(f"  Output: {config_output_dir}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        "python", "run_4fold_cv.py",
        "--encoder", encoder,
        "--classifier", config["classifier"],
        "--pooling", config["pooling"],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--output-dir", str(config_output_dir),
        "--folds", ",".join(map(str, folds_to_run)),
    ]

    if config["use_lora"]:
        cmd.append("--use-lora")

    print(f"Command: {' '.join(cmd)}\n")

    # Run the experiment
    try:
        start_time = datetime.now()
        result = subprocess.run(cmd, check=True)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print(f"\n✓ Configuration '{config_name}' completed successfully in {duration:.1f} minutes")
        return {
            "config": config_name,
            "success": True,
            "duration_minutes": duration,
            "error": None
        }
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Configuration '{config_name}' failed with error code {e.returncode}")
        return {
            "config": config_name,
            "success": False,
            "duration_minutes": 0,
            "error": str(e)
        }


def plot_confusion_matrices(output_dir, configs_run):
    """Create confusion matrix plots for each configuration."""
    print(f"\n{'='*80}")
    print("CREATING CONFUSION MATRIX PLOTS")
    print(f"{'='*80}\n")

    class_names = ['NC', 'G3', 'G4', 'G5']

    # Collect data only for configurations that were actually run
    cm_data = []
    for config in configs_run:
        config_name = config["name"]
        conf_matrices = []

        for fold in range(1, 5):
            fold_file = output_dir / config_name / f"fold{fold}_results.json"
            if fold_file.exists():
                with open(fold_file, 'r') as f:
                    data = json.load(f)
                    cm = data.get("test_metrics", {}).get("confusion_matrix", None)
                    if cm:
                        conf_matrices.append(np.array(cm))

        if conf_matrices:
            avg_cm = np.mean(conf_matrices, axis=0)
            avg_cm_pct = avg_cm / avg_cm.sum(axis=1, keepdims=True) * 100
            num_folds = len(conf_matrices)
            cm_data.append((config_name, avg_cm_pct, num_folds))
            print(f"✓ Loaded confusion matrix for {config_name} ({num_folds} folds)")

    if not cm_data:
        print("⚠️  No confusion matrix data available")
        return

    # Determine grid size based on number of configurations
    n = len(cm_data)
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    elif n == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()
    elif n <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

    # Plot each confusion matrix
    for idx, (config_name, avg_cm_pct, num_folds) in enumerate(cm_data):
        ax = axes[idx]
        sns.heatmap(avg_cm_pct, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Percentage (%)'})
        ax.set_title(f'{config_name}\n(averaged over {num_folds} fold(s))',
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Hide unused subplots
    for idx in range(len(cm_data), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Confusion matrices saved to {cm_path}")


def plot_metric_comparisons(output_dir, configs_run):
    """Create bar charts comparing all metrics across configurations."""
    print(f"\n{'='*80}")
    print("CREATING METRIC COMPARISON CHARTS")
    print(f"{'='*80}\n")

    # Collect data only for configurations that were run
    data = []
    for config in configs_run:
        config_name = config["name"]
        summary_file = output_dir / config_name / "4fold_summary.json"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                result = json.load(f)

            metrics = result.get("aggregated_metrics", {})
            data.append({
                "Configuration": config_name,
                "Accuracy": metrics.get('accuracy', {}).get('mean', 0) * 100,
                "Accuracy_std": metrics.get('accuracy', {}).get('std', 0) * 100,
                "F1-Score": metrics.get('f1_macro', {}).get('mean', 0) * 100,
                "F1_std": metrics.get('f1_macro', {}).get('std', 0) * 100,
                "Kappa": metrics.get('kappa_quadratic', {}).get('mean', 0) * 100,
                "Kappa_std": metrics.get('kappa_quadratic', {}).get('std', 0) * 100,
                "AUC": metrics.get('auc_weighted', {}).get('mean', 0) * 100,
                "AUC_std": metrics.get('auc_weighted', {}).get('std', 0) * 100,
            })
            print(f"✓ Loaded metrics for {config_name}")

    if not data:
        print("⚠️  No data available for plotting")
        return

    df = pd.DataFrame(data)

    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)

    # Skip plotting if only one configuration (summary table is sufficient)
    if len(df) == 1:
        print(f"⚠️  Only 1 configuration - skipping comparison charts (use summary table instead)")
        return

    # Create bar charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = [
        ('Accuracy', 'Accuracy_std', 'Accuracy (%)', axes[0, 0]),
        ('F1-Score', 'F1_std', 'F1-Score (%)', axes[0, 1]),
        ('Kappa', 'Kappa_std', 'Kappa (× 100)', axes[1, 0]),
        ('AUC', 'AUC_std', 'AUC (× 100)', axes[1, 1]),
    ]

    colors = ['#2ecc71' if 'lora' in name else '#3498db' for name in df['Configuration']]

    for metric, std, ylabel, ax in metrics_to_plot:
        x = np.arange(len(df))
        bars = ax.bar(x, df[metric], yerr=df[std], capsize=5, color=colors,
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{ylabel} Comparison (4-fold CV)', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, val, err) in enumerate(zip(bars, df[metric], df[std])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err,
                   f'{val:.1f}±{err:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='With LoRA'),
        Patch(facecolor='#3498db', edgecolor='black', label='Without LoRA')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 0.98), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart_path = output_dir / "metric_comparisons.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metric comparison charts saved to {chart_path}")


def create_latex_table(output_dir, configs_run):
    """Create a LaTeX table ready for paper."""
    print(f"\n{'='*80}")
    print("CREATING LaTeX TABLE")
    print(f"{'='*80}\n")

    # Collect data only for configurations that were run
    data = []
    for config in configs_run:
        config_name = config["name"]
        summary_file = output_dir / config_name / "4fold_summary.json"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                result = json.load(f)

            metrics = result.get("aggregated_metrics", {})
            data.append({
                "Configuration": config_name.replace('_', ' ').title(),
                "Classifier": config["classifier"].upper(),
                "LoRA": "\\checkmark" if config["use_lora"] else "---",
                "Accuracy": f"{metrics.get('accuracy', {}).get('mean', 0)*100:.2f} $\\pm$ {metrics.get('accuracy', {}).get('std', 0)*100:.2f}",
                "F1": f"{metrics.get('f1_macro', {}).get('mean', 0)*100:.2f} $\\pm$ {metrics.get('f1_macro', {}).get('std', 0)*100:.2f}",
                "Kappa": f"{metrics.get('kappa_quadratic', {}).get('mean', 0):.3f} $\\pm$ {metrics.get('kappa_quadratic', {}).get('std', 0):.3f}",
                "AUC": f"{metrics.get('auc_weighted', {}).get('mean', 0):.3f} $\\pm$ {metrics.get('auc_weighted', {}).get('std', 0):.3f}",
                "Acc_Mean": metrics.get('accuracy', {}).get('mean', 0),
            })
            print(f"✓ Added {config_name} to LaTeX table")

    if not data:
        print("⚠️  No data available for LaTeX table")
        return

    df = pd.DataFrame(data)
    df = df.sort_values('Acc_Mean', ascending=False)
    df = df.drop('Acc_Mean', axis=1)

    # Create LaTeX table
    latex_table = []
    latex_table.append("\\begin{table}[htbp]")
    latex_table.append("\\centering")
    latex_table.append("\\caption{Foundation Model Grid Search Results on SICAPv2 (Cross-Validation)}")
    latex_table.append("\\label{tab:grid_search_results}")
    latex_table.append("\\begin{tabular}{llcccc}")
    latex_table.append("\\toprule")
    latex_table.append("\\textbf{Configuration} & \\textbf{Classifier} & \\textbf{LoRA} & \\textbf{Accuracy (\\%)} & \\textbf{F1-Score (\\%)} & \\textbf{Kappa} & \\textbf{AUC} \\\\")
    latex_table.append("\\midrule")

    for _, row in df.iterrows():
        line = f"{row['Configuration']} & {row['Classifier']} & {row['LoRA']} & {row['Accuracy']} & {row['F1']} & {row['Kappa']} & {row['AUC']} \\\\"
        latex_table.append(line)

    latex_table.append("\\bottomrule")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\end{table}")

    # Save to file
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_table))

    print(f"✓ LaTeX table saved to {latex_path}")

    # Also print to console
    print(f"\n{'-'*80}")
    print("LaTeX Table (copy-paste into your paper):")
    print(f"{'-'*80}\n")
    print('\n'.join(latex_table))
    print(f"\n{'-'*80}\n")


def create_summary_table(output_dir, configs_run):
    """Create a summary table comparing all configurations."""
    print(f"\n{'='*80}")
    print("CREATING SUMMARY TABLE")
    print(f"{'='*80}\n")

    results = []

    for config in configs_run:
        config_name = config["name"]
        summary_file = output_dir / config_name / "4fold_summary.json"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)

            metrics = data.get("aggregated_metrics", {})

            result = {
                "Configuration": config_name,
                "Classifier": config["classifier"],
                "LoRA": "Yes" if config["use_lora"] else "No",
                "Accuracy": f"{metrics.get('accuracy', {}).get('mean', 0)*100:.2f} ± {metrics.get('accuracy', {}).get('std', 0)*100:.2f}",
                "F1-Score": f"{metrics.get('f1_macro', {}).get('mean', 0)*100:.2f} ± {metrics.get('f1_macro', {}).get('std', 0)*100:.2f}",
                "Kappa": f"{metrics.get('kappa_quadratic', {}).get('mean', 0):.3f} ± {metrics.get('kappa_quadratic', {}).get('std', 0):.3f}",
                "AUC": f"{metrics.get('auc_weighted', {}).get('mean', 0):.3f} ± {metrics.get('auc_weighted', {}).get('std', 0):.3f}",
                "Acc_Mean": metrics.get('accuracy', {}).get('mean', 0),
            }
            results.append(result)
            print(f"✓ Loaded summary for {config_name}")
        else:
            print(f"⚠️  Summary not found for {config_name}")

    if results:
        # Create DataFrame and sort by accuracy
        df = pd.DataFrame(results)
        df = df.sort_values('Acc_Mean', ascending=False)
        df = df.drop('Acc_Mean', axis=1)  # Remove the sorting column

        # Save to CSV
        csv_path = output_dir / "grid_search_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Summary saved to {csv_path}")

        # Print table
        print(f"\n{'='*80}")
        if len(results) > 1:
            print("GRID SEARCH RESULTS (sorted by accuracy)")
        else:
            print("EXPERIMENT RESULTS")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        print(f"\n{'='*80}\n")

        # Save as formatted text
        txt_path = output_dir / "grid_search_summary.txt"
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRID SEARCH RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
        print(f"✓ Summary also saved to {txt_path}")

        # Create visualizations
        plot_confusion_matrices(output_dir, configs_run)
        plot_metric_comparisons(output_dir, configs_run)
        create_latex_table(output_dir, configs_run)
    else:
        print("⚠️  No results to summarize")


def interactive_menu():
    """Interactive menu for selecting encoder, configurations, and folds."""
    print("="*80)
    print("INTERACTIVE GRID SEARCH FOR SICAPV2")
    print("="*80)

    # Step 1: Choose encoder
    print("\n" + "="*80)
    print("STEP 1: SELECT ENCODER")
    print("="*80)

    available_encoders = {
        "1": ("facebook/dinov2-base", "Dinov2-Base (86M params, general purpose vision)"),
        "2": ("facebook/dinov2-small", "Dinov2-Small (22M params, general purpose vision)"),
        "3": ("owkin/phikon", "Phikon (86M params, medical imaging Vision Transformer)"),
        "4": ("ikim-uk-essen/BiomedCLIP_ViT_patch16_224", "BiomedCLIP (86M params, biomedical Vision Transformer)"),
        "5": ("/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/UNI_model", "UNI (1.1B params, Histopathology Foundation Model)"),
        "6": ("/Users/salvatorevella/Documents/GitHub/DataScience/CP8321/Project/Virchow2", "Virchow2 (632M params, Histopathology Foundation Model)"),
    }

    print("\nAvailable Encoders:")
    for key, (path, desc) in available_encoders.items():
        print(f"  {key}. {desc}")

    while True:
        choice = input("\nSelect encoder [1-6]: ").strip()
        if choice in available_encoders:
            encoder, encoder_desc = available_encoders[choice]
            print(f"✓ Selected: {encoder_desc}")
            break
        else:
            print("⚠️  Invalid choice. Please enter 1-6.")

    # Step 2: Choose configurations
    print("\n" + "="*80)
    print("STEP 2: SELECT CONFIGURATIONS TO TEST")
    print("="*80)

    print("\nAvailable configurations:")
    for i, config in enumerate(CONFIGURATIONS, 1):
        lora_str = " + LoRA" if config["use_lora"] else ""
        print(f"  {i}. {config['classifier'].upper():<8} (cls pooling){lora_str}")

    print("\nOptions:")
    print("  a. Run ALL configurations (recommended)")
    print("  1-6. Run specific configuration")
    print("  Custom: Enter comma-separated numbers (e.g., '1,2,5')")

    while True:
        choice = input("\nYour choice [a/1-6/custom]: ").strip().lower()

        if choice == 'a':
            configs_to_run = CONFIGURATIONS
            print(f"✓ Selected: ALL {len(CONFIGURATIONS)} configurations")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(CONFIGURATIONS):
            configs_to_run = [CONFIGURATIONS[int(choice)-1]]
            print(f"✓ Selected: {configs_to_run[0]['name']}")
            break
        elif ',' in choice:
            try:
                indices = [int(x.strip())-1 for x in choice.split(',')]
                if all(0 <= i < len(CONFIGURATIONS) for i in indices):
                    configs_to_run = [CONFIGURATIONS[i] for i in indices]
                    print(f"✓ Selected: {len(configs_to_run)} configurations")
                    break
                else:
                    print("⚠️  Invalid indices. Please try again.")
            except ValueError:
                print("⚠️  Invalid format. Use comma-separated numbers (e.g., '1,2,5').")
        else:
            print("⚠️  Invalid choice. Enter 'a', a number 1-6, or comma-separated numbers.")

    # Step 3: Choose number of folds
    print("\n" + "="*80)
    print("STEP 3: SELECT NUMBER OF FOLDS")
    print("="*80)

    print("\nFold options:")
    print("  1. Single fold (fold1) - Quick test (~20-30 min)")
    print("  2. Two folds (fold1-2) - Faster evaluation (~40-60 min)")
    print("  3. Three folds (fold1-3) - Good balance (~60-90 min)")
    print("  4. Four folds (FULL CV) - Most robust (recommended, ~80-120 min)")

    while True:
        choice = input("\nSelect number of folds [1-4]: ").strip()
        if choice in ['1', '2', '3', '4']:
            num_folds = int(choice)
            folds_to_run = list(range(1, num_folds + 1))
            print(f"✓ Selected: {num_folds} fold(s) - {folds_to_run}")
            break
        else:
            print("⚠️  Invalid choice. Please enter 1-4.")

    # Step 4: Training parameters
    print("\n" + "="*80)
    print("STEP 4: TRAINING PARAMETERS")
    print("="*80)

    print("\nRecommended settings:")
    print(f"  Epochs: 6 (neural classifiers)")
    print(f"  Batch size: 16 (use 8 for large models like UNI/Virchow2)")
    print(f"  Learning rate: 1e-4")

    use_defaults = input("\nUse recommended settings? [Y/n]: ").strip().lower()

    if use_defaults in ['', 'y', 'yes']:
        epochs = 6
        batch_size = 16 if choice not in ['5', '6'] else 8  # Smaller batch for large models
        lr = 1e-4
        print(f"✓ Using: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    else:
        epochs = int(input("  Epochs [6]: ").strip() or "6")
        batch_size = int(input("  Batch size [16]: ").strip() or "16")
        lr = float(input("  Learning rate [1e-4]: ").strip() or "1e-4")
        print(f"✓ Using: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Encoder: {encoder_desc}")
    print(f"Configurations: {len(configs_to_run)}")
    for cfg in configs_to_run:
        print(f"  - {cfg['name']}")
    print(f"Folds: {folds_to_run}")
    print(f"Total runs: {len(configs_to_run)} configs × {len(folds_to_run)} folds = {len(configs_to_run) * len(folds_to_run)} runs")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("="*80)

    confirm = input("\nProceed with grid search? [Y/n]: ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("❌ Grid search cancelled.")
        sys.exit(0)

    return encoder, configs_to_run, folds_to_run, epochs, batch_size, lr


def main():
    parser = argparse.ArgumentParser(
        description="Run grid search for foundation models with 4-fold CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python run_grid_search.py

  # Non-interactive mode
  python run_grid_search.py --encoder facebook/dinov2-base --epochs 6 --folds 1,2,3,4
        """
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Encoder to use (skips interactive menu if provided)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Training epochs for neural classifiers"
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
        "--output-dir",
        type=str,
        default="foundation_results/grid_search",
        help="Base output directory for all results"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from configuration index (useful for resuming)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated list of config names to run (e.g., 'linear_cls,rf_cls')"
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of folds to run (e.g., '1,2' or '1,2,3,4')"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive menu, use command-line args only"
    )

    args = parser.parse_args()

    # Interactive mode if encoder not specified
    if args.encoder is None and not args.non_interactive:
        encoder, configs_to_run, folds_to_run, epochs, batch_size, lr = interactive_menu()
    else:
        # Non-interactive mode
        if args.encoder is None:
            print("❌ Error: --encoder required in non-interactive mode")
            sys.exit(1)

        encoder = args.encoder
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr

        # Parse folds
        folds_to_run = [int(f) for f in args.folds.split(',')]

        # Filter configurations if specific ones requested
        if args.configs:
            requested = [c.strip() for c in args.configs.split(',')]
            configs_to_run = [c for c in CONFIGURATIONS if c['name'] in requested]
        else:
            configs_to_run = CONFIGURATIONS[args.start_from:]

        print("="*80)
        print("GRID SEARCH (NON-INTERACTIVE MODE)")
        print("="*80)
        print(f"Encoder: {encoder}")
        print(f"Configurations: {len(configs_to_run)}")
        print(f"Folds: {folds_to_run}")
        print(f"Total runs: {len(configs_to_run) * len(folds_to_run)}")
        print("="*80 + "\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all configurations
    start_time = datetime.now()
    run_results = []

    for i, config in enumerate(configs_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# PROGRESS: {i}/{len(configs_to_run)}")
        print(f"{'#'*80}")

        result = run_configuration(
            config=config,
            encoder=encoder,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_dir=output_dir,
            folds_to_run=folds_to_run
        )
        run_results.append(result)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60

    # Summary
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)")

    successful = sum(1 for r in run_results if r['success'])
    failed = len(run_results) - successful

    print(f"Successful: {successful}/{len(run_results)}")
    print(f"Failed: {failed}/{len(run_results)}")

    if failed > 0:
        print("\nFailed configurations:")
        for r in run_results:
            if not r['success']:
                print(f"  - {r['config']}: {r['error']}")

    print(f"{'='*80}\n")

    # Create summary table
    if successful > 0:
        create_summary_table(output_dir, configs_to_run)

    # Save run log
    log_file = output_dir / "grid_search_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_minutes": total_duration,
            "encoder": encoder,
            "configurations_run": len(run_results),
            "successful": successful,
            "failed": failed,
            "results": run_results
        }, f, indent=2)

    print(f"✓ Run log saved to {log_file}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
