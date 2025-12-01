# CP8321 Project 4: Foundation Models for Prostate Cancer Grading

End-to-end pipeline for evaluating foundation models on the SICAPv2 dataset with 4-fold cross-validation. This project tests multiple vision encoders (Dinov2, PHIKON, UNI, Virchow2) with various classifier heads for Gleason grading.

## Quick Command Reference

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Organize dataset
python organize_sicapv2_4fold.py

# 3. Run experiments
python p4_driver.py --experiments A --folds 1              # Quick test (Experiment A, fold 1)
python p4_driver.py --experiments A,B --folds 1,2,3,4      # Full baseline comparison
python p4_driver.py --experiments C1,C2 --folds 1          # LoRA experiments
python p4_driver.py                                         # Run ALL experiments

# 4. Direct training (advanced)
python p4_end2end_explained_simple.py --encoder facebook/dinov2-base --classifier linear --data-dir data/fold1
```

## Prerequisites

- Python 3.10 or higher
- 10+ GB disk space for dataset and models
- GPU recommended (CUDA or Apple Metal) but not required

## Quick Start

Follow these steps to get up and running with your first experiment:

### Step 1: Install Dependencies

Create a virtual environment (recommended) and install all required packages:

```bash
# Optional: Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` includes:
- PyTorch for deep learning
- Hugging Face Transformers for foundation models
- scikit-learn for classical ML classifiers
- pandas/openpyxl for dataset loading
- matplotlib/seaborn for visualization

### Step 2: Download the SICAPv2 Dataset

1. Download from Mendeley Data: https://data.mendeley.com/datasets/9xxm58dvs3/1
   - If you prefer the CLI, download and unzip directly into this repository root:
     ```bash
     curl -L -o SICAPv2.zip https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9xxm58dvs3-1.zip
     unzip SICAPv2.zip -d .
     rm SICAPv2.zip
     ```
2. Ensure the extracted folder lives in this directory (e.g., `./SICAPv2`)
3. Verify the structure:
   ```
   SICAPv2/
   ├── images/           # ~14,500 histopathology patches
   └── partition/        # Excel files with train/val/test splits
       ├── Test/
       └── Validation/
           ├── Val1/
           ├── Val2/
           ├── Val3/
           └── Val4/
   ```

### Step 3: Organize Dataset for 4-Fold Cross-Validation

Edit the paths in `organize_sicapv2_4fold.py`:

```python
SICAP_ROOT = '/path/to/your/SICAPv2'      # Update this
OUTPUT_DIR = '/path/to/output/data'       # Update this
```

Then run the organization script:

```bash
python organize_sicapv2_4fold.py
```

This creates the following structure:

```
data/
├── test/              # Shared test set (2,122 patches, 31 slides)
│   ├── NC/
│   ├── G3/
│   ├── G4/
│   └── G5/
├── fold1/
│   ├── train/         # 7,472 patches
│   └── valid/         # 2,487 patches
├── fold2/
│   ├── train/         # 7,793 patches
│   └── valid/         # 2,166 patches
├── fold3/
│   ├── train/         # 8,166 patches
│   └── valid/         # 1,793 patches
└── fold4/
    ├── train/         # 6,446 patches
    └── valid/         # 3,513 patches
```

### Step 4: Run Your First Experiment

Start with a small experiment to verify everything works. The driver script runs pre-defined experiments:

```bash
python p4_driver.py --experiments A --folds 1
```

**What this does:**
- Runs **Experiment A** (Dinov2-Base baseline)
- Tests 3 configurations: linear, MLP, and Random Forest classifiers
- Tests both CLS and mean pooling
- Uses only **Fold 1** for quick validation
- Should complete in **~30-40 minutes** on GPU

This verifies:
- Dataset is properly organized
- Dependencies are correctly installed
- GPU/CPU is working
- Results are being saved

**Expected output:**
```
=== Exp A (DINO-v2 baseline), config linear_cls, pooling cls, fold 1 ===
python p4_end2end_explained_simple.py --data-dir data/fold1 --test-dir data/test ...
...
✓ Experiment completed successfully
```

After completion, check the results:

```bash
# View fold results
cat foundation_results/p4_driver_runs/A_cls/linear_cls/fold1_results.json

# View all experiment outputs
ls -R foundation_results/p4_driver_runs/
```

## Running Full Experiments

Once your first experiment succeeds, scale up to comprehensive evaluations:

### Available Pre-Defined Experiments

The driver includes several carefully designed experiments:

- **Experiment A**: Dinov2-Base baseline (linear, MLP, RF classifiers)
- **Experiment B**: Phikon baseline (linear, MLP classifiers)
- **Experiment C1**: Dinov2-Base LoRA vs frozen comparison
- **Experiment C2**: Phikon LoRA vs frozen comparison
- **Experiment C3**: UNI LoRA vs frozen comparison (requires local model)
- **Experiment C4**: Virchow2 LoRA vs frozen comparison (requires local model)

### Run All Experiments on All Folds

```bash
# Run everything (all experiments, all 4 folds)
python p4_driver.py
```

This will execute all experiments across all 4 folds. Estimated time: **6-12 hours** depending on GPU.

### Run Specific Experiments

```bash
# Run just Experiment A on all 4 folds
python p4_driver.py --experiments A --folds 1,2,3,4

# Run Experiments C1 and C2 on fold 1 only
python p4_driver.py --experiments C1,C2 --folds 1

# Run Experiment B with custom hyperparameters
python p4_driver.py --experiments B --epochs 10 --batch-size 32 --lr 5e-5
```

### Customize Hyperparameters

```bash
# Adjust LoRA parameters for all experiments
python p4_driver.py \
  --experiments C1,C2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1

# Use early stopping
python p4_driver.py \
  --experiments A \
  --early-stop-patience 2 \
  --early-stop-metric f1_macro
```

## Understanding the Results

After running experiments with `p4_driver.py`, results are organized by experiment and configuration:

```
foundation_results/p4_driver_runs/
├── A_cls/                    # Experiment A with CLS pooling
│   ├── linear_cls/
│   │   ├── fold1_results.json
│   │   ├── fold2_results.json
│   │   ├── fold3_results.json
│   │   ├── fold4_results.json
│   │   └── 4fold_summary.json
│   ├── mlp_cls/
│   └── rf_cls/
├── A_mean/                   # Experiment A with mean pooling
│   ├── linear_cls/
│   ├── mlp_cls/
│   └── rf_cls/
├── C1_cls/                   # Experiment C1 (LoRA comparison)
│   ├── linear_cls/
│   └── linear_cls_lora/
└── ...
```

### Pre-Existing Results

**Note**: This repository includes pre-computed results in the `run_results/` directory from previous experimental runs. These results follow the same structure as above and include:
- Experiments A and B (baseline comparisons)
- Experiments D1-D5 (additional model evaluations)
- Complete 4-fold cross-validation results for each configuration

You can examine these results without re-running experiments:
```bash
# View a completed experiment
cat run_results/A_cls/linear_cls/4fold_summary.json

# List all available pre-computed results
ls -R run_results/
```

### Individual Fold Results

Each `fold*_results.json` file contains:
- **Metrics**: Accuracy, F1 (macro/weighted), Cohen's Kappa, AUC
- **Confusion Matrix**: Per-class predictions
- **Test Predictions**: Model outputs for test set
- **Training History**: Loss and validation metrics per epoch

Each configuration also includes a `4fold_summary.json` with aggregated statistics across all folds.

## Advanced Usage

### Direct Single Model Evaluation

You can bypass the driver and call the training script directly for custom experiments:

```bash
python p4_end2end_explained_simple.py \
  --data-dir data/fold1 \
  --test-dir data/test \
  --encoder facebook/dinov2-base \
  --classifier linear \
  --pooling cls \
  --epochs 6 \
  --batch-size 16 \
  --lr 1e-4 \
  --output-json results/custom_experiment.json
```

### Available Encoders

The code supports these foundation models:
- `facebook/dinov2-base` - General-purpose vision (86M params)
- `facebook/dinov2-small` - Smaller version (22M params)
- `owkin/phikon` - Medical imaging specialist (86M params)
- `ikim-uk-essen/BiomedCLIP_ViT_patch16_224` - Biomedical vision (86M params)
- Local paths for UNI (1.1B params) and Virchow2 (632M params) - see `p4_driver.py` for paths

### Custom Experiments with LoRA

```bash
python p4_end2end_explained_simple.py \
  --data-dir data/fold1 \
  --encoder facebook/dinov2-base \
  --classifier mlp \
  --use-lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --epochs 10 \
  --early-stop-patience 2
```

### Using Different Classifiers

```bash
# Random Forest (sklearn)
python p4_end2end_explained_simple.py \
  --encoder owkin/phikon \
  --classifier rf \
  --rf-estimators 200

# MLP with 2 hidden layers
python p4_end2end_explained_simple.py \
  --encoder facebook/dinov2-base \
  --classifier mlp \
  --pooling mean
```

## Project Structure

```
CP8321/
├── requirements.txt                    # All Python dependencies
├── organize_sicapv2_4fold.py           # Dataset organization for 4-fold CV
├── p4_driver.py                        # Main experiment driver with pre-defined experiments
├── p4_end2end_explained_simple.py      # Core training pipeline (called by driver)
├── hf_processor_loader.py              # Utilities for loading HF models and processors
├── data/                               # Organized dataset (created by organize script)
│   ├── test/                          # Shared test set
│   ├── fold1/
│   ├── fold2/
│   ├── fold3/
│   └── fold4/
├── run_results/                        # Pre-computed results from previous runs
│   ├── A_cls/                         # Dinov2-Base baseline (CLS pooling)
│   ├── A_mean/                        # Dinov2-Base baseline (mean pooling)
│   ├── B_cls/                         # Phikon baseline (CLS pooling)
│   ├── B_mean/                        # Phikon baseline (mean pooling)
│   └── D1_cls/ ... D5_cls/            # Additional experiments
└── foundation_results/                 # New experiment outputs (created when you run p4_driver.py)
    └── p4_driver_runs/
        ├── A_cls/                     # Experiment A with CLS pooling
        │   ├── linear_cls/
        │   │   ├── fold1_results.json
        │   │   ├── fold2_results.json
        │   │   ├── fold3_results.json
        │   │   ├── fold4_results.json
        │   │   └── 4fold_summary.json
        │   ├── mlp_cls/
        │   └── rf_cls/
        ├── A_mean/                    # Experiment A with mean pooling
        ├── C1_cls/                    # LoRA comparison experiments
        └── ...
```

## Troubleshooting

**Issue: "No module named 'transformers'"**
- Solution: `pip install -r requirements.txt`

**Issue: "Images directory not found"**
- Solution: Update `SICAP_ROOT` in `organize_sicapv2_4fold.py` to point to your extracted SICAPv2 folder

**Issue: "CUDA out of memory"**
- Solution: Reduce batch size (try `--batch-size 8` or `--batch-size 4`)

**Issue: Hugging Face authentication error**
- Solution: Some models require login: `huggingface-cli login`

**Issue: Slow training on CPU**
- Solution: This is expected. GPU highly recommended. Consider reducing `--epochs` for testing.

## Citation

If you use this code or the SICAPv2 dataset, please cite:

```bibtex
@article{silva2020sicap,
  title={SICAP: A comprehensive histopathology dataset for prostate cancer grading},
  author={Silva-Rodr{\'\i}guez, Julio and others},
  journal={Medical Image Analysis},
  year={2020}
}
```

## License

This project is for academic use. Please refer to individual model licenses for the foundation models used.
