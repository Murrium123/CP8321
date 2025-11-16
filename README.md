# CP8321 Project 4: Foundation Models for Prostate Cancer Grading

End-to-end pipeline for evaluating foundation models on the SICAPv2 dataset with 4-fold cross-validation. This project tests multiple vision encoders (Dinov2, PHIKON, BiomedCLIP, UNI, Virchow2) with various classifier heads for Gleason grading.

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
2. Extract to a local directory (e.g., `./SICAPv2`)
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

Start with a small experiment to verify everything works. The grid search script has an **interactive menu** that guides you through configuration:

```bash
python run_grid_search.py
```

**Recommended first experiment:**
- **Encoder**: `facebook/dinov2-base` (smallest, fastest)
- **Configurations**: Select `1` (linear classifier only)
- **Folds**: Select `1` (single fold for quick test)
- **Parameters**: Use defaults (epochs=6, batch_size=16, lr=1e-4)

This will take approximately **20-30 minutes** and verify:
- Dataset is properly organized
- Dependencies are correctly installed
- GPU/CPU is working
- Results are being saved

**Expected output:**
```
================================================================================
STEP 1: SELECT ENCODER
================================================================================
Available Encoders:
  1. Dinov2-Base (86M params, general purpose vision)
  2. Dinov2-Small (22M params, general purpose vision)
  3. Phikon (86M params, medical imaging Vision Transformer)
  4. BiomedCLIP (86M params, biomedical Vision Transformer)
  5. UNI (1.1B params, Histopathology Foundation Model)
  6. Virchow2 (632M params, Histopathology Foundation Model)

Select encoder [1-6]: 1
✓ Selected: Dinov2-Base (86M params, general purpose vision)
...
```

After completion, check the results:

```bash
# View summary
cat foundation_results/grid_search/grid_search_summary.txt

# View detailed metrics
cat foundation_results/grid_search/linear_cls/fold1_results.json
```

## Running Full Experiments

Once your first experiment succeeds, scale up to comprehensive evaluations:

### Full 4-Fold Cross-Validation

Run all configurations on all 4 folds:

```bash
python run_grid_search.py
```

Select:
- Encoder of your choice
- **a** (all configurations)
- **4** (all 4 folds)

This tests 6 configurations × 4 folds = **24 total runs**:
1. `linear_cls` - Linear classifier
2. `linear_cls_lora` - Linear classifier + LoRA fine-tuning
3. `mlp_cls` - MLP classifier (2-layer)
4. `mlp_cls_lora` - MLP classifier + LoRA fine-tuning
5. `rf_cls` - Random Forest (200 trees)
6. `svm_cls` - SVM with RBF kernel

**Time estimates (Dinov2-Base on GPU):**
- Single fold: ~20-30 min
- Full 4-fold CV: ~80-120 min

### Non-Interactive Mode

For scripting or running on remote servers:

```bash
# Run specific configurations on specific folds
python run_grid_search.py \
  --encoder facebook/dinov2-base \
  --configs "linear_cls,rf_cls" \
  --folds "1,2,3,4" \
  --epochs 6 \
  --batch-size 16 \
  --non-interactive

# Run all configs on fold 1 only
python run_grid_search.py \
  --encoder owkin/phikon \
  --folds "1" \
  --epochs 6 \
  --non-interactive
```

## Understanding the Results

After a successful grid search, you'll get:

1. **Summary Table** (`grid_search_summary.txt`)
   - Performance comparison across all configurations
   - Mean ± std for accuracy, F1, kappa, AUC

2. **Confusion Matrices** (`confusion_matrices.png`)
   - Visual comparison of classification patterns
   - Averaged across folds

3. **Metric Comparison Charts** (`metric_comparisons.png`)
   - Bar charts comparing accuracy, F1, kappa, AUC
   - Color-coded by LoRA usage

4. **LaTeX Table** (`results_table.tex`)
   - Ready to paste into your paper

5. **Individual Fold Results** (per config folder)
   - `fold1_results.json`, `fold2_results.json`, etc.
   - Detailed metrics, confusion matrices, predictions

## Advanced Usage

### Single Model Evaluation

To test a single configuration without grid search:

```bash
python p4_end2end_explained.py \
  --data-dir data/fold1 \
  --test-dir data/test \
  --encoder facebook/dinov2-base \
  --classifier linear \
  --pooling cls \
  --epochs 6 \
  --batch-size 16 \
  --output-json results/my_experiment.json
```

### Testing Different Encoders

Available encoders:
- `facebook/dinov2-base` - General-purpose vision (86M params)
- `facebook/dinov2-small` - Smaller version (22M params)
- `owkin/phikon` - Medical imaging specialist (86M params)
- `ikim-uk-essen/BiomedCLIP_ViT_patch16_224` - Biomedical vision (86M params)
- Local paths for UNI and Virchow2 (see `run_grid_search.py`)

### Adjusting Hyperparameters

For LoRA fine-tuning:

```bash
python p4_end2end_explained.py \
  --encoder facebook/dinov2-base \
  --classifier linear \
  --use-lora \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --epochs 6
```

For sklearn classifiers:

```bash
# Random Forest
python p4_end2end_explained.py \
  --classifier rf \
  --rf-estimators 200

# SVM
python p4_end2end_explained.py \
  --classifier svm \
  --svm-c 1.0
```

## Project Structure

```
CP8321/
├── requirements.txt              # All Python dependencies
├── organize_sicapv2_4fold.py     # Dataset organization for 4-fold CV
├── p4_end2end_explained.py       # Single-run pipeline (heavily commented)
├── run_grid_search.py            # Grid search with interactive menu
├── hf_processor_loader.py        # Utilities for loading HF models
└── foundation_results/           # Output directory
    └── grid_search/
        ├── grid_search_summary.txt
        ├── grid_search_summary.csv
        ├── confusion_matrices.png
        ├── metric_comparisons.png
        ├── results_table.tex
        ├── grid_search_log.json
        └── [config_name]/
            ├── fold1_results.json
            ├── fold2_results.json
            ├── fold3_results.json
            ├── fold4_results.json
            └── 4fold_summary.json
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
