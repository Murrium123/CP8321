# CP8321 Project 4: Foundation Models for Prostate Cancer Grading

Source code that accompanies the report for Project 4 in course CP8321. The project is titled: A Study of Gleason Grading using Foundation Models on the SICAPv2 Dataset

## Quick Command Reference

```bash
# 1. Install dependencies
install -r requirements.txt

# 2. Parse data SICAPv2 dataset after it has been loaded and create the data
#    directories
python organize_sicapv2_4fold.py

# 3. Run experiments - Examples below
python p4_driver.py --experiments A --folds 1              # Quick test (Experiment A, fold 1) 
							# (about 10 minutes on a Macbook Pro M4 with 48 MB memory)
python p4_driver.py                                        # Run ALL experiments 
							# (about 48 hours on a Macbook Pro M4 with 48 MB memory)

```

## Quick Start

### Step 1: Install Dependencies

Create a virtual environment and install required packages:

```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download the SICAPv2 Dataset

Download from Mendeley Data: https://data.mendeley.com/datasets/9xxm58dvs3/1

1. Download the zip file from the link above
2. Extract to this repository root (creates `./SICAPv2/`)

```

### Step 3: Organize Dataset for 4-Fold Cross-Validation

Edit paths in `organize_sicapv2_4fold.py`:

```python
SICAP_ROOT = '/path/to/your/SICAPv2'      # Update to your SICAPv2 location (line 60)
OUTPUT_DIR = '/path/to/output/data'       # Update to desired output location (line 62)
```

Run the organization script:

```bash
python organize_sicapv2_4fold.py
```

### Step 3: Run Your First Experiment

Verify everything works with a quick test:

```bash
python p4_driver.py --experiments A --folds 1
```

## Project Files

```
CP8321/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── organize_sicapv2_4fold.py           # Dataset organization script
├── p4_driver.py                        # Experiment driver (main entry point)
├── p4_end2end_explained_simple.py      # Core training pipeline
├── hf_processor_loader.py              # Model/processor loading utilities
│
├── data/                               # Organized dataset (you create this)
│   ├── test/                           # Shared test set
│   ├── fold1/ ... fold4/               # Cross-validation splits
│
├── run_results-GOLDEN/                 # Results used in paper - FROZEN
│   ├── A_cls/, A_mean/                 # Dinov2 baseline
│   ├── B_cls/, B_mean/                 # Phikon baseline
│   └── C1_cls/ ... C5_cls/             # LoRA / non-LoRA runs
│
└── foundation_results/                 # New experiment outputs
    └── p4_driver_runs/
        ├── A_cls/, A_mean/             # Your new results
        ├── C1_cls/ ... C4_cls/
        └── ...
```

