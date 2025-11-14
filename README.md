# CP8321 Project 4

Reference implementation for the SICAPv2 end-to-end pipeline described in `p4_end2end_explained.py`. Use this document to set up the Python environment and run the script with sensible defaults.

## Setup
- Use Python 3.10+ (the project was developed with the `DataScience` conda environment).
- Install all dependencies once:
  ```bash
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```
  Feel free to run these inside a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`).

## Running the pipeline
- Make sure your SICAPv2 dataset lives at `path/to/SICAPv2` with `train`, `valid`, and `test` subfolders (each containing `NC`, `G3`, `G4`, `G5` directories).
- Execute the explained pipeline with default settings:
  ```bash
  python p4_end2end_explained.py --data-dir data
  ```
  Swap `--encoder`, `--classifier`, or other CLI flags as needed (see `p4_end2end_explained.py -h`).

## Notes
- The script writes metrics to `foundation_results/p4_end2end_metrics.json` by default; adjust `--output-json` if you want a different location.
- Hugging Face encoders (Dinov2, PHIKON, BiomedCLIP, UNI) are downloaded on first use, so ensure you are logged in (`huggingface-cli login`) if the checkpoint requires authentication.

## Example run

```
(DataScience) 8:44:03>python p4_end2end_explained.py --data-dir data           
[p4_end2end_explained] Loaded SICAPv2 splits: 9959 train / 2487 valid / 2122 test patches.
[p4_end2end_explained] Step 1/4: Converting whole-slide patches into tensor batches.
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.48, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
[p4_end2end_explained] Step 2/4: Training frozen-head classifier.
[p4_end2end_explained] Epoch 1/6 - loss 0.9275 - val acc 0.6273
[p4_end2end_explained] Epoch 2/6 - loss 0.6966 - val acc 0.6799
[p4_end2end_explained] Epoch 3/6 - loss 0.6251 - val acc 0.7181
[p4_end2end_explained] Epoch 4/6 - loss 0.5857 - val acc 0.7193
[p4_end2end_explained] Epoch 5/6 - loss 0.5585 - val acc 0.7354
[p4_end2end_explained] Epoch 6/6 - loss 0.5391 - val acc 0.7523
[p4_end2end_explained] Step 4/4: Evaluating downstream Gleason grading performance.
[p4_end2end_explained] Test metrics:
[p4_end2end_explained]   accuracy: 0.7055
[p4_end2end_explained]   precision_macro: 0.6907
[p4_end2end_explained]   recall_macro: 0.6332
[p4_end2end_explained]   f1_macro: 0.6412
[p4_end2end_explained]   kappa_quadratic: 0.6674
[p4_end2end_explained]   confusion_matrix: [[606, 24, 12, 2], [99, 225, 62, 7], [138, 96, 590, 29], [47, 0, 109, 76]]
[p4_end2end_explained]   auc_weighted: 0.8944
[p4_end2end_explained] Saved metrics to foundation_results/p4_end2end_metrics.json
(DataScience) 9:37:55>

```
