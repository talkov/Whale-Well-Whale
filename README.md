# Whale Well Whale 🐋

Comparison of two deep-learning pipelines for North Atlantic Right Whale call detection:

1. ResNet18 trained on log-mel spectrograms.
2. BEATs-based audio representations trained with a lightweight classification head.

The project evaluates how model family and preprocessing affect binary whale-call detection performance.

## Task

The task is binary classification of 2-second audio clips:

```text
*_1.aif -> whale call
*_0.aif -> noise / no whale call
```

The dataset used in this project is the ICML 2013 Right Whale Redux dataset:

https://www.kaggle.com/competitions/the-icml-2013-whale-challenge-right-whale-redux/data

## Repository structure

```text
configs/      YAML configuration files for each experiment
data/         Dataset construction and dataloader logic
models/       Model definitions and classification heads
scripts/      Preprocessing scripts
train.py      Main training and evaluation entry point
README.md     Setup and reproduction instructions
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate
```

```bash
# Windows
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The expected `requirements.txt` is:

```txt
numpy
pandas
tqdm
PyYAML
scikit-learn
torch
torchvision
librosa
soundfile
scipy
opencv-python
Pillow
wandb
```

## External BEATs dependency

The BEATs experiments require a local copy of the BEATs codebase and a BEATs checkpoint.

Update the relevant BEATs config file with your local paths:

```yaml
beats_repo_dir: /path/to/BEATs
beats_checkpoint_path: /path/to/BEATs_iter3_plus_AS2M.pt
```

The current project imports the BEATs implementation from `beats_repo_dir`.

## Dataset preparation

Download the dataset from Kaggle and place the audio files in a local directory.

Then update the dataset path inside the relevant YAML config file under `configs/`.

Example:

```yaml
data_dir: /path/to/right-whale-data
```

Labels are inferred directly from the filename suffix:

```text
*_1.aif -> positive whale-call sample
*_0.aif -> negative noise sample
```

## Experimental design

Two input settings are compared:

1. Full-band input.
2. Filtered input in the 50-350 Hz frequency band.

For ResNet18, audio is converted into log-mel spectrograms.

For BEATs, the waveform pipeline uses the same 50-350 Hz restriction through bandpass filtering. The best-performing BEATs setup uses precomputed filtered embeddings.

The train/validation/test split is fixed across experiments so that ResNet18 and BEATs are compared on the same samples.

## Running experiments

### Train ResNet18 on spectrograms

```bash
python train.py --config configs/resnet18_config.yaml --model_type resnet18
```

### Train BEATs on waveforms

```bash
python train.py --config configs/beats_config.yaml --model_type beats
```

### Train BEATs using precomputed embeddings

```bash
python train.py --config configs/beats_precomputed_config.yaml --model_type beats
```

Use the exact config names that exist in your local `configs/` directory.

## Outputs

Each training run saves checkpoints and evaluation outputs according to the paths defined in the YAML config.

Typical outputs include:

```text
checkpoints/
outputs/
```

The training script logs metrics such as loss, ROC-AUC, PR-AUC, F1, precision, and recall. If Weights & Biases is enabled in the config, metrics are also logged to W&B.

## Preprocessing examples

### Spectrogram examples

<img width="4155" height="4483" alt="figure_spectrogram_before_after_preprocessing" src="https://github.com/user-attachments/assets/75aa56d8-021f-4404-a615-62d09687c59b" />

### Waveform and filtering examples

<img width="4144" height="3778" alt="figure_waveform_before_after_filtering" src="https://github.com/user-attachments/assets/1e2b3b8c-2718-4d15-bd0f-99414dd6d964" />

## Embedding-space analysis

The following PCA projections compare the learned feature spaces for ResNet18 and BEATs before and after preprocessing.

The visualization is used as a qualitative diagnostic for class separability in the learned representation space.

<img width="3570" height="1485" alt="resnet18_pca_comparison" src="https://github.com/user-attachments/assets/ab1ef6bc-7057-4bfb-830e-1aab589d1d2e" />

<img width="3570" height="1485" alt="beats_pca_comparison" src="https://github.com/user-attachments/assets/e79e4c06-9206-49ca-b326-8c49ef6e2860" />

## Result summary

The strongest setup in the experiments was:

```text
BEATs + precomputed embeddings + 50-350 Hz filtering
```

Best saved test metrics:

```text
ROC-AUC:    0.962
PR-AUC:     0.825
Best F1:    0.765
Recall:     0.744
Precision:  0.779
```

<img width="944" height="754" alt="best_beats_filtered_results" src="https://github.com/user-attachments/assets/b3b65aa9-615a-472a-ad86-8c1dc01dfe1b" />

For full experimental motivation, plots, comparison tables, and discussion, see the accompanying technical report.

## Main experiment settings

The main comparison runs use:

```text
Dataset:       2-second .aif clips sampled at 2 kHz
Split:         80% train / 10% validation / 10% test
Training:      100 epochs
Batch size:    32
Optimizer:     AdamW
Learning rate: 1e-4
Weight decay:  1e-4
Backbones:     frozen, with lightweight classification heads
```

These settings are controlled through the YAML config files.
