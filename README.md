# Whale Well Whale 🐋

Comparison of **ResNet18 on log-mel spectrograms** vs. **BEATs on audio embeddings** for North Atlantic Right Whale call detection.

## Preprocessing

Two input settings were compared for both model families:

- **Full-band input**
- **Filtered input (50-350 Hz)**

For ResNet18, audio was converted to log-mel spectrograms.  
For BEATs, the waveform pipeline used the same frequency restriction through bandpass filtering, and the best setup used **precomputed filtered embeddings**. This preprocessing setup is the core experimental design of the project. 

### Spectrogram examples
<img width="4155" height="4483" alt="figure_spectrogram_before_after_preprocessing" src="https://github.com/user-attachments/assets/75aa56d8-021f-4404-a615-62d09687c59b" />


### Waveform / filtering examples
<img width="4144" height="3778" alt="figure_waveform_before_after_filtering" src="https://github.com/user-attachments/assets/1e2b3b8c-2718-4d15-bd0f-99414dd6d964" />

## Embedding Space Comparison

The figure below shows a PCA projection of the learned feature spaces for both model families, before and after preprocessing.  
It gives a compact view of class separability in the embedding space: **BEATs shows clearer separation between whale calls and noise than ResNet18**, and the filtered pipeline further improves this structure.


<img width="3570" height="1485" alt="resnet18_pca_comparison" src="https://github.com/user-attachments/assets/ab1ef6bc-7057-4bfb-830e-1aab589d1d2e" />
<img width="3570" height="1485" alt="beats_pca_comparison" src="https://github.com/user-attachments/assets/e79e4c06-9206-49ca-b326-8c49ef6e2860" />
---

## Results

The main result is clear: **BEATs outperformed ResNet18**, and the best overall setup was:

**BEATs + precomputed embeddings + 50-350 Hz filtering**

### Best test metrics
- **ROC-AUC:** 0.962
- **PR-AUC:** 0.825
- **Best F1:** 0.765
- **Recall:** 0.744
- **Precision:** 0.779

These values come from the final saved test summary of the best BEATs filtered run.
<img width="944" height="754" alt="image" src="https://github.com/user-attachments/assets/b3b65aa9-615a-472a-ad86-8c1dc01dfe1b" />


## Experimental Setup

- **Dataset:** 2-second `.aif` clips, sampled at **2 kHz**
- **Labels:** inferred from filenames (`*_1.aif` = whale, `*_0.aif` = noise)
- **Split:** **80% train / 10% val / 10% test**, fixed across all experiments
- **Training:** 100 epochs, batch size 32, AdamW, learning rate `1e-4`, weight decay `1e-4`
- **Backbones:** frozen, with lightweight classification heads
source: https://www.kaggle.com/competitions/the-icml-2013-whale-challenge-right-whale-redux/data
These settings were shared across the main comparison runs. 

