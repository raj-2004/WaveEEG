# WaveEEG  
**Contextual Transformer-Based Modeling of Harmful Brain Activity in EEG**

WaveEEG is a long-context EEG classification framework designed to detect and categorize clinically relevant harmful brain activity patterns in continuous multichannel EEG.  
The approach adapts a speech-pretrained Wave2Vec2.0 transformer to raw EEG signals, enabling contextual modeling over extended temporal windows that are critical for identifying periodic and rhythmic abnormalities.

This repository provides a reference implementation and full experimental analysis corresponding to the WaveEEG framework.

Project repository:  
https://github.com/Deepak-Mewada/WaveEEG

---

## Key Contributions

- Direct modeling of **raw multichannel EEG** without handcrafted feature engineering  
- **Long-context temporal modeling** using a pretrained transformer backbone  
- Adaptation of **speech-pretrained Wave2Vec2.0** to EEG signals  
- Classification of **six ACNS-aligned harmful brain activity patterns**  
- Comprehensive evaluation including accuracy, weighted F1-score, Cohen’s kappa, ROC, PR curves, calibration, and embedding analysis  

---

## Dataset

**HMS Harmful Brain Activity Classification (HBAC)**

- Public Kaggle dataset  
- Continuous ICU EEG recordings  
- Expert consensus labels with vote distributions  
- Six target classes:
  - Seizure  
  - LPD (Lateralized Periodic Discharges)  
  - GPD (Generalized Periodic Discharges)  
  - LRDA (Lateralized Rhythmic Delta Activity)  
  - GRDA (Generalized Rhythmic Delta Activity)  
  - Other  

Dataset usage is subject to the original HMS/Kaggle terms and conditions.

---

## Repository Structure
WaveEEG/
│
├── WaveEEG.py
│ Main training and evaluation script implementing the WaveEEG model
│
├── posthoc_analysis.py
│ Post-hoc analysis and extended visualization utilities
│
├── README.md
│ Repository documentation
│
├── per_class_metrics.csv
│ Per-class precision, recall, F1-score, and support
│
├── Figures (PNG / PDF)
│ ├── 01_accuracy_loss_curves.*
│ ├── 02_f1_kappa_curves.*
│ ├── 03_confusion_matrix.*
│ ├── 04_confusion_matrix_normalized.*
│ ├── 05_per_class_metrics.*
│ ├── 06_roc_curves.*
│ ├── 07_precision_recall_curves.*
│ ├── 07_calibration_curve.*
│ ├── t-SNE visualizations (true labels, confidence, correctness)

All figures are generated automatically by the evaluation pipeline and are suitable for direct inclusion in manuscripts or supplementary material.

---

## Method Overview

WaveEEG operates on fixed-length EEG segments extracted from continuous recordings:

1. Raw EEG segments are standardized channel-wise  
2. Multichannel EEG is reshaped into a one-dimensional sequence compatible with Wave2Vec2.0  
3. A frozen convolutional feature encoder extracts low-level temporal representations  
4. A fine-tuned transformer encoder models long-range temporal dependencies  
5. Mean pooling over time produces a segment-level embedding  
6. A linear classifier predicts one of six harmful brain activity classes  

The design explicitly targets EEG patterns defined by sustained temporal organization over tens of seconds.

---

## Training Protocol

- Optimizer: AdamW  
- Learning rate: 1e-5  
- Loss function: Cross-entropy  
- Feature extractor: Frozen  
- Transformer encoder: Fine-tuned  
- Evaluation metrics:
  - Accuracy  
  - Weighted F1-score  
  - Cohen’s kappa  
  - ROC-AUC (macro and weighted)  
  - Precision–Recall curves  
  - Calibration analysis  

Early stopping is applied based on validation accuracy.
---

## Running the Code

1. Install dependencies:
   ```bash
   pip install torch transformers numpy pandas scikit-learn
   ```

2. Update the dataset path in `WaveEEG.py`:
   ```python
   CONFIG["base_dir"] = "/path/to/hms-hbac"
   ```

3. Train and evaluate the model:
   ```bash
   python WaveEEG.py
   ```

All metrics, figures, and tables are generated automatically.

---

## Reproducibility Notes

- Deterministic data splits are used  
- Random seeds are fixed where applicable  
- All reported figures are generated directly from model outputs  
- No test-time tuning is performed  

---

## Disclaimer

This software is provided for research purposes only.  
Not a medical device.  
Not approved for clinical diagnosis or treatment.  
Outputs must not be used for patient care decisions.

---

## Citation

If you use this code or build upon this work, please cite:

```
Deepak Mewada et al.
Transformer-Based Contextual Modeling of Harmful Brain Activity in Multichannel EEG
2025
```



---

## Contact

For questions or issues related to the implementation, please open a GitHub issue in this repository.

