# -*- coding: utf-8 -*-
"""
WaveEEG: Contextual Transformer-Based Modeling of Harmful Brain Activity in EEG

Reference implementation accompanying the manuscript:
"Transformer-Based Contextual Modeling of Harmful Brain Activity in Multichannel EEG"

This script implements a long-context EEG classification framework that adapts
a speech-pretrained Wave2Vec2.0 model to raw multichannel EEG recordings.
The design follows the methodological choices described in the paper and is
intended for reproducibility and research transparency.

IMPORTANT NOTES:
- Research use only.
- Not a medical device.
- Not intended for clinical diagnosis or treatment decisions.

Dataset:
HMS – Harmful Brain Activity Classification (HBAC)

Author: Deepak et al.
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
# Standard libraries
import os
import copy
import warnings
warnings.filterwarnings("ignore")

# Scientific computing
import numpy as np
import pandas as pd

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

# Evaluation utilities
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# HuggingFace Transformers
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# ---------------------------------------------------------------------
# Configuration (paper-aligned experimental settings)
# ---------------------------------------------------------------------
CONFIG = {
    # Root directory containing HMS dataset files
    "base_dir": "/path/to/hms-hbac",  # <-- UPDATE THIS PATH

    # Pretrained speech model used as temporal feature extractor
    "model_name": "facebook/wav2vec2-base",

    # Number of ACNS-defined classes
    "num_classes": 6,

    # EEG acquisition parameters
    "sampling_rate": 200,              # Hz
    "segment_duration_sec": 50,         # seconds

    # Training hyperparameters
    "batch_size": 128,
    "num_epochs": 100,
    "learning_rate": 1e-5,
    "num_workers": 4,
}

# Standard 10–20 EEG montage used in HMS dataset
EEG_CHANNELS = [
    'Fp1','F3','C3','P3','F7','T3','T5','O1',
    'Fz','Cz','Pz',
    'Fp2','F4','C4','P4','F8','T4','T6','O2'
]

# Mapping from clinical labels to integer class indices
CLASS_MAP = {
    "seizure": 0,
    "lpd": 1,
    "gpd": 2,
    "lrda": 3,
    "grda": 4,
    "other": 5,
}


# ---------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------
class EEGDataset(Dataset):
    """
    Dataset class for HMS EEG segments.

    Each sample corresponds to a 50-second multichannel EEG segment centered
    around an expert-labeled event. Signals are independently standardized
    per segment to reflect realistic clinical variability and to avoid
    information leakage across recordings.
    """

    def __init__(self, meta_df, base_dir, processor):
        """
        Parameters
        ----------
        meta_df : pandas.DataFrame
            Metadata table containing EEG file identifiers and label offsets.
        base_dir : str
            Root directory containing EEG parquet files.
        processor : Wav2Vec2Processor
            HuggingFace processor for waveform normalization and padding.
        """
        self.meta_df = meta_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.processor = processor
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.meta_df)

    def _load_eeg(self, eeg_id):
        """
        Load a single EEG recording from disk.

        Parameters
        ----------
        eeg_id : str or int
            Unique identifier of the EEG recording.

        Returns
        -------
        pandas.DataFrame
            Multichannel EEG signal.
        """
        path = os.path.join(self.base_dir, "train_eegs", f"{eeg_id}.parquet")
        eeg = pd.read_parquet(path)[EEG_CHANNELS]
        return eeg

    def __getitem__(self, idx):
        """
        Retrieve a single EEG segment and its label.

        Steps:
        1. Load the full EEG recording.
        2. Extract the 50-second window aligned with expert annotation.
        3. Apply per-channel z-score normalization.
        4. Concatenate channels along the temporal axis.
        5. Convert to Wave2Vec2-compatible input format.
        """
        row = self.meta_df.loc[idx]

        eeg = self._load_eeg(row["eeg_id"])
        start = int(row["eeg_label_offset_seconds"] * CONFIG["sampling_rate"])
        end = start + CONFIG["sampling_rate"] * CONFIG["segment_duration_sec"]

        # Shape: (channels, time)
        segment = eeg.iloc[start:end].to_numpy().T

        # Per-segment normalization
        segment = self.scaler.fit_transform(segment.T).T

        # Flatten channels along the temporal dimension
        segment = segment.reshape(1, -1)

        label = CLASS_MAP[row["expert_consensus"].lower()]

        inputs = self.processor(
            segment.squeeze(),
            sampling_rate=16000,   # Required by Wave2Vec2
            return_tensors="pt",
            padding=True,
        )

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------
class WaveEEG(Wav2Vec2ForSequenceClassification):
    """
    WaveEEG model.

    Architecture:
    - Wave2Vec2.0 convolutional feature encoder (frozen)
    - Transformer encoder for long-context temporal modeling
    - Mean pooling over time
    - Linear classification head

    This design explicitly targets sustained EEG patterns defined by
    ACNS criteria, rather than short transient events.
    """

    loss_fct = None

    @classmethod
    def from_pretrained(cls, name, num_labels, criterion):
        """
        Load a pretrained Wave2Vec2 model and attach a task-specific classifier.
        """
        cls.loss_fct = criterion
        return super().from_pretrained(name, num_labels=num_labels)

    def forward(self, input_values, labels=None, **kwargs):
        """
        Forward pass with mean temporal pooling.
        """
        outputs = self.wav2vec2(
            input_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # Contextual representations from transformer encoder
        hidden = outputs.last_hidden_state

        # Temporal aggregation
        pooled = hidden.mean(dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


# ---------------------------------------------------------------------
# Evaluation routine
# ---------------------------------------------------------------------
def evaluate(model, loader, device):
    """
    Evaluate model performance on a given dataset split.

    Metrics reported:
    - Accuracy
    - Weighted F1-score
    - Cohen's kappa (inter-rater agreement proxy)
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)

            out = model(x)
            preds = out.logits.argmax(dim=-1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)

    return acc, f1, kappa


# ---------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata and shuffle deterministically
    meta = pd.read_csv(os.path.join(CONFIG["base_dir"], "train.csv"))
    meta = meta.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Train / validation / test split
    n = len(meta)
    train_df = meta[: int(0.8 * n)]
    val_df   = meta[int(0.8 * n): int(0.9 * n)]
    test_df  = meta[int(0.9 * n):]

    processor = Wav2Vec2Processor.from_pretrained(CONFIG["model_name"])

    train_ds = EEGDataset(train_df, CONFIG["base_dir"], processor)
    val_ds   = EEGDataset(val_df, CONFIG["base_dir"], processor)
    test_ds  = EEGDataset(test_df, CONFIG["base_dir"], processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )
    val_loader  = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

    # Loss and model initialization
    criterion = CrossEntropyLoss()

    model = WaveEEG.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_classes"],
        criterion=criterion,
    ).to(device)

    # Freeze convolutional feature extractor to prevent overfitting
    for p in model.wav2vec2.feature_extractor.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    best_val = 0.0

    # Training loop
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            out = model(x, labels=y)
            out.loss.backward()
            optimizer.step()

        val_acc, val_f1, val_k = evaluate(model, val_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            model.save_pretrained("waveeeg_best")

        print(
            f"Epoch {epoch:03d} | "
            f"Val Acc {val_acc:.4f} | "
            f"F1 {val_f1:.4f} | "
            f"Kappa {val_k:.4f}"
        )

    # Final evaluation on held-out test set
    model = WaveEEG.from_pretrained(
        "waveeeg_best",
        num_labels=CONFIG["num_classes"],
        criterion=criterion,
    ).to(device)

    test_acc, test_f1, test_k = evaluate(model, test_loader, device)

    print("\nFINAL TEST PERFORMANCE")
    print(f"Accuracy     : {test_acc:.4f}")
    print(f"F1-score     : {test_f1:.4f}")
    print(f"Cohen's kappa: {test_k:.4f}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
