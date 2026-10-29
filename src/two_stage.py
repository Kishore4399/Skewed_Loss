"""Two-stage bias correction, the paper's baseline ("Approach 2", eqs. 5-7)."""
import numpy as np


def fit_two_stage(val_preds, val_labels):
    """Fit pred = b1 * age + b0 on the validation set. Returns (b1, b0)."""
    b1, b0 = np.polyfit(np.asarray(val_labels, float), np.asarray(val_preds, float), 1)
    return float(b1), float(b0)


def apply_two_stage(preds, b1, b0):
    """Corrected prediction (pred - b0) / b1. Uses no test labels."""
    return (np.asarray(preds, float) - b0) / b1
