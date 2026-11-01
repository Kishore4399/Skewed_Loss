"""MAE and age-delta correlation (ADC)."""
import numpy as np


def mae(preds, labels):
    return float(np.mean(np.abs(np.asarray(preds) - np.asarray(labels))))


def adc(preds, labels):
    """Pearson correlation between the delta (pred - age) and the true age. Negative = regression to the mean."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    delta = preds - labels
    if np.std(delta) == 0 or np.std(labels) == 0:
        return 0.0
    return float(np.corrcoef(delta, labels)[0, 1])
