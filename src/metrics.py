"""MAE and age-delta correlation (ADC)."""
import numpy as np
from scipy import stats


def mae(preds, labels):
    return float(np.mean(np.abs(np.asarray(preds) - np.asarray(labels))))


def adc(preds, labels, kind='pearson'):
    """Correlation between the delta (pred - age) and the true age. Negative = regression to the mean."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    delta = preds - labels
    if np.std(delta) == 0 or np.std(labels) == 0:
        return 0.0
    if kind == 'pearson':
        return float(np.corrcoef(delta, labels)[0, 1])
    if kind == 'spearman':
        return float(stats.spearmanr(delta, labels)[0])
    raise ValueError(f'unknown correlation {kind}')
