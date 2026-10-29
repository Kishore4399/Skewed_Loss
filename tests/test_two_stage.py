import numpy as np

from src.metrics import adc, mae
from src.two_stage import apply_two_stage, fit_two_stage


def test_two_stage_removes_linear_bias():
    rng = np.random.default_rng(0)
    age_val, age_test = rng.uniform(1, 100, 2000), rng.uniform(1, 100, 2000)
    shrink = lambda a: 0.7 * a + 0.3 * 40 + rng.normal(0, 5, a.shape)  # regression toward mean 40
    pred_val, pred_test = shrink(age_val), shrink(age_test)
    assert adc(pred_test, age_test) < -0.5
    b1, b0 = fit_two_stage(pred_val, age_val)
    corrected = apply_two_stage(pred_test, b1, b0)
    assert abs(adc(corrected, age_test)) < 0.05
    assert b1 == np.float64(b1) and abs(b1 - 0.7) < 0.02


def test_metrics_basic():
    assert mae([1, 2, 3], [1, 2, 5]) == 2 / 3
    assert adc([5, 5, 5], [1, 2, 3]) == -1.0
