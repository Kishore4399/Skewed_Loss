import numpy as np
import pytest

from src.dynamic_lambda import DynamicLambda


def test_stage1_never_updates():
    lam = DynamicLambda(init=1.0, stage2_start=10, stage3_start=20, interval=2)
    adcs = []
    for epoch in range(1, 10):
        adcs.append(-0.8)
        assert lam.step(epoch, adcs) is None
    assert lam.value == 1.0 and lam.pairs == []


def test_stage2_multiplies_and_divides():
    lam = DynamicLambda(init=1.0, stage2_start=1, stage3_start=100, interval=1, eta=1.5, theta=0.1, smooth=False)
    assert lam.step(1, [-0.5]) == pytest.approx(1.5)     # under-corrected -> stronger skew
    assert lam.step(2, [-0.5, 0.4]) == pytest.approx(1.0)  # over-corrected -> weaker skew
    assert lam.step(3, [-0.5, 0.4, 0.05]) is None          # inside the band -> unchanged


def test_stage3_finds_zero_crossing():
    lam = DynamicLambda(init=1.0, stage2_start=1, stage3_start=3, interval=1, smooth=False, history=0)
    true_adc = lambda lm: 0.5 * lm - 0.8  # zero at lambda = 1.6
    adcs = []
    for epoch in range(1, 4):
        adcs.append(true_adc(lam.value))
        lam.step(epoch, adcs)
    assert lam.value == pytest.approx(1.6, abs=1e-6)


def test_closed_loop_converges_with_noise():
    """Toy model: ADC responds linearly to lambda_max, with noise and a drifting offset."""
    rng = np.random.default_rng(0)
    lam = DynamicLambda(init=1.0, stage2_start=30, stage3_start=60, interval=3)
    adcs = []
    for epoch in range(1, 91):
        drift = -0.9 + 0.003 * epoch            # the "network" changes as it trains
        adcs.append(0.45 * lam.value + drift + rng.normal(0, 0.05))
        lam.step(epoch, adcs)
    # the ADC actually reached over the last 10 trained epochs
    assert abs(np.mean(adcs[-10:])) < 0.1


def test_stage3_step_limit():
    lam = DynamicLambda(init=1.0, stage2_start=1, stage3_start=2, interval=1, smooth=False, max_step=1.5)
    lam.step(1, [-0.9])           # stage 2: 1.0 -> 1.5
    lam.step(2, [-0.9, -0.89])    # nearly flat fit -> far zero crossing, limited to 1.5 * 1.5
    assert lam.value == pytest.approx(2.25)


def test_lambda_is_clipped():
    lam = DynamicLambda(init=1.0, stage2_start=1, stage3_start=2, interval=1, smooth=False, cap=5.0,
                        max_step=0)
    lam.step(1, [-0.9])
    lam.step(2, [-0.9, -0.89])  # nearly flat in lambda, far below zero -> huge zero crossing
    assert 0.0 <= lam.value <= 5.0
