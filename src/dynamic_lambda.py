"""Dynamic lambda training strategy (Algorithm 1 of Wang et al., 2023).

Stage 1 (epoch < stage2_start): train with a fixed lambda_max.
Stage 2 (stage2_start <= epoch < stage3_start): every `interval` epochs, measure
    validation ADC; if ADC < -theta multiply lambda_max by eta, if ADC > +theta
    divide by eta. Every (lambda_max, ADC) pair is recorded.
Stage 3 (epoch >= stage3_start): every `interval` epochs, fit ADC ~ a*lambda + b on
    the recorded pairs and set lambda_max to the zero crossing -b/a.

Two stabilizers from the authors' released code are available (and on by default):
    smooth=True   -> the ADC used for an update is the median of a 3-point moving
                     average over the last `interval` epochs, not a single epoch's value
    history=10    -> stage 3 fits only the 10 most recent pairs (0 = all pairs, as in the paper)
One stabilizer is added here, not in the paper or the authors' code:
    max_step=1.5  -> a stage-3 update can change lambda_max by at most this factor
                     (0 = unlimited). Late in training the recorded lambdas sit close
                     together, the fitted slope gets noisy, and an unlimited zero
                     crossing can jump far away; this keeps the update local.
lambda_max is always kept within [0, cap].
"""
import numpy as np


class DynamicLambda:
    def __init__(self, init=1.0, stage2_start=30, stage3_start=60, interval=3, eta=1.5,
                 theta=0.1, smooth=True, history=10, cap=20.0, max_step=1.5):
        if not stage2_start < stage3_start:
            raise ValueError('stage2_start must be before stage3_start')
        self.value = float(init)
        self.stage2_start = stage2_start
        self.stage3_start = stage3_start
        self.interval = interval
        self.eta = eta
        self.theta = theta
        self.smooth = smooth
        self.history = history
        self.cap = cap
        self.max_step = max_step
        self.pairs = []  # (lambda_max, ADC measured while training with it)

    def stage(self, epoch):
        if epoch < self.stage2_start:
            return 1
        return 2 if epoch < self.stage3_start else 3

    def is_update_epoch(self, epoch):
        return epoch >= self.stage2_start and (epoch - self.stage2_start) % self.interval == 0

    def _current_adc(self, val_adcs):
        recent = np.asarray(val_adcs[-self.interval:], dtype=float)
        if not self.smooth or len(recent) == 0:
            return float(val_adcs[-1])
        if len(recent) >= 3:
            recent = np.convolve(recent, np.ones(3) / 3, mode='valid')
        return float(np.median(recent))

    def _fit_zero_crossing(self):
        pairs = self.pairs[-self.history:] if self.history else self.pairs
        lam = np.array([p[0] for p in pairs])
        corr = np.array([p[1] for p in pairs])
        if len(np.unique(lam)) < 2:
            return float(lam.mean())
        slope, intercept = np.polyfit(lam, corr, 1)
        if abs(slope) < 1e-2:
            # flat fit: the zero crossing would be far away and meaningless
            return float(lam.mean())
        return float(-intercept / slope)

    def step(self, epoch, val_adcs):
        """Call once per epoch after validation. Returns the new lambda_max if it changed, else None."""
        if not self.is_update_epoch(epoch):
            return None
        adc = self._current_adc(val_adcs)
        self.pairs.append((self.value, adc))
        old = self.value
        if self.stage(epoch) == 2:
            if adc < -self.theta:
                self.value *= self.eta
            elif adc > self.theta:
                self.value /= self.eta
        else:
            target = self._fit_zero_crossing()
            if self.max_step and old > 0:
                target = float(np.clip(target, old / self.max_step, old * self.max_step))
            self.value = target
        self.value = float(np.clip(self.value, 0.0, self.cap))
        return self.value if self.value != old else None
