"""Skewed loss (Wang et al., IEEE TMI 2023), vectorized.

    loss(y, y_hat) = L(y, y_hat) * exp(sign(y - y_hat) * lambda(y))

lambda(y) is piecewise linear in the true age y:
    -lambda_max at the youngest training age,
     0          at the median training age,
    +lambda_max at the oldest training age.
For the young, over-predictions (y_hat > y) get weight exp(+|lambda|) and
under-predictions exp(-|lambda|); for the old it is the reverse.
"""
import torch
import torch.nn as nn


def lambda_of_age(y, lambda_max, age_min, age_median, age_max):
    """Piecewise-linear lambda(y), eqs. (15)-(16) of the paper. Works on any tensor shape."""
    low = -(1.0 - (y - age_min) / (age_median - age_min)) * lambda_max
    high = ((y - age_median) / (age_max - age_median)) * lambda_max
    lam = torch.where(y <= age_median, low, high)
    # ages outside the training range (val/test) are clipped to the end values
    return lam.clamp(-abs(lambda_max), abs(lambda_max))


class SkewedLoss(nn.Module):
    def __init__(self, age_stats, lambda_max=1.0, base='L1'):
        super().__init__()
        if base not in ('L1', 'L2'):
            raise ValueError(f'unknown base loss {base}')
        self.age_min = age_stats['min']
        self.age_median = age_stats['median']
        self.age_max = age_stats['max']
        self.lambda_max = float(lambda_max)
        self.base = base

    def set_lambda_max(self, value):
        # the network keeps training; only the loss shape changes
        self.lambda_max = float(value)

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        base = diff.abs() if self.base == 'L1' else diff.pow(2)
        lam = lambda_of_age(y_true, self.lambda_max, self.age_min, self.age_median, self.age_max)
        weight = torch.exp(torch.sign(y_true - y_pred) * lam)  # no gradient flows through sign()
        return (base * weight).mean()


def build_train_loss(args, age_stats):
    if args.mode == 'skewed':
        return SkewedLoss(age_stats, lambda_max=args.lambda_init, base=args.base_loss)
    return nn.L1Loss() if args.base_loss == 'L1' else nn.MSELoss()
