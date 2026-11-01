"""Skewed L1 loss (Wang et al., IEEE TMI 2023), vectorized.

    loss(y, y_hat) = |y - y_hat| * exp(sign(y - y_hat) * lambda(y))

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
    def __init__(self, age_stats, lambda_max=1.0):
        super().__init__()
        self.age_min = age_stats['min']
        self.age_median = age_stats['median']
        self.age_max = age_stats['max']
        self.lambda_max = float(lambda_max)

    def set_lambda_max(self, value):
        # the network keeps training; only the loss shape changes
        self.lambda_max = float(value)

    def forward(self, y_pred, y_true):
        lam = lambda_of_age(y_true, self.lambda_max, self.age_min, self.age_median, self.age_max)
        weight = torch.exp(torch.sign(y_true - y_pred) * lam)  # no gradient flows through sign()
        return ((y_pred - y_true).abs() * weight).mean()


class L2SP:
    """L2-SP regularizer (Li, Grandvalet & Davoine, ICML 2018).

        alpha/2 * ||w_backbone - w_pretrained||^2  +  beta/2 * ||w_head||^2

    The backbone is pulled toward its ImageNet starting point instead of toward zero;
    the new regression head (no pretrained weights) gets plain L2.
    """

    def __init__(self, net, alpha, beta, head_prefix='fc.'):
        self.alpha, self.beta = alpha, beta
        self.backbone, self.head, self.start = [], [], []
        for name, p in net.named_parameters():
            if name.startswith(head_prefix):
                self.head.append(p)
            else:
                self.backbone.append(p)
                self.start.append(p.detach().clone())  # frozen copy of the pretrained weights
        if not self.head:
            raise ValueError(f'no parameters start with {head_prefix!r}; cannot tell head from backbone')

    def __call__(self):
        sp = sum(((p - p0) ** 2).sum() for p, p0 in zip(self.backbone, self.start))
        l2 = sum((p ** 2).sum() for p in self.head)
        return 0.5 * self.alpha * sp + 0.5 * self.beta * l2


def build_train_loss(args, age_stats):
    if args.mode == 'skewed':
        return SkewedLoss(age_stats, lambda_max=args.lambda_init)
    return nn.L1Loss()
