import pytest

torch = pytest.importorskip('torch')
from src.losses import L2SP, SkewedLoss  # noqa: E402

STATS = {'min': 1.0, 'max': 101.0, 'median': 30.0}


def reference_skewed(y_pred, y_true, lam_max):
    """Per-sample loop, written the way the authors' utils/metrics_utils.py computes it."""
    lams = torch.zeros_like(y_true)
    for i in range(len(y_true)):
        y = y_true[i][0]
        if y <= STATS['median']:
            g = (y - STATS['min']) / (STATS['median'] - STATS['min'])
            lams[i][0] = (1 - g) * (-lam_max)
        else:
            g = (y - STATS['median']) / (STATS['max'] - STATS['median'])
            lams[i][0] = g * lam_max
    err = (y_pred - y_true).abs()
    return torch.mean(err * torch.exp(torch.sign(y_true - y_pred) * lams))


@pytest.mark.parametrize('lam', [0.3, 1.0, 2.7])
def test_matches_reference_loss_and_gradient(lam):
    torch.manual_seed(0)
    y = torch.rand(257, 1) * 100 + 1
    p1 = (y + torch.randn(257, 1) * 15).requires_grad_()
    p2 = p1.detach().clone().requires_grad_()
    ref = reference_skewed(p1, y, lam)
    mine = SkewedLoss(STATS, lam)(p2, y)
    ref.backward()
    mine.backward()
    assert torch.allclose(ref, mine, atol=1e-5)
    assert torch.allclose(p1.grad, p2.grad, atol=1e-6)


def test_direction_of_skew():
    loss = SkewedLoss(STATS, lambda_max=1.0)
    young, old = torch.tensor([[1.0]]), torch.tensor([[101.0]])
    # young: over-predicting costs more than under-predicting by the same amount
    assert loss(young + 5, young) > loss(young - 5, young)
    # old: under-predicting costs more
    assert loss(old - 5, old) > loss(old + 5, old)
    # median age: symmetric
    mid = torch.tensor([[30.0]])
    assert torch.isclose(loss(mid + 5, mid), loss(mid - 5, mid))


def test_zero_lambda_is_plain_l1():
    y = torch.rand(64, 1) * 100
    p = y + torch.randn(64, 1)
    assert torch.isclose(SkewedLoss(STATS, 0.0)(p, y), torch.nn.L1Loss()(p, y))


def test_l2sp_zero_at_start_and_pulls_back():
    net = torch.nn.Sequential()
    net.add_module('body', torch.nn.Linear(4, 4))
    net.add_module('fc', torch.nn.Linear(4, 1))
    reg = L2SP(net, alpha=0.1, beta=0.0)
    assert reg().item() == 0.0  # backbone at its starting point, head unpenalized
    with torch.no_grad():
        net.body.weight += 1.0
    penalty = reg()
    assert torch.isclose(penalty, torch.tensor(0.5 * 0.1 * 16))  # 16 weights each moved by 1
    penalty.backward()
    assert torch.all(net.body.weight.grad > 0)  # gradient points back toward the start


def test_l2sp_head_gets_plain_l2():
    net = torch.nn.Sequential()
    net.add_module('body', torch.nn.Linear(2, 2))
    net.add_module('fc', torch.nn.Linear(2, 1))
    with torch.no_grad():
        net.fc.weight.fill_(2.0)
        net.fc.bias.fill_(0.0)
    reg = L2SP(net, alpha=0.0, beta=0.5)
    assert torch.isclose(reg(), torch.tensor(0.5 * 0.5 * 8.0))  # two weights of 2 -> sum of squares 8
