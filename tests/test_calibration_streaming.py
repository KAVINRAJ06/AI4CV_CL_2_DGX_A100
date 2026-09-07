import pytest
import torch
from torch import nn
from tqsi.train import calibrate_binary_threshold


class Model(nn.Module):
    def forward(self, x):
        return x


def test_streaming_matches_brute_force_with_ties_and_ignored_pixels():
    thresholds = torch.unique(torch.cat((torch.logit(torch.linspace(0, 1, 17)).nan_to_num(), torch.tensor([0.]))))
    x = torch.cat((torch.linspace(-5, 5, 51), thresholds[1:-1])).reshape(1, 1, 1, -1)
    y = (torch.arange(x.numel()) % 3 == 0).long().reshape(1, 1, -1)
    y[..., 3] = -100
    valid = y.flatten() != -100
    logits, truth = x.flatten()[valid], y.flatten()[valid].bool()
    scores = torch.tensor([2*((logits > t) & truth).sum().double()/((logits > t).sum()+truth.sum()) for t in thresholds])
    threshold, dice = calibrate_binary_threshold(Model(), [(x, y)], 'cpu', quantiles=17)
    assert threshold == float(thresholds[scores.argmax()])
    assert dice == pytest.approx(float(scores.max()))
    pieces = [(a, b) for a, b in zip(x.split(7, -1), y.split(7, -1))]
    assert calibrate_binary_threshold(Model(), pieces, 'cpu', quantiles=17) == (threshold, dice)


def test_large_stream_never_calls_quantile(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unbounded quantile must not be used')
    monkeypatch.setattr(torch, 'quantile', forbidden)
    x = torch.ones(1, 1, 512, 512)
    y = torch.ones(1, 512, 512, dtype=torch.long)
    # Over 2**24 pixels, with only one batch retained by the generator.
    result = calibrate_binary_threshold(Model(), ((x, y) for _ in range(65)), 'cpu')
    assert result[1] == 1.


@pytest.mark.parametrize('kind', ['empty', 'ignored', 'nonfinite'])
def test_invalid_calibration_input(kind):
    x = torch.zeros(1, 1, 2, 2)
    y = torch.zeros(1, 2, 2, dtype=torch.long)
    if kind == 'ignored':
        y.fill_(-100)
    if kind == 'nonfinite':
        x.fill_(float('nan'))
    with pytest.raises(ValueError):
        calibrate_binary_threshold(Model(), [] if kind == 'empty' else [(x, y)], 'cpu')
