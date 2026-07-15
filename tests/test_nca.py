# Test della regola di update dell'NCA.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from src.models.nca import NCA
from src.models.encoding import to_nca_state
from src.tiles import NUM_TILES


def _make(hidden=12, **kw):
    rooms = torch.randint(0, NUM_TILES, (2, 11, 16))
    return NCA(hidden_channels=hidden, **kw), to_nca_state(rooms, hidden_channels=hidden)


def test_shape_preserved():
    nca, state = _make()
    assert nca(state).shape == state.shape


def test_zero_init_identity():
    # con w2 a zero il primo passo non cambia lo stato
    nca, state = _make()
    assert torch.allclose(nca(state), state)


def test_no_update_when_prob_zero():
    nca, state = _make()
    nn.init.normal_(nca.w2.weight)
    nca.update_prob = 0.0
    assert torch.allclose(nca(state), state)


def test_gradient_flows():
    nca, state = _make()
    s = state.clone().requires_grad_(True)
    for _ in range(8):
        s = nca(s)
    s.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in nca.parameters())


def test_sobel_zero_on_constant():
    # col padding replicato una regione costante non deve produrre gradienti falsi
    nca, _ = _make()
    const = torch.ones(1, nca.num_channels, 8, 8)
    per = nca.perceive(const).reshape(1, nca.num_channels, 3, 8, 8)
    assert per[:, :, 1:3].abs().sum() < 1e-4


def test_locality():
    # dopo un passo, perturbare una cella non deve toccare celle lontane
    nca, state = _make()
    nca.update_prob = 1.0
    nn.init.normal_(nca.w2.weight, std=0.1)
    s2 = state.clone()
    s2[0, :, 5, 7] += 5.0
    diff = (nca(state) - nca(s2)).abs().sum(1)[0]
    outside = (diff > 1e-6).clone()
    outside[4:7, 6:9] = False
    assert int(outside.sum()) == 0


def test_laplacian_adds_a_filter():
    nca, _ = _make(use_laplacian=True)
    const = torch.ones(1, nca.num_channels, 8, 8)
    assert nca.perceive(const).shape[1] // nca.num_channels == 4


if __name__ == "__main__":
    for fn in (test_shape_preserved, test_zero_init_identity, test_no_update_when_prob_zero,
               test_gradient_flows, test_sobel_zero_on_constant, test_locality,
               test_laplacian_adds_a_filter):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")