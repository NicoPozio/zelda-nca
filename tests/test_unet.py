# Test dell'U-Net: interfaccia compatibile con l'NCA e campo recettivo globale.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from src.models.unet import UNet
from src.models.encoding import to_nca_state
from src.tiles import NUM_TILES


def _make(hidden=12, **kw):
    rooms = torch.randint(0, NUM_TILES, (2, 11, 16))
    return UNet(hidden_channels=hidden, **kw), to_nca_state(rooms, hidden_channels=hidden)


def test_shape_preserved():
    net, state = _make()
    assert net(state).shape == state.shape


def test_zero_init_identity():
    # come l'NCA: head a zero -> il primo passo non cambia lo stato
    net, state = _make()
    assert torch.allclose(net(state), state, atol=1e-6)


def test_receptive_field_is_global():
    # la proprieta' che distingue l'U-Net dall'NCA: perturbare una cella
    # influenza anche l'angolo opposto della griglia
    net, _ = _make()
    nn.init.normal_(net.head.weight, std=0.1)
    s1 = to_nca_state(torch.randint(0, NUM_TILES, (1, 11, 16)), 12)
    s2 = s1.clone()
    s2[0, :, 0, 0] += 5.0
    diff = (net(s1) - net(s2)).abs().sum(1)[0]
    assert diff[-1, -1].item() > 1e-6


def test_odd_dimensions_handled():
    # 11x16 non e' divisibile per 4: il padding-e-ritaglio deve reggere
    net, state = _make()
    out = net(state)
    assert out.shape[-2:] == (11, 16)


if __name__ == "__main__":
    for fn in (test_shape_preserved, test_zero_init_identity,
               test_receptive_field_is_global, test_odd_dimensions_handled):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")