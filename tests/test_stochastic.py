# Test dei danni stocastici A1 (erasure) e A2 (tile-flip).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from src.damage.stochastic import erasure, tile_flip
from src.models.encoding import to_nca_state, visible_channels
from src.tiles import NUM_TILES


def _state(n=4, hidden=12):
    rooms = torch.randint(0, NUM_TILES, (n, 11, 16))
    return to_nca_state(rooms, hidden_channels=hidden)


def test_erasure_kills_cells():
    state = _state()
    d, m = erasure(state, np.random.default_rng(0), fraction=0.25)
    # le celle colpite sono a zero su tutti i canali
    assert torch.all(d[m.expand_as(d)] == 0)
    # fuori dalla maschera nulla cambia
    keep = ~m.expand_as(d)
    assert torch.equal(d[keep], state[keep])


def test_tile_flip_keeps_one_hot():
    state = _state()
    d, m = tile_flip(state, np.random.default_rng(1), fraction=0.2)
    cells = m[:, 0]
    # esattamente un tile visibile acceso sulle celle colpite
    assert torch.all(visible_channels(d).sum(1)[cells] == 1)
    # nascosti azzerati sulle celle colpite
    assert torch.all(d[:, NUM_TILES:].permute(0, 2, 3, 1)[cells] == 0)


def test_tile_flip_changes_tile():
    state = _state()
    d, m = tile_flip(state, np.random.default_rng(2), fraction=0.3)
    cells = m[:, 0]
    changed = visible_channels(d).argmax(1) != visible_channels(state).argmax(1)
    assert torch.all(changed[cells])


def test_reproducible():
    state = _state()
    a, _ = erasure(state, np.random.default_rng(7), 0.3)
    b, _ = erasure(state, np.random.default_rng(7), 0.3)
    assert torch.equal(a, b)


def test_bad_fraction():
    state = _state()
    for f in (0.0, 1.5):
        try:
            erasure(state, np.random.default_rng(0), f)
            assert False, "doveva sollevare ValueError"
        except ValueError:
            pass


if __name__ == "__main__":
    for fn in (test_erasure_kills_cells, test_tile_flip_keeps_one_hot,
               test_tile_flip_changes_tile, test_reproducible, test_bad_fraction):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")