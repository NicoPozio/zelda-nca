# Test del sample pool.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from src.train.pool import SamplePool
from src.models.encoding import to_one_hot, visible_channels
from src.tiles import NUM_TILES


def _rooms(n=30):
    return np.random.default_rng(0).integers(0, NUM_TILES, size=(n, 11, 16))


def _pool(pool_size=64, hidden=12):
    return SamplePool(_rooms(), pool_size=pool_size, hidden_channels=hidden, device="cpu", seed=1)


def test_init_shapes():
    pool = _pool()
    assert pool.state.shape == (64, NUM_TILES + 12, 11, 16)
    assert pool.target.shape == (64, 11, 16)


def test_sample_shapes():
    pool = _pool()
    slots, states, targets = pool.sample(8)
    assert states.shape == (8, NUM_TILES + 12, 11, 16)
    assert targets.shape == (8, 11, 16)


def test_commit_writes_back():
    pool = _pool()
    slots, states, _ = pool.sample(8)
    pool.commit(slots, states + 1.0)
    assert torch.equal(pool.state[slots], states + 1.0)


def test_reseed_resets_state():
    pool = _pool()
    slots, _, _ = pool.sample(8)
    pool.reseed(slots)
    # i nascosti tornano a zero e i visibili coincidono col nuovo target
    assert torch.equal(pool.state[slots][:, NUM_TILES:], torch.zeros(8, 12, 11, 16))
    assert torch.equal(visible_channels(pool.state[slots]), to_one_hot(pool.target[slots]))


def test_checkpoint_roundtrip():
    pool = _pool()
    pool.sample(8)  # muove l'rng
    sd = pool.state_dict()
    other = SamplePool(_rooms(), pool_size=64, hidden_channels=12, device="cpu", seed=999)
    other.load_state_dict(sd)
    assert torch.equal(pool.state, other.state)
    assert torch.equal(pool.target, other.target)


if __name__ == "__main__":
    for fn in (test_init_shapes, test_sample_shapes, test_commit_writes_back,
               test_reseed_resets_state, test_checkpoint_roundtrip):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")