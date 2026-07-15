# Test del loop di training, poche iterazioni su CPU.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from src.models.nca import NCA
from src.train.pool import SamplePool
from src.train.trainer import Trainer


def _rooms(n=40):
    return np.random.default_rng(0).integers(0, 10, size=(n, 11, 16))


def _trainer(seed=0):
    nca = NCA(hidden_channels=12)
    pool = SamplePool(_rooms(), pool_size=32, hidden_channels=12, device="cpu", seed=seed)
    return Trainer(nca, pool, lr=1e-3, grad_clip=1.0, bptt_min=6, bptt_max=10,
                   batch_size=8, damage_prob=0.5, damage_fractions=[0.2, 0.4],
                   device="cpu", seed=seed)


def test_step_runs_and_finite():
    # la loss deve restare finita: se esplode, le difese anti-BPTT non funzionano
    tr = _trainer()
    losses = [tr.train_step() for _ in range(10)]
    assert all(np.isfinite(losses))
    assert tr.step == 10


def test_weights_change():
    tr = _trainer()
    before = [p.clone() for p in tr.nca.parameters()]
    for _ in range(5):
        tr.train_step()
    assert any(not torch.equal(a, b) for a, b in zip(before, tr.nca.parameters()))


def test_checkpoint_roundtrip():
    # serve a riprendere le sessioni Kaggle interrotte senza perdere lavoro
    tr = _trainer(seed=0)
    for _ in range(5):
        tr.train_step()
    sd = tr.state_dict()
    other = _trainer(seed=99)
    other.load_state_dict(sd)
    assert other.step == tr.step
    for a, b in zip(tr.nca.parameters(), other.nca.parameters()):
        assert torch.equal(a, b)


if __name__ == "__main__":
    for fn in (test_step_runs_and_finite, test_weights_change, test_checkpoint_roundtrip):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")