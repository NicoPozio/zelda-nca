# Test dello split train/val/test a livello di stanza.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.data.splits import train_val_test_split
from src.data.dedup import deduplicate


def _rooms(n, seed=0):
    return np.random.default_rng(seed).integers(0, 10, size=(n, 11, 16), dtype=np.int64)


def _symmetries(room):
    # riscritte qui apposta invece di importarle da symmetry.py: cosi' il test
    # verifica il leakage in modo indipendente dal modulo che sta controllando
    return [room, room[:, ::-1], room[::-1, :], room[::-1, ::-1]]


def test_sizes():
    tr, va, te = train_val_test_split(_rooms(100), val_fraction=0.15, test_fraction=0.15, seed=0)
    assert len(tr) == 70 and len(va) == 15 and len(te) == 15


def test_partition_is_complete_and_disjoint():
    r = _rooms(100)
    tr, va, te = train_val_test_split(r, 0.15, 0.15, seed=0)
    keys = lambda arr: {x.tobytes() for x in arr}
    ktr, kva, kte = keys(tr), keys(va), keys(te)
    # nessuna stanza in due insiemi, e insieme coprono tutto
    assert len(ktr & kva) == 0 and len(ktr & kte) == 0 and len(kva & kte) == 0
    assert len(ktr | kva | kte) == 100


def test_no_symmetry_leak():
    # su stanze gia' dedotte a simmetria, nessuna simmetria di val o test deve
    # ricomparire nel train: e' la garanzia che regge tutta la pipeline dati
    r = deduplicate(_rooms(120), mode="symmetry")
    tr, va, te = train_val_test_split(r, 0.2, 0.2, seed=7)
    tr_keys = {x.tobytes() for x in tr}
    leaks = 0
    for arr in (va, te):
        leaks += sum(any(s.tobytes() in tr_keys for s in _symmetries(x)) for x in arr)
    assert leaks == 0


def test_deterministic():
    r = _rooms(50)
    a = train_val_test_split(r, 0.2, 0.2, seed=1)[2]
    b = train_val_test_split(r, 0.2, 0.2, seed=1)[2]
    assert np.array_equal(a, b)


def test_bad_fractions():
    with pytest.raises(ValueError):
        train_val_test_split(_rooms(10), 0.6, 0.6, seed=0)


if __name__ == "__main__":
    for fn in (test_sizes, test_partition_is_complete_and_disjoint,
               test_no_symmetry_leak, test_deterministic, test_bad_fractions):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")