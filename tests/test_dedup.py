# Test sulla deduplicazione, sia esatta che a meno di simmetria.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data.dedup import deduplicate


def _rooms(n, seed=0):
    return np.random.default_rng(seed).integers(0, 10, size=(n, 11, 16), dtype=np.int64)


def test_none_keeps_all():
    assert len(deduplicate(_rooms(10), mode="none")) == 10


def test_exact():
    base = _rooms(10)
    both = np.concatenate([base, base])
    assert len(deduplicate(both, mode="exact")) == 10


def test_symmetry_collapses_mirror():
    base = _rooms(10)
    with_mirror = np.concatenate([base, base[:, :, ::-1]], axis=0)
    # gli esatti restano 20, ma a meno di simmetria tornano 10
    assert len(deduplicate(with_mirror, mode="exact")) == 20
    assert len(deduplicate(with_mirror, mode="symmetry")) == 10


def test_index_consistency():
    r = _rooms(20)
    uniq, idx = deduplicate(r, mode="symmetry", return_index=True)
    assert np.array_equal(uniq, r[idx])


def test_deterministic():
    r = _rooms(30, seed=1)
    assert np.array_equal(deduplicate(r, mode="symmetry"), deduplicate(r, mode="symmetry"))


if __name__ == "__main__":
    for fn in (test_none_keeps_all, test_exact, test_symmetry_collapses_mirror,
               test_index_consistency, test_deterministic):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")