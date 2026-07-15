# Test dell'augmentation a 4 simmetrie.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data.augment import augment
from src.data.dedup import deduplicate


def _rooms(n, seed=0):
    return np.random.default_rng(seed).integers(0, 10, size=(n, 11, 16), dtype=np.int64)


def test_multiplies_by_four():
    assert len(augment(_rooms(10))) == 40


def test_keeps_originals():
    r = _rooms(5)
    aug_keys = {x.tobytes() for x in augment(r)}
    assert all(x.tobytes() in aug_keys for x in r)


def test_dedup_roundtrip():
    # augment aggiunge solo simmetrie, quindi deduplicando a simmetria si torna
    # esattamente alle stanze di partenza: prova che i due usano lo stesso gruppo
    r = deduplicate(_rooms(30), mode="symmetry")
    assert len(deduplicate(augment(r), mode="symmetry")) == len(r)


if __name__ == "__main__":
    for fn in (test_multiplies_by_four, test_keeps_originals, test_dedup_roundtrip):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")