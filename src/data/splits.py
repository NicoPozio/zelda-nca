# Split train/val/test a livello di stanza.
# Si fa sulle stanze gia' deduplicate, cosi' una stanza (o una sua simmetria) non
# puo' finire in due insiemi diversi. Il test resta intatto per il numero finale.
#
# Perche' tre insiemi e non due: la val serve a scegliere gli iperparametri durante
# le ablation, il test a misurare la performance finale. Se la val facesse entrambe
# le cose, il risultato finale sarebbe distorto (avresti gia' "sbirciato").
from __future__ import annotations

import numpy as np


def train_val_test_split(rooms, val_fraction, test_fraction, seed):
    """Divide le stanze in (train, val, test) prendendo stanze intere.

    Args:
        rooms: array (M, H, W) di stanze dedotte.
        val_fraction: quota destinata alla validation.
        test_fraction: quota destinata al test.
        seed: seme della permutazione (deterministico).

    Returns:
        (train, val, test).
    """
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("le frazioni devono stare in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val + test deve lasciare spazio al train")

    n = len(rooms)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = round(n * val_fraction)
    n_test = round(n * test_fraction)

    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return rooms[train_idx], rooms[val_idx], rooms[test_idx]