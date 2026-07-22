# Test del campo di distanza ausiliario.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.tiles import CHAR_MAP
from src.models.aux_targets import access_distance_field, aux_targets


def _room():
    return np.array([[CHAR_MAP[c] for c in r] for r in
                     ["WWWWW", "WFFFW", "WFFFW", "WWDWW"]])


def test_access_is_zero():
    f = access_distance_field(_room())
    assert f[3, 2] == 0.0


def test_wall_is_max():
    # un muro non e' raggiungibile camminando: distanza massima
    f = access_distance_field(_room())
    assert f[0, 0] == 1.0


def test_distance_grows_away_from_access():
    f = access_distance_field(_room())
    assert f[2, 2] < f[1, 2]


def test_values_in_unit_range():
    f = access_distance_field(_room())
    assert f.min() >= 0.0 and f.max() <= 1.0


def test_batch_shape():
    t = aux_targets(np.stack([_room(), _room()]))
    assert t.shape == (2, 1, 4, 5)


if __name__ == "__main__":
    for fn in (test_access_is_zero, test_wall_is_max, test_distance_grows_away_from_access,
               test_values_in_unit_range, test_batch_shape):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")