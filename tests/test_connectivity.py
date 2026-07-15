# Test della metrica di connettivita', su stanze costruite a mano di cui si
# conosce la risposta a priori.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.tiles import CHAR_MAP
from src.metrics.connectivity import (preserves_topology, rsr, tile_accuracy,
                                      walkable_components)


def _parse(rows):
    return np.array([[CHAR_MAP[c] for c in r] for r in rows])


def _room():
    # stanza semplice: pavimento con una porta in basso
    return _parse(["WWWWW", "WFFFW", "WFFFW", "WWDWW"])


def _spiral():
    # spirale con scala isolata al centro e porta in basso: la scala non e'
    # raggiungibile nemmeno da intatta, nel gioco ci si arriva coi blocchi spingibili
    return _parse(["WWWWWWW", "WFFFFFW", "WFWWWFW", "WFWSWFW",
                   "WFWWWFW", "WFFFFFW", "WWWDWWW"])


def test_identity_preserves():
    r = _room()
    assert preserves_topology(r, r.copy()) is True


def test_walled_door_fails():
    # la porta sparisce diventando muro: la stanza perde l'uscita
    r = _room()
    broken = r.copy()
    broken[3, 2] = CHAR_MAP['W']
    assert preserves_topology(r, broken) is False


def test_door_turned_to_floor_fails():
    # la porta diventa pavimento: l'area resta connessa ma la stanza non ha uscita
    r = _room()
    broken = r.copy()
    broken[3, 2] = CHAR_MAP['F']
    assert preserves_topology(r, broken) is False


def test_blocked_door_fails():
    # la porta c'e' ma un muro davanti la rende irraggiungibile
    r = _room()
    broken = r.copy()
    broken[2, 2] = CHAR_MAP['W']
    assert preserves_topology(r, broken) is False


def test_spurious_door_fails():
    # una porta in piu' cambia i collegamenti della stanza col resto del dungeon
    r = _room()
    other = r.copy()
    other[0, 2] = CHAR_MAP['D']
    assert preserves_topology(r, other) is False


def test_cosmetic_change_preserves():
    # un mostro sul pavimento resta calpestabile: la topologia non cambia
    r = _room()
    other = r.copy()
    other[1, 1] = CHAR_MAP['M']
    assert preserves_topology(r, other) is True


def test_pristine_with_isolated_access_is_not_penalized():
    s = _spiral()
    assert walkable_components(s)[1] == 2
    assert preserves_topology(s, s.copy()) is True


def test_wrongly_connecting_isolated_access_fails():
    # aprire il muro connette la scala che era isolata: topologia diversa
    s = _spiral()
    opened = s.copy()
    opened[3, 2] = CHAR_MAP['F']
    assert preserves_topology(s, opened) is False


def test_rsr_and_accuracy():
    r = _room()
    broken = r.copy()
    broken[2, 2] = CHAR_MAP['W']
    assert rsr(np.stack([r, r]), np.stack([r.copy(), broken])) == 0.5
    assert tile_accuracy([r], [r.copy()]) == 1.0


def test_high_accuracy_can_hide_broken_topology():
    # il punto della Sub-RQ 1: un solo tile sbagliato, accuratezza altissima,
    # ma la stanza e' ingiocabile
    r = _room()
    broken = r.copy()
    broken[2, 2] = CHAR_MAP['W']
    assert tile_accuracy([r], [broken]) > 0.9
    assert preserves_topology(r, broken) is False


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        preserves_topology(_room(), _spiral())


if __name__ == "__main__":
    for fn in (test_identity_preserves, test_walled_door_fails, test_door_turned_to_floor_fails,
               test_blocked_door_fails, test_spurious_door_fails, test_cosmetic_change_preserves,
               test_pristine_with_isolated_access_is_not_penalized,
               test_wrongly_connecting_isolated_access_fails, test_rsr_and_accuracy,
               test_high_accuracy_can_hide_broken_topology, test_shape_mismatch_raises):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")