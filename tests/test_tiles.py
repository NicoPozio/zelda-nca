# Test sull'alfabeto dei tile e sulla calpestabilita' che dipende dal contesto.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # per il lancio diretto

import numpy as np
from src.tiles import CHAR_MAP, IDX_TO_CHAR, NUM_TILES, walkable_mask, door_mask


def _parse(rows):
    return np.array([[CHAR_MAP[c] for c in r] for r in rows])


def test_alphabet():
    assert NUM_TILES == 10
    assert len(set(CHAR_MAP.values())) == 10
    assert all(IDX_TO_CHAR[i] in CHAR_MAP for i in range(10))


def test_walkable_base():
    # la stanza ha del pavimento, quindi qui '-' e' un precipizio e non si cammina
    room = _parse(["FDMSOBWPI-"])
    got = list(walkable_mask(room)[0])
    assert got == [True, True, True, True, True, False, False, False, False, False]


def test_element_floor_flag():
    room = _parse(["FO"])
    assert walkable_mask(room, passable_element_floor=True)[0, 1]
    assert not walkable_mask(room, passable_element_floor=False)[0, 1]


def test_void_context():
    v = CHAR_MAP['-']
    # senza pavimento e' una stanza dell'anziano: il nero si cammina
    old_man = _parse(["WWWWW", "W---W", "W-D-W", "WWWWW"])
    assert walkable_mask(old_man)[old_man == v].all()
    # con pavimento i '-' sono buchi, quindi ostacoli
    pit = _parse(["WWWWW", "WFF-W", "WF-DW", "WWWWW"])
    assert (~walkable_mask(pit)[pit == v]).all()


def test_doors():
    room = _parse(["WDW", "WFW"])
    dm = door_mask(room)
    assert dm[0, 1] and dm.sum() == 1


if __name__ == "__main__":
    for fn in (test_alphabet, test_walkable_base, test_element_floor_flag,
               test_void_context, test_doors):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")