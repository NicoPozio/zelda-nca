# Test dei danni mirati B1-B4.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from src.tiles import CHAR_MAP
from src.damage.targeted import (select_doors, select_wall_segment, select_access_isolation,
                                 select_articulation, articulation_points, kill_cells)
from src.metrics.connectivity import preserves_topology
from src.models.encoding import to_nca_state


def _room():
    return np.array([[CHAR_MAP[c] for c in r] for r in
                     ["WWWWWWW", "WFFFFFW", "WFFFFFW", "WFFFFFW", "WWWDWWW"]])


def _narrow():
    # stanza con un passaggio stretto: ha punti di articolazione
    return np.array([[CHAR_MAP[c] for c in r] for r in
                     ["WWWWWWW", "WFFWFFW", "WFFFFFW", "WWWFWWW", "WWWDWWW"]])


def test_b1_hits_door():
    room = _room()
    m = select_doors(room, np.random.default_rng(0), n_doors=1)
    assert m.sum() == 1
    assert room[m][0] == CHAR_MAP['D']


def test_b2_hits_only_walls():
    room = _room()
    m = select_wall_segment(room, np.random.default_rng(3), length=4)
    assert m.sum() > 0
    assert np.all(room[m] == CHAR_MAP['W'])


def test_b3_isolates_without_touching_door():
    room = _room()
    m = select_access_isolation(room, np.random.default_rng(0))
    assert m.sum() > 0
    assert not m[4, 3]                       # la porta stessa non viene toccata
    # e la topologia risulta effettivamente rotta
    broken = room.copy()
    broken[m] = CHAR_MAP['W']
    assert preserves_topology(room, broken) is False


def test_b4_finds_articulation_points():
    room = _narrow()
    pts = articulation_points(room)
    assert pts.sum() > 0
    m = select_articulation(room, np.random.default_rng(0), k=1)
    assert m.sum() == 1
    assert pts[m][0]                         # il punto scelto e' di articolazione


def test_kill_cells_zeroes_all_channels():
    room = _room()
    state = to_nca_state(torch.as_tensor(room).unsqueeze(0), hidden_channels=12)
    m = select_doors(room, np.random.default_rng(0), 1)
    dmg, _ = kill_cells(state, m)
    mm = torch.as_tensor(m)
    assert torch.all(dmg[0][:, mm] == 0)                 # celle colpite azzerate
    assert torch.equal(dmg[0][:, ~mm], state[0][:, ~mm])  # il resto invariato


if __name__ == "__main__":
    for fn in (test_b1_hits_door, test_b2_hits_only_walls,
               test_b3_isolates_without_touching_door, test_b4_finds_articulation_points,
               test_kill_cells_zeroes_all_channels):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")