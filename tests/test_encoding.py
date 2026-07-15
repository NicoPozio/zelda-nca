# Test sulla costruzione dello stato NCA (one-hot + canali nascosti).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.encoding import to_one_hot, to_nca_state, visible_channels, decode
from src.tiles import CHAR_MAP, NUM_TILES


def _rooms(n=4):
    return torch.randint(0, NUM_TILES, (n, 11, 16))


def test_one_hot_shape_and_exclusive():
    oh = to_one_hot(_rooms())
    assert oh.shape == (4, NUM_TILES, 11, 16)
    assert torch.equal(oh.sum(1), torch.ones(4, 11, 16))


def test_state_shape_and_hidden_zero():
    state = to_nca_state(_rooms(), hidden_channels=12)
    assert state.shape == (4, NUM_TILES + 12, 11, 16)
    assert torch.equal(state[:, NUM_TILES:], torch.zeros(4, 12, 11, 16))


def test_visible_matches_one_hot():
    rooms = _rooms()
    state = to_nca_state(rooms, hidden_channels=8)
    assert torch.equal(visible_channels(state), to_one_hot(rooms))


def test_roundtrip():
    rooms = _rooms()
    assert torch.equal(to_one_hot(rooms).argmax(1), rooms)


def test_decode_dead_cell_is_void():
    # una cella con tutti i canali a zero non e' un tile: va decodificata come void
    rooms = _rooms(1)
    state = to_nca_state(rooms, hidden_channels=12)
    assert torch.equal(decode(state), rooms)
    state[:, :, 3, 5] = 0.0
    assert decode(state)[0, 3, 5] == CHAR_MAP['-']


if __name__ == "__main__":
    for fn in (test_one_hot_shape_and_exclusive, test_state_shape_and_hidden_zero,
               test_visible_matches_one_hot, test_roundtrip, test_decode_dead_cell_is_void):
        fn()
        print(f"ok  {fn.__name__}")
    print("tutti i test passati")