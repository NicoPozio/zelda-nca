# Test su parsing, slicing e transpose, usando un dungeon costruito a mano.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.tiles import RAW_ROWS, RAW_COLS
from src.data.vglc import parse_dungeon, slice_dungeon, extract_rooms


def _valid_room():
    # stanza 16x11 con muri, pavimento e una porta in basso
    rows = ["WWWWWWWWWWW", "WWWWWWWWWWW"]
    rows += ["WWFFFFFFFWW"] * 12
    rows += ["WWWWDDDWWWW", "WWWWWWWWWWW"]
    return rows


def _write_dungeon(tmp_path):
    # dungeon di 1x2 stanze: una vera a sinistra, una vuota a destra
    room = _valid_room()
    empty = "-" * RAW_COLS
    lines = [room[r] + empty for r in range(RAW_ROWS)]
    p = tmp_path / "tloz9_1.txt"
    p.write_text("\n".join(lines))
    return str(p)


def test_parse_dimensions(tmp_path):
    grid = parse_dungeon(_write_dungeon(tmp_path))
    assert grid.shape == (RAW_ROWS, 2 * RAW_COLS)


def test_void_filter(tmp_path):
    grid = parse_dungeon(_write_dungeon(tmp_path))
    rooms = slice_dungeon(grid, to_visual=False)
    # la stanza vuota viene scartata, resta solo quella vera
    assert len(rooms) == 1
    assert rooms[0][0].shape == (RAW_ROWS, RAW_COLS)


def test_transpose_preserves(tmp_path):
    grid = parse_dungeon(_write_dungeon(tmp_path))
    native = slice_dungeon(grid, to_visual=False)[0][0]
    visual = slice_dungeon(grid, to_visual=True)[0][0]
    assert visual.shape == (RAW_COLS, RAW_ROWS)
    # il transpose cambia solo l'orientamento, non quali tile sono presenti
    assert np.array_equal(np.bincount(native.ravel(), minlength=10),
                          np.bincount(visual.ravel(), minlength=10))
    assert np.array_equal(visual, native.T)


def test_unknown_char(tmp_path):
    p = tmp_path / "tloz1_1.txt"
    p.write_text("\n".join(["Z" * RAW_COLS] * RAW_ROWS))
    with pytest.raises(ValueError):
        parse_dungeon(str(p))


def test_real_data(tmp_path=None):
    raw = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    ok = os.path.isdir(raw) and any(f.endswith(".txt") for f in os.listdir(raw))
    if not ok:
        pytest.skip("nessun dato reale in data/raw/")
    rooms, sources = extract_rooms(raw, to_visual=True)
    assert rooms.shape[1:] == (RAW_COLS, RAW_ROWS)
    assert len(rooms) == len(sources) > 0


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fn in (test_parse_dimensions, test_void_filter,
                   test_transpose_preserves, test_unknown_char):
            fn(tmp)
            print(f"ok  {fn.__name__}")

    raw = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    if os.path.isdir(raw) and any(f.endswith(".txt") for f in os.listdir(raw)):
        test_real_data()
        print("ok  test_real_data")
    else:
        print("skip test_real_data (nessun dato in data/raw)")
    print("tutti i test passati")