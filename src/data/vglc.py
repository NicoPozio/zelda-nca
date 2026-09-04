"""
Parsing ed estrazione delle stanze dal corpus VGLC
"""
from __future__ import annotations

import glob
import os

import numpy as np

from src.tiles import CHAR_MAP, RAW_COLS, RAW_ROWS

VOID = CHAR_MAP['-']


def parse_dungeon(path: str) -> np.ndarray:
    """Legge un file e restituisce un array"""
    lines = [ln.rstrip('\n') for ln in open(path, encoding='utf-8') if ln.strip('\n') != '']
    if not lines:
        raise ValueError(f"File dungeon vuoto: {path}")
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width, '-') for ln in lines]      
    try:
        grid = np.array([[CHAR_MAP[c] for c in row] for row in lines], dtype=np.int64)
    except KeyError as e:                               
        raise ValueError(f"Carattere fuori alfabeto {e!s} in {os.path.basename(path)}") from e
    h, w = grid.shape
    if h % RAW_ROWS or w % RAW_COLS:
        raise ValueError(
            f"{os.path.basename(path)}: dimensione {h}x{w} non multipla di stanza "
            f"{RAW_ROWS}x{RAW_COLS}"
        )
    return grid


def _is_real_room(room: np.ndarray) -> bool:
    """Verifica che sia effettivamente una stanza"""
    return not np.all(room == VOID)


def slice_dungeon(grid: np.ndarray, to_visual: bool = True) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """
    Separa il dungeon in array multipli ognuno rappresentante una vera stanza
    """
    gr, gc = grid.shape[0] // RAW_ROWS, grid.shape[1] // RAW_COLS
    out = []
    for i in range(gr):
        for j in range(gc):
            room = grid[i * RAW_ROWS:(i + 1) * RAW_ROWS, j * RAW_COLS:(j + 1) * RAW_COLS]
            if not _is_real_room(room):
                continue
            if to_visual:
                room = room.T                           
            out.append((room.copy(), (i, j)))
    return out


def extract_rooms(raw_dir: str, to_visual: bool = True) -> tuple[np.ndarray, list[str]]:
    """Estrae tutte le stanze dai file .txt

    Args:
        raw_dir: cartella con i file
        to_visual: applica il transpose agli array delle stanze

    Returns:
        (rooms, sources): array (N, H, W) e lista parallela di tag "file:i,j"
    """
    files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, '*.txt'))
        if os.path.basename(f).lower() != 'readme.txt'
    )
    if not files:
        raise FileNotFoundError(f"Nessun file dungeon .txt in {raw_dir!r}")

    rooms, sources = [], []
    for path in files:
        grid = parse_dungeon(path)
        for room, (i, j) in slice_dungeon(grid, to_visual=to_visual):
            rooms.append(room)
            sources.append(f"{os.path.basename(path)}:{i},{j}")

    return np.stack(rooms), sources