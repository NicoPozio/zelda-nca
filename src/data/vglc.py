"""Parsing ed estrazione delle stanze dal corpus VGLC (The Legend of Zelda, NES).

Legge i file dungeon .txt, li affetta in stanze RAW_ROWS x RAW_COLS (16x11) e,
di default, applica il transpose all'orientamento di gioco (11x16). La dedup NON
avviene qui: e' un modulo separato (dedup.py), cosi' si possono ispezionare anche
le stanze grezze coi duplicati.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from src.tiles import CHAR_MAP, RAW_COLS, RAW_ROWS

VOID = CHAR_MAP['-']


def parse_dungeon(path: str) -> np.ndarray:
    """Legge un file dungeon in un array 2D di indici tile (righe ragged paddate a void)."""
    lines = [ln.rstrip('\n') for ln in open(path, encoding='utf-8') if ln.strip('\n') != '']
    if not lines:
        raise ValueError(f"File dungeon vuoto: {path}")
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width, '-') for ln in lines]      # pad righe irregolari con void
    try:
        grid = np.array([[CHAR_MAP[c] for c in row] for row in lines], dtype=np.int64)
    except KeyError as e:                                # carattere fuori alfabeto -> fail loud
        raise ValueError(f"Carattere fuori alfabeto {e!s} in {os.path.basename(path)}") from e
    h, w = grid.shape
    if h % RAW_ROWS or w % RAW_COLS:
        raise ValueError(
            f"{os.path.basename(path)}: dimensione {h}x{w} non multipla di stanza "
            f"{RAW_ROWS}x{RAW_COLS}"
        )
    return grid


def _is_real_room(room: np.ndarray) -> bool:
    """Stanza reale = non interamente void (scarta gli slot di stanza assenti)."""
    return not np.all(room == VOID)


def slice_dungeon(grid: np.ndarray, to_visual: bool = True) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Affetta un dungeon in stanze reali. Con to_visual traspone alla vista di gioco.

    Il transpose preserva la 4-adiacenza: la topologia e' invariata, cambia solo
    l'orientamento (11x16, landscape come il gioco).
    """
    gr, gc = grid.shape[0] // RAW_ROWS, grid.shape[1] // RAW_COLS
    out = []
    for i in range(gr):
        for j in range(gc):
            room = grid[i * RAW_ROWS:(i + 1) * RAW_ROWS, j * RAW_COLS:(j + 1) * RAW_COLS]
            if not _is_real_room(room):
                continue
            if to_visual:
                room = room.T                            # -> orientamento di gioco
            out.append((room.copy(), (i, j)))
    return out


def extract_rooms(raw_dir: str, to_visual: bool = True) -> tuple[np.ndarray, list[str]]:
    """Estrae tutte le stanze reali dai file dungeon in raw_dir.

    Args:
        raw_dir: cartella con i file dungeon .txt.
        to_visual: applica il transpose all'orientamento di gioco (11x16).

    Returns:
        (rooms, sources): array (N, H, W) e lista parallela di tag "file:i,j".
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