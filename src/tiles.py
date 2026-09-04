"""
File per lettura dati .txt scaricati dal VGLC
"""

from __future__ import annotations

import numpy as np

#geometria
RAW_ROWS = 16   #righe di una stanza nel file .txt
RAW_COLS = 11   #colonne di una stanza nel file .txt
ROOM_H = 11     #altezza stanza di lavoro
ROOM_W = 16     #larghezza stanza di lavoro

#Corrispondenza char-int mantenuta in tutto il progetto
CHAR_MAP: dict[str, int] = {
    'F': 0, 'B': 1, 'M': 2, 'P': 3, 'O': 4,
    'I': 5, 'D': 6, 'S': 7, 'W': 8, '-': 9,
}
IDX_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_MAP.items()}
NUM_TILES: int = len(CHAR_MAP)   #10 canali visibili

TILE_NAMES: dict[str, str] = {
    'F': 'floor', 'B': 'block', 'M': 'monster', 'P': 'element', 'O': 'element+floor',
    'I': 'element+block', 'D': 'door', 'S': 'stair', 'W': 'wall', '-': 'void',
}


#Calpestabilità di una stanza
def walkable_mask(room, passable_element_floor: bool = True) -> np.ndarray:
    """Maschera booleana delle celle calpestabili in una stanza

    Calpestabili: F, D, M, S, O

    Caso speciale '-':
      1) la stanza non ha neanche una casella floor F  -> è una stanza segreta dove '-' è calpestabile
      2) la stanza ha delle caselle floor F -> i caratteri '-' sono precipizi, non calpestabili

    Non calpestabili: W (muro), B (block), P (elemento profondo),I (element+block)
    """
    room = np.asarray(room)
    walk_chars = {'F', 'D', 'M', 'S'}
    if passable_element_floor:
        walk_chars.add('O')
    walk_idx = [CHAR_MAP[c] for c in walk_chars]
    mask = np.isin(room, walk_idx)

    # '-' calpestabile solo se non ci sono altri caratteri nella stanza
    if not np.any(room == CHAR_MAP['F']):
        mask |= (room == CHAR_MAP['-'])
    return mask


def door_mask(room) -> np.ndarray:
    """Maschera booleana delle celle Door di una stanza"""
    return np.asarray(room) == CHAR_MAP['D']