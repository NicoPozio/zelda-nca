"""File per il formato VGLC di The Legend of Zelda (NES).

Unica fonte di verita' per la semantica dei tile: dati, loss, danno e metrica
importano tutti da qui indici e regola di calpestabilita', cosi' "cos'e' un
floor / una porta / un precipizio" e' definito in un solo posto.


Geometria e orientamento (verificati sui dati + confronto col PNG del gioco):
  * Nel .txt una stanza e' 16 righe x 11 colonne (RAW_ROWS x RAW_COLS).
  * Il .txt e' memorizzato TRASPOSTO rispetto a come Zelda appare a schermo:
    trasponendo si torna all'orientamento di gioco (landscape). Applichiamo il
    transpose in fase di load come CONVENZIONE VISIVA deliberata (rende leggibili
    le figure dei Results). E' topologicamente irrilevante -- il transpose
    preserva la 4-adiacenza -- ma comodo. Dopo il transpose la stanza di lavoro
    e' ROOM_H x ROOM_W = 11 x 16.
"""

from __future__ import annotations

import numpy as np

#geometria
RAW_ROWS = 16   #righe di una stanza nel file .txt
RAW_COLS = 11   #colonne di una stanza nel file .txt
ROOM_H = 11     #altezza stanza di lavoro
ROOM_W = 16     #larghezza stanza di lavoro

#Alfabeto canonico, l'ordine degli indici dipende solo
#dall'ordine di apparenza nel readme di VGLC
CHAR_MAP: dict[str, int] = {
    'F': 0, 'B': 1, 'M': 2, 'P': 3, 'O': 4,
    'I': 5, 'D': 6, 'S': 7, 'W': 8, '-': 9,
}
IDX_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_MAP.items()}
NUM_TILES: int = len(CHAR_MAP)   #10 canali visibili (one-hot)

TILE_NAMES: dict[str, str] = {
    'F': 'floor', 'B': 'block', 'M': 'monster', 'P': 'element', 'O': 'element+floor',
    'I': 'element+block', 'D': 'door', 'S': 'stair', 'W': 'wall', '-': 'void',
}


#Calpestabilità di una stanza
def walkable_mask(room, passable_element_floor: bool = True) -> np.ndarray:
    """Maschera booleana delle celle calpestabili di una stanza (indici tile).

    Calpestabili: F, D, M, S, O

    Caso speciale '-':
      1) la stanza non ha neanche una casella floor F  -> stanza dove c'è l'anziano, '-' è calpestabile
      2) la stanza ha delle caselle floor F -> i caratteri '-' sono precipizi, non calpestabili.

    Non calpestabili sempre: W (muro), B (block), P (elemento profondo),
    I (element+block).
    """
    room = np.asarray(room)
    walk_chars = {'F', 'D', 'M', 'S'}
    if passable_element_floor:
        walk_chars.add('O')
    walk_idx = [CHAR_MAP[c] for c in walk_chars]
    mask = np.isin(room, walk_idx)

    # '-' calpestabile solo nelle stanze senza floor (anziano)
    if not np.any(room == CHAR_MAP['F']):
        mask |= (room == CHAR_MAP['-'])
    return mask


def door_mask(room) -> np.ndarray:
    """Maschera booleana delle celle porta (target del BFS di connettivita')."""
    return np.asarray(room) == CHAR_MAP['D']