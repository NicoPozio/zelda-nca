"""
Funzioni per modello NCA con implementata distanza BFS
come segnale per cercare di superare limiti di località
"""
from __future__ import annotations

from collections import deque

import numpy as np
from src.metrics.connectivity import access_mask
from src.tiles import walkable_mask

_NEIGH = ((-1, 0), (1, 0), (0, -1), (0, 1))   

def access_distance_field(room, passable_element_floor: bool = True) -> np.ndarray:
    """
    Funzione che ritorna per ogni stanza un array contenente la distanza
    normalizzata per ogni cella rispetto all'accesso piu vicina

    Args:
        room: stanza (H, W) in indici di tile

    Returns:
        Array float32 (H, W) con valori in [0, 1]
    """
    room = np.asarray(room)
    h, w = room.shape
    walk = walkable_mask(room, passable_element_floor)
    acc = access_mask(room) & walk

    dist = np.full((h, w), np.inf, dtype=np.float32)

    #Inizializzazione coda con le coordinate degli accessi
    #Mettiamo a 0 gli accessi nel tensore distanze
    q = deque()
    for r, c in zip(*np.nonzero(acc)):
        dist[r, c] = 0.0
        q.append((r, c))

    #Per ogni cella della stanza calcoliamo la distanza rispetto all'accesso piu vicino
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGH:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and walk[rr, cc] and dist[rr, cc] == np.inf:
                dist[rr, cc] = dist[r, c] + 1.0
                q.append((rr, cc))

    #Scaliamo dove necessario
    scale = float(h + w)
    #Celle non calpestabili hanno distanza 1
    field = np.where(np.isfinite(dist), dist / scale, 1.0)
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def aux_targets(rooms, passable_element_floor: bool = True) -> np.ndarray:
    fields = [access_distance_field(r, passable_element_floor) for r in np.asarray(rooms)]
    #Convertiamo a forma adeguata per i modelli
    return np.stack(fields)[:, None]