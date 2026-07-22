# Bersagli ausiliari per il ramo multitask (Sub-RQ 3).
#
# Il segnale scelto e' la distanza geodetica dall'accesso piu' vicino: per ogni
# cella calpestabile, quanti passi servono per raggiungere una porta o una scala
# camminando. E' una quantita' GLOBALE (dipende dall'intera stanza) ma esprimibile
# come campo locale, quindi e' esattamente cio' che una regola locale dovrebbe
# saper propagare. E' anche l'analogo del gradiente chimico della morfogenesi, che
# e' la metafora biologica degli NCA.
#
# Il bersaglio si calcola sulla stanza INTATTA: il modello riceve in ingresso una
# stanza danneggiata e deve inferire il campo corretto. In inferenza i canali
# ausiliari non vengono usati (si decodificano solo i visibili), quindi non c'e'
# alcuna contaminazione della metrica.
from __future__ import annotations

from collections import deque

import numpy as np

from src.metrics.connectivity import access_mask
from src.tiles import walkable_mask

_NEIGH = ((-1, 0), (1, 0), (0, -1), (0, 1))   # 4-adiacenza, come la metrica


def access_distance_field(room, passable_element_floor: bool = True) -> np.ndarray:
    """Campo di distanza dall'accesso piu' vicino, normalizzato in [0, 1].

    BFS multi-sorgente sulle celle calpestabili, partendo dagli accessi. La
    normalizzazione usa una costante fissa (H + W) invece del massimo per stanza,
    cosi' il valore ha lo stesso significato in tutte le stanze. Celle non
    calpestabili o non raggiungibili valgono 1.0 (distanza massima).

    Args:
        room: stanza (H, W) in indici di tile.

    Returns:
        Array float32 (H, W) con valori in [0, 1].
    """
    room = np.asarray(room)
    h, w = room.shape
    walk = walkable_mask(room, passable_element_floor)
    acc = access_mask(room) & walk

    dist = np.full((h, w), np.inf, dtype=np.float32)
    q = deque()
    for r, c in zip(*np.nonzero(acc)):
        dist[r, c] = 0.0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGH:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and walk[rr, cc] and dist[rr, cc] == np.inf:
                dist[rr, cc] = dist[r, c] + 1.0
                q.append((rr, cc))

    scale = float(h + w)
    field = np.where(np.isfinite(dist), dist / scale, 1.0)
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def aux_targets(rooms, passable_element_floor: bool = True) -> np.ndarray:
    """Campi ausiliari per un insieme di stanze: (N, 1, H, W)."""
    fields = [access_distance_field(r, passable_element_floor) for r in np.asarray(rooms)]
    return np.stack(fields)[:, None]