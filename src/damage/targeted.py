# Danni mirati, usati SOLO in valutazione: l'NCA non li vede mai in training.
# Sono loro a rispondere alla research question: se il modello ripara un danno
# semanticamente mirato che non ha mai incontrato, e' perche' la regola locale ha
# codificato il vincolo topologico globale, non perche' ha memorizzato la riparazione.
#
# B1 door_ablation:       cancella dei punti di accesso (porte/scale)
# B2 wall_ablation:       cancella un segmento contiguo di muro perimetrale
# B3 access_isolation:    cancella le celle attorno a un accesso, isolandolo
# B4 articulation_removal: cancella i punti di articolazione del grafo calpestabile
#
# Il meccanismo e' lo stesso del danno stocastico (azzerare tutti i canali della
# cella): cambia solo QUALI celle vengono colpite. Cosi' l'unica variabile in gioco
# tra famiglia A e famiglia B e' la scelta del bersaglio.
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label

from src.tiles import CHAR_MAP, walkable_mask
from src.metrics.connectivity import access_mask, main_component_mask, _STRUCT


def kill_cells(state: torch.Tensor, mask: np.ndarray):
    """Azzera tutti i canali delle celle indicate: celle morte da rigenerare.

    Args:
        state: stato NCA (1, C, H, W) oppure (C, H, W).
        mask: maschera booleana (H, W) delle celle da uccidere.

    Returns:
        (state, mask_tensor): stato danneggiato e maschera come tensore (1, 1, H, W).
    """
    single = state.dim() == 3
    if single:
        state = state.unsqueeze(0)
    state = state.clone()
    m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=state.device)
    state[:, :, m] = 0.0
    out_mask = m.unsqueeze(0).unsqueeze(0)
    return (state.squeeze(0) if single else state), out_mask


def _neighbours(r, c, h, w):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def select_doors(room, rng, n_doors: int = 1) -> np.ndarray:
    """B1: seleziona le celle di alcuni punti di accesso (porte o scale).

    Gli accessi contigui vengono trattati come un'unica porta, cosi' 'una porta'
    significa l'intero varco e non un singolo tile.
    """
    room = np.asarray(room)
    acc = access_mask(room)
    mask = np.zeros(room.shape, dtype=bool)
    if not acc.any():
        return mask
    labels, n = label(acc, structure=_STRUCT)      # gruppi di accessi contigui
    chosen = rng.choice(np.arange(1, n + 1), size=min(n_doors, n), replace=False)
    for k in np.atleast_1d(chosen):
        mask |= (labels == int(k))
    return mask


def select_wall_segment(room, rng, length: int = 5) -> np.ndarray:
    """B2: seleziona un segmento contiguo di muro perimetrale.

    Sceglie un lato a caso e cancella `length` celle di muro consecutive lungo di
    esso: mette alla prova il ripristino dell'enclosure della stanza.
    """
    room = np.asarray(room)
    h, w = room.shape
    wall = (room == CHAR_MAP['W'])
    mask = np.zeros(room.shape, dtype=bool)

    side = rng.integers(0, 4)
    if side in (0, 1):                             # bordo alto / basso
        r = 0 if side == 0 else h - 1
        cols = np.flatnonzero(wall[r])
        if len(cols) == 0:
            return mask
        start = int(rng.integers(0, max(1, len(cols) - length + 1)))
        for c in cols[start:start + length]:
            mask[r, c] = True
    else:                                          # bordo sinistro / destro
        c = 0 if side == 2 else w - 1
        rows = np.flatnonzero(wall[:, c])
        if len(rows) == 0:
            return mask
        start = int(rng.integers(0, max(1, len(rows) - length + 1)))
        for r in rows[start:start + length]:
            mask[r, c] = True
    return mask


def select_access_isolation(room, rng, passable_element_floor: bool = True) -> np.ndarray:
    """B3: seleziona le celle calpestabili attorno a un accesso, per isolarlo.

    Cancella l'anello di celle percorribili adiacenti a un accesso: la porta resta
    al suo posto ma non e' piu' raggiungibile dall'interno. Mette alla prova il
    ripristino del collegamento, non del tile.
    """
    room = np.asarray(room)
    h, w = room.shape
    acc = access_mask(room)
    walk = walkable_mask(room, passable_element_floor)
    mask = np.zeros(room.shape, dtype=bool)
    if not acc.any():
        return mask

    labels, n = label(acc, structure=_STRUCT)
    k = int(rng.choice(np.arange(1, n + 1)))
    target = (labels == k)
    for r, c in zip(*np.nonzero(target)):
        for rr, cc in _neighbours(r, c, h, w):
            if walk[rr, cc] and not target[rr, cc]:
                mask[rr, cc] = True
    return mask


def articulation_points(room, passable_element_floor: bool = True) -> np.ndarray:
    """Punti di articolazione del grafo calpestabile (4-adiacenza).

    Una cella e' un punto di articolazione se rimuoverla spezza una componente in
    due: e' il collo di bottiglia della topologia. Calcolati per forza bruta, che
    su una griglia 11x16 e' immediato e meno soggetto a errori di un algoritmo
    dedicato.
    """
    walk = walkable_mask(np.asarray(room), passable_element_floor)
    _, base = label(walk, structure=_STRUCT)
    pts = np.zeros(walk.shape, dtype=bool)
    for r, c in zip(*np.nonzero(walk)):
        probe = walk.copy()
        probe[r, c] = False
        _, n = label(probe, structure=_STRUCT)
        if n > base:                               # la rimozione ha spezzato una componente
            pts[r, c] = True
    return pts


def select_articulation(room, rng, k: int = 1, passable_element_floor: bool = True) -> np.ndarray:
    """B4: seleziona k punti di articolazione, il danno minimo che disconnette."""
    pts = articulation_points(room, passable_element_floor)
    idx = np.flatnonzero(pts.ravel())
    mask = np.zeros(np.asarray(room).shape, dtype=bool)
    if len(idx) == 0:
        return mask
    chosen = rng.choice(idx, size=min(k, len(idx)), replace=False)
    flat = mask.ravel()
    flat[np.atleast_1d(chosen)] = True
    return flat.reshape(mask.shape)


# Registro dei danni mirati: nome -> funzione che restituisce la maschera.
# Ogni funzione ha firma (room, rng, **kwargs) -> maschera booleana (H, W).
TARGETED = {
    "B1_door": select_doors,
    "B2_wall": select_wall_segment,
    "B3_isolation": select_access_isolation,
    "B4_articulation": select_articulation,
}