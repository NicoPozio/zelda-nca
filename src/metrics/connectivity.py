# Metrica di connettivita' topologica.
#
# Il criterio NON e' assoluto ("dall'interno raggiungi tutte le porte") ma
# RELATIVO alla stanza intatta. Motivo: il VGLC annota la geometria ma non le
# affordance del gioco (muri bombabili, blocchi spingibili), quindi diverse stanze
# intatte hanno gia' un accesso irraggiungibile da un BFS ingenuo. Un criterio
# assoluto misurerebbe i buchi del dataset invece del modello.
#
# La stanza intatta e' la verita' di riferimento: la riparazione ha successo se
# le celle rilevanti si raggruppano nelle stesse componenti connesse.
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from src.tiles import CHAR_MAP, walkable_mask

# 4-adiacenza: Link si muove su/giu'/sinistra/destra, non in diagonale
_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

ACCESS_CHARS = ("D", "S")   # porte e scale: i punti che Link deve poter raggiungere


def walkable_components(room, passable_element_floor: bool = True):
    """Componenti connesse (4-adiacenza) delle celle calpestabili.

    Returns:
        (labels, n): etichette (0 = non calpestabile) e numero di componenti.
    """
    return label(walkable_mask(room, passable_element_floor), structure=_STRUCT)


def access_mask(room):
    """Maschera booleana dei punti di accesso (porte e scale)."""
    room = np.asarray(room)
    m = np.zeros(room.shape, dtype=bool)
    for c in ACCESS_CHARS:
        m |= (room == CHAR_MAP[c])
    return m


def main_component_mask(room, passable_element_floor: bool = True):
    """Maschera della componente calpestabile piu' grande (l'area interna principale)."""
    labels, n = walkable_components(room, passable_element_floor)
    if n == 0:
        return np.zeros(np.asarray(room).shape, dtype=bool)
    sizes = [(labels == k).sum() for k in range(1, n + 1)]
    return labels == 1 + int(np.argmax(sizes))


def partition_signature(room, probe_mask, passable_element_floor: bool = True):
    """Firma canonica di come le celle sonda si raggruppano in componenti.

    Per ogni cella sonda (in ordine row-major) restituisce l'indice della prima
    cella sonda che sta nella sua stessa componente, oppure -1 se in questa stanza
    la cella non e' calpestabile. Due stanze hanno la stessa firma se e solo se
    raggruppano le sonde allo stesso modo: il confronto non dipende da come sono
    numerate le componenti.
    """
    labels, _ = walkable_components(room, passable_element_floor)
    probe_idx = np.flatnonzero(np.asarray(probe_mask).ravel())
    lab_flat = labels.ravel()

    sig = np.full(len(probe_idx), -1, dtype=np.int64)
    first_of_component = {}
    for k, cell in enumerate(probe_idx):
        comp = int(lab_flat[cell])
        if comp == 0:
            continue                       # cella non calpestabile qui
        if comp not in first_of_component:
            first_of_component[comp] = int(cell)
        sig[k] = first_of_component[comp]
    return sig


def probe_mask(pristine, passable_element_floor: bool = True):
    """Celle su cui si valuta la topologia: l'area interna principale piu' gli accessi.

    Includere entrambe e' necessario: con le sole porte la firma sarebbe banale e
    non direbbe se la porta e' ancora raggiungibile dall'interno.
    """
    return main_component_mask(pristine, passable_element_floor) | access_mask(pristine)


def preserves_topology(pristine, repaired, passable_element_floor: bool = True) -> bool:
    """True se la stanza riparata conserva accessi e struttura di connettivita'.

    Due condizioni, entrambe necessarie per la giocabilita':
      1. gli accessi (porte e scale) sono ancora li', e non ne compaiono di nuovi:
         una porta diventata pavimento e' una stanza senza uscita, anche se l'area
         calpestabile resta connessa;
      2. le celle rilevanti si raggruppano nelle stesse componenti dell'intatta.
    """
    pristine = np.asarray(pristine)
    repaired = np.asarray(repaired)
    if pristine.shape != repaired.shape:
        raise ValueError("le due stanze devono avere la stessa forma")

    if not np.array_equal(access_mask(pristine), access_mask(repaired)):
        return False

    probes = probe_mask(pristine, passable_element_floor)
    if not probes.any():
        return True                        # niente di calpestabile da preservare
    sig_p = partition_signature(pristine, probes, passable_element_floor)
    sig_r = partition_signature(repaired, probes, passable_element_floor)
    return bool(np.array_equal(sig_p, sig_r))


def rsr(pristine_batch, repaired_batch, passable_element_floor: bool = True) -> float:
    """Regeneration Success Rate: frazione di stanze che preservano la topologia."""
    ok = [preserves_topology(p, r, passable_element_floor)
          for p, r in zip(pristine_batch, repaired_batch)]
    return float(np.mean(ok)) if ok else 0.0


def tile_accuracy(pristine_batch, repaired_batch) -> float:
    """Accuratezza per tile: la metrica 'pixel' contro cui confrontare la RSR."""
    return float((np.asarray(pristine_batch) == np.asarray(repaired_batch)).mean())