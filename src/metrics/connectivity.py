from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from src.tiles import CHAR_MAP, walkable_mask

#Array che esprime le direzioni in cui si può spostare Link, serve per definire le componenti connesse
_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

ACCESS_CHARS = ("D", "S")   # porte e scale: i punti che Link deve poter raggiungere


def walkable_components(room, passable_element_floor: bool = True):
    """Componenti connesse composte da celle calpestabili

    Returns:
        (labels, n): etichette (0 = non calpestabile) e numero di componenti
    """
    return label(walkable_mask(room, passable_element_floor), structure=_STRUCT)


def access_mask(room):
    """Maschera booleana dei punti di accesso"""
    room = np.asarray(room)
    m = np.zeros(room.shape, dtype=bool)
    for c in ACCESS_CHARS:
        m |= (room == CHAR_MAP[c])
    return m


def main_component_mask(room, passable_element_floor: bool = True):
    """Maschera della componente calpestabile principale (la piu grande)"""
    labels, n = walkable_components(room, passable_element_floor)
    if n == 0:
        return np.zeros(np.asarray(room).shape, dtype=bool)
    sizes = [(labels == k).sum() for k in range(1, n + 1)]
    return labels == 1 + int(np.argmax(sizes))


def partition_signature(room, probe_mask, passable_element_floor: bool = True):
    """
    Ogni sonda (in ordine di riga) restituisce l'indice della prima
    sonda nella sua stessa componente, oppure -1 se in questa stanza
    la cella non e' calpestabile
    Due stanze hanno la stessa firma se e solo se
    raggruppano le sonde allo stesso modo
    """
    labels, _ = walkable_components(room, passable_element_floor)
    probe_idx = np.flatnonzero(np.asarray(probe_mask).ravel())
    lab_flat = labels.ravel()

    sig = np.full(len(probe_idx), -1, dtype=np.int64)
    first_of_component = {}
    for k, cell in enumerate(probe_idx):
        comp = int(lab_flat[cell])
        if comp == 0:
            continue                       
        if comp not in first_of_component:
            first_of_component[comp] = int(cell)
        sig[k] = first_of_component[comp]
    return sig


def probe_mask(pristine, passable_element_floor: bool = True):
    """
    Celle su cui si valuta la correttezza della riparazione:
    l'area interna principale piu' gli accessi
    """
    return main_component_mask(pristine, passable_element_floor) | access_mask(pristine)


def preserves_topology(pristine, repaired, passable_element_floor: bool = True) -> bool:
    pristine = np.asarray(pristine)
    repaired = np.asarray(repaired)
    if pristine.shape != repaired.shape:
        raise ValueError("le due stanze devono avere la stessa forma")

    if not np.array_equal(access_mask(pristine), access_mask(repaired)):
        return False

    probes = probe_mask(pristine, passable_element_floor)
    #Niente di calpestabile da preservare
    if not probes.any():
        return True                        #
    sig_p = partition_signature(pristine, probes, passable_element_floor)
    sig_r = partition_signature(repaired, probes, passable_element_floor)
    return bool(np.array_equal(sig_p, sig_r))


def rsr(pristine_batch, repaired_batch, passable_element_floor: bool = True) -> float:
    """Funzione per misurare la RSR (regeneration success rate)"""
    ok = [preserves_topology(p, r, passable_element_floor)
          for p, r in zip(pristine_batch, repaired_batch)]
    return float(np.mean(ok)) if ok else 0.0


def tile_accuracy(pristine_batch, repaired_batch) -> float:
    """Funzione per misurare accuracy a livello di tile"""
    return float((np.asarray(pristine_batch) == np.asarray(repaired_batch)).mean())