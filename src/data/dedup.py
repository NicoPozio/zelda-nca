#Deduplicazione delle stanze


from __future__ import annotations

from collections import OrderedDict

from src.data.symmetry import symmetries


def _canonical_key(room):
    """Se ho due stanze specchiate, il risultato di symmetries(room) è lo stesso
    solo che elecato in ordine diverso, ma prendendo il minimo so che sto prendendo
     la stessa chiave per entrambe le stanze """
    return min(s.tobytes() for s in symmetries(room))


def deduplicate(rooms, mode="symmetry", return_index=False):
    """Restituisce le stanze uniche secondo il tipo di simmetria specificato
    mode: 'none' | 'exact' | 'symmetry'
    Con return_index restituisce anche gli indici delle stanze tenute
    """

    #Nessuna deduplicazione
    if mode == "none":
        idx = list(range(len(rooms)))
        return (rooms, idx) if return_index else rooms

    if mode not in ("exact", "symmetry"):
        raise ValueError(f"mode sconosciuto: {mode!r}")

    """Se mode=="exact" allora valutiamo l'uguaglianza delle stanze in maniera immediata con una lambda
    senno usiamo la funzione privata _canonical_key, key è la funzione chiave da usare """
    key = (lambda r: r.tobytes()) if mode == "exact" else _canonical_key
    #Un OrderedDict è un dizionario che ricorda l'ordine di inserimento
    seen = OrderedDict()
    for n, r in enumerate(rooms):
        #Calcola chiave di byte
        k = key(r)
        if k not in seen:
            seen[k] = n
    idx = list(seen.values())
    #Solo le stanze uniche manteniamo, sono sempre ordinate in base a n, 
    #se cosi non fosse ogni volta gli split sarebbero diversi nonostante lo stesso seed.
    uniq = rooms[idx] 
    return (uniq, idx) if return_index else uniq
