# Deduplicazione delle stanze.
# Zelda riusa gli stessi layout tra dungeon diversi anche nella sola prima quest:
# un template puo' comparire piu' volte. Senza dedup prima dello split, copie
# identiche finiscono sia in train che in val e falsano la metrica. La dedup a
# simmetria e' l'unica coerente con l'augmentation, che aggiunge quelle 4 simmetrie.
from __future__ import annotations

from collections import OrderedDict

from src.data.symmetry import symmetries


def _canonical_key(room):
    # chiave invariante per simmetria: i bytes della simmetria piu' piccola
    return min(s.tobytes() for s in symmetries(room))


def deduplicate(rooms, mode="symmetry", return_index=False):
    """Restituisce le stanze uniche secondo l'equivalenza scelta.

    mode: 'none' | 'exact' | 'symmetry'. Deterministico: tiene la prima occorrenza.
    Con return_index restituisce anche gli indici delle stanze tenute.
    """
    if mode == "none":
        idx = list(range(len(rooms)))
        return (rooms, idx) if return_index else rooms
    if mode not in ("exact", "symmetry"):
        raise ValueError(f"mode sconosciuto: {mode!r}")

    key = (lambda r: r.tobytes()) if mode == "exact" else _canonical_key
    seen = OrderedDict()
    for n, r in enumerate(rooms):
        k = key(r)
        if k not in seen:
            seen[k] = n
    idx = list(seen.values())
    uniq = rooms[idx]
    return (uniq, idx) if return_index else uniq