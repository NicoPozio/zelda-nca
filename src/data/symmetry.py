# Le 4 simmetrie che preservano la forma 11x16 di una stanza.
# Definite qui una volta sola: dedup e augment devono usare lo stesso gruppo,
# altrimenti la garanzia "niente leakage tra train e val" non regge piu'.
from __future__ import annotations


def symmetries(room):
    """Le 4 simmetrie: identita', flip orizzontale, flip verticale, rotazione di 180."""
    return [room, room[:, ::-1], room[::-1, :], room[::-1, ::-1]]