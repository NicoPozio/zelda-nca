# Augmentation delle stanze tramite le 4 simmetrie che preservano la forma.
# Da applicare solo al train, dopo lo split, per non creare leakage.
from __future__ import annotations

import numpy as np

from src.data.symmetry import symmetries


def augment(rooms):
    """Espande le stanze con le loro 4 simmetrie, quadruplicando il train.

    Nota: una stanza gia' simmetrica produce copie ripetute; e' dentro il train,
    quindi non e' un problema di leakage, al massimo un lieve sovrappeso.
    """
    out = []
    for room in rooms:
        out.extend(symmetries(room))
    return np.stack(out)