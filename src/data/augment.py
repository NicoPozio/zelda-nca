# Augmentation delle stanze tramite le 4 simmetrie che preservano la forma
from __future__ import annotations

import numpy as np

from src.data.symmetry import symmetries


def augment(rooms):
    """
    Espande il training set con le simmetrie, quadruplicandone le dimensioni
    Non abbiamo problemi di leakage perchè l'augmentation viene fatta dopo 
    lo split di train test e val
    """
    out = []
    for room in rooms:
        #Perché non append?
        out.extend(symmetries(room))
    return np.stack(out)