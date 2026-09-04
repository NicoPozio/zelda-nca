"""
Danni stocastici, usati in training
A1 erasure: azzera una patch contigua
A2 tile-flip: sostituisce celle sparse con tile di altre categorie
"""
from __future__ import annotations
import torch
from src.tiles import NUM_TILES


def _check_fraction(fraction: float):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction deve stare in (0, 1]")


def erasure(state: torch.Tensor, rng, fraction: float):
    """
    Azzera una patch rettangolare contigua in ogni stanza del batch
    Args:
        state: stato NCA (N, C, H, W)
        rng: generatore numpy (np.random.default_rng), per la riproducibilita'
        fraction: frazione dell'area di danno

    Returns:
        (state, mask): lo stato danneggiato e la maschera (N, 1, H, W) delle celle colpite
    """
    _check_fraction(fraction)
    n, c, h, w = state.shape
    state = state.clone()
    mask = torch.zeros(n, 1, h, w, dtype=torch.bool, device=state.device)

    #Usiamo il max perche per valori di fraction molto piccoli round darebbe 0
    #e quindi non cancelleremmo nessuna tile
    ph = min(h, max(1, round((fraction ** 0.5) * h)))
    pw = min(w, max(1, round((fraction ** 0.5) * w)))
    for i in range(n):
        #Per ogni stanza si calcola l'angolo sinistro in alto della zona da cancellare
        #cosi poi con lo slicing si annulla in un colpo solo tutta la patch
        top = int(rng.integers(0, h - ph + 1))
        left = int(rng.integers(0, w - pw + 1))
        state[i, :, top:top + ph, left:left + pw] = 0.0     #cella morta
        mask[i, 0, top:top + ph, left:left + pw] = True
    return state, mask


def tile_flip(state: torch.Tensor, rng, fraction: float):
    """
    Sostituisce celle sparse con altre tile
    Ogni cella colpita viene azzerata e poi si mette a 1 un
    canale visibile diverso da quello originale

    Args:
        state: stato NCA (N, C, H, W)
        rng: generatore numpy
        fraction: frazione di celle da corrompere

    Returns:
        (state, mask): stato danneggiato e maschera (N, 1, H, W) delle celle colpite
    """
    _check_fraction(fraction)
    n, c, h, w = state.shape
    state = state.clone()
    mask = torch.zeros(n, 1, h, w, dtype=torch.bool, device=state.device)

    #Garanzia che il danno venga applicato su almeno 1 cella
    k = max(1, round(fraction * h * w))
    for i in range(n):
        #Estrazione degli indici delle celle da sostituire
        flat = rng.choice(h * w, size=k, replace=False)
        for pos in flat:
            r, col = int(pos // w), int(pos % w)
            #Si vede a quale categoria appartiene la tile in quella posizione
            true_tile = int(state[i, :NUM_TILES, r, col].argmax())
            #Si estrae una categoria sbagliata casuale
            wrong = int(rng.integers(0, NUM_TILES - 1))
            if wrong >= true_tile:              #Scegli uniformemente tra i 9 tile diversi
                wrong += 1
            state[i, :, r, col] = 0.0           #Azzera tutti i canali
            state[i, wrong, r, col] = 1.0       #Metti a 1 la categoria sbagliata
            mask[i, 0, r, col] = True
    return state, mask