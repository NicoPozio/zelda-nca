# Danni stocastici, usati SOLO in training per instillare la capacita' di riparare.
# Sono danni non mirati: non insegnano la topologia specifica (porte, muri), quindi
# non intaccano la claim che i danni mirati (famiglia B) siano out-of-distribution.
#
# A1 erasure:   azzera una patch contigua -> celle morte da rifar crescere.
# A2 tile-flip: sostituisce celle sparse con un tile sbagliato -> struttura errata.
#
# Entrambi operano sullo stato NCA (N, C, H, W) e azzerano TUTTI i canali della
# cella colpita (visibili e nascosti), cosi' non si creano stati contraddittori
# in cui il tile visibile e la memoria nascosta si contraddicono.
from __future__ import annotations

import torch

from src.tiles import NUM_TILES


def _check_fraction(fraction: float):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction deve stare in (0, 1]")


def erasure(state: torch.Tensor, rng, fraction: float):
    """A1: azzera una patch rettangolare contigua in ogni stanza del batch.

    Args:
        state: stato NCA (N, C, H, W).
        rng: generatore numpy (np.random.default_rng), per la riproducibilita'.
        fraction: frazione dell'area coperta dalla patch (asse-x delle curve di danno).

    Returns:
        (state, mask): lo stato danneggiato e la maschera (N, 1, H, W) delle celle colpite.
    """
    _check_fraction(fraction)
    n, c, h, w = state.shape
    state = state.clone()
    mask = torch.zeros(n, 1, h, w, dtype=torch.bool, device=state.device)

    ph = min(h, max(1, round((fraction ** 0.5) * h)))
    pw = min(w, max(1, round((fraction ** 0.5) * w)))
    for i in range(n):
        top = int(rng.integers(0, h - ph + 1))
        left = int(rng.integers(0, w - pw + 1))
        state[i, :, top:top + ph, left:left + pw] = 0.0     # cella morta: tutti i canali a zero
        mask[i, 0, top:top + ph, left:left + pw] = True
    return state, mask


def tile_flip(state: torch.Tensor, rng, fraction: float):
    """A2: sostituisce celle sparse con un tile visibile sbagliato.

    Ogni cella colpita viene azzerata (anche i nascosti) e poi le si accende un
    canale visibile diverso da quello originale.

    Args:
        state: stato NCA (N, C, H, W).
        rng: generatore numpy.
        fraction: frazione di celle da corrompere.

    Returns:
        (state, mask): stato danneggiato e maschera (N, 1, H, W) delle celle colpite.
    """
    _check_fraction(fraction)
    n, c, h, w = state.shape
    state = state.clone()
    mask = torch.zeros(n, 1, h, w, dtype=torch.bool, device=state.device)

    k = max(1, round(fraction * h * w))
    for i in range(n):
        flat = rng.choice(h * w, size=k, replace=False)
        for pos in flat:
            r, col = int(pos // w), int(pos % w)
            true_tile = int(state[i, :NUM_TILES, r, col].argmax())
            wrong = int(rng.integers(0, NUM_TILES - 1))
            if wrong >= true_tile:              # scegli uniformemente tra i 9 tile diversi
                wrong += 1
            state[i, :, r, col] = 0.0           # azzera visibili e nascosti
            state[i, wrong, r, col] = 1.0       # accendi il tile sbagliato
            mask[i, 0, r, col] = True
    return state, mask