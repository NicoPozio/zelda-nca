# Costruzione dello stato NCA a partire dalle stanze in indici di tile.
# Lo stato che l'NCA fa evolvere ha due parti: i canali visibili (one-hot dei 10
# tile) e i canali nascosti, azzerati all'inizio, che servono alle celle per
# comunicare. Il numero di nascosti e' un iperparametro (varia nelle ablation).
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.tiles import CHAR_MAP, NUM_TILES


def to_one_hot(rooms: torch.Tensor) -> torch.Tensor:
    """Converte stanze di indici in one-hot, il target della CrossEntropy.

    Args:
        rooms: tensore (N, H, W) di indici di tile (interi in [0, NUM_TILES)).

    Returns:
        Tensore (N, NUM_TILES, H, W) float con l'one-hot sui canali visibili.
    """
    one_hot = F.one_hot(rooms.long(), num_classes=NUM_TILES)   # (N, H, W, C)
    return one_hot.permute(0, 3, 1, 2).float()                 # (N, C, H, W)


def to_nca_state(rooms: torch.Tensor, hidden_channels: int) -> torch.Tensor:
    """Costruisce lo stato NCA: canali visibili one-hot piu' i nascosti azzerati.

    Args:
        rooms: tensore (N, H, W) di indici di tile.
        hidden_channels: numero di canali nascosti (iperparametro, varia nelle ablation).

    Returns:
        Tensore (N, NUM_TILES + hidden_channels, H, W) float; i canali visibili
        contengono l'one-hot, i nascosti sono a zero.
    """
    visible = to_one_hot(rooms)                                # (N, NUM_TILES, H, W)
    n, _, h, w = visible.shape
    hidden = torch.zeros((n, hidden_channels, h, w), dtype=visible.dtype, device=visible.device)
    return torch.cat([visible, hidden], dim=1)


def visible_channels(state: torch.Tensor) -> torch.Tensor:
    """Estrae i soli canali visibili da uno stato NCA (per loss e decodifica)."""
    return state[:, :NUM_TILES]

def decode(state: torch.Tensor, dead_threshold: float = 1e-3,
           dead_tile: int = CHAR_MAP['-']) -> torch.Tensor:
    """Decodifica uno stato NCA in indici di tile.

    Una cella i cui canali visibili sono tutti sotto soglia non contiene nessun
    tile: e' una cella morta che il modello non ha mai riempito. Va decodificata
    esplicitamente, non con un argmax che su un vettore di zeri restituirebbe
    l'indice 0 (cioe' un tile a caso, deciso dall'ordine dell'alfabeto).
    Il default e' il void: una cella morta non contiene nulla, e void e' proprio
    il tile del nulla. Nelle stanze con pavimento il void e' un precipizio, quindi
    non calpestabile: e' la scelta conservativa per una metrica di giocabilita'.

    Args:
        state: stato NCA (N, C, H, W).
        dead_threshold: soglia sotto la quale un canale e' considerato spento.
        dead_tile: indice del tile con cui decodificare le celle morte.

    Returns:
        Tensore (N, H, W) di indici di tile.
    """
    visible = visible_channels(state)
    idx = visible.argmax(dim=1)
    dead = visible.max(dim=1).values < dead_threshold
    idx = idx.masked_fill(dead, dead_tile)
    return idx