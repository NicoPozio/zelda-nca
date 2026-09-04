#Costruzione stato a partire dalle stanze


from __future__ import annotations
import torch
import torch.nn.functional as F
from src.tiles import CHAR_MAP, NUM_TILES


def to_one_hot(rooms: torch.Tensor) -> torch.Tensor:
    """One-hot encoding delle stanze

    Args:
        rooms: tensore (N, H, W) con interi 

    Returns:
        tensore (N, NUM_TILES, H, W) one-hot encoded
    """
    one_hot = F.one_hot(rooms.long(), num_classes=NUM_TILES)   
    return one_hot.permute(0, 3, 1, 2).float()                 


def to_nca_state(rooms: torch.Tensor, hidden_channels: int) -> torch.Tensor:
    """Aggiunge i canali nascosti

    Args:
        rooms: tensore (N, H, W) 
        hidden_channels: numero di canali nascosti

    Returns:
        tensore (N, NUM_TILES + hidden_channels, H, W)
    """
    visible = to_one_hot(rooms)                               
    n, _, h, w = visible.shape
    hidden = torch.zeros((n, hidden_channels, h, w), dtype=visible.dtype, device=visible.device)
    return torch.cat([visible, hidden], dim=1)


def visible_channels(state: torch.Tensor) -> torch.Tensor:
    """Estrazione canali visibili (serve per loss e decodifica)"""
    return state[:, :NUM_TILES]

def decode(state: torch.Tensor, dead_threshold: float = 1e-3,
           dead_tile: int = CHAR_MAP['-']) -> torch.Tensor:
    """Decodifica uno stato NCA in una stanza vera e propria

    (Se in una cella i canali visibili sono tutti sotto una certa soglia non contiene nessun
    tile viene considerata come void, quindi come cella non calpestabile)

    Args:
        state: stato NCA (N, C, H, W)
        dead_threshold: soglia sotto la quale un canale e' considerato spento
        dead_tile: indice del tile con cui decodificare le celle morte

    Returns:
        tensore (N, H, W) di indici di tile
    """
    visible = visible_channels(state)
    """Per ogni tile di ogni stanza prendi l'indice della categoria più "votata" da 0 a 9"""
    idx = visible.argmax(dim=1) 
    """Facciamo una seconda selezione tra queste, e selezioniamo le celle che risultano morte"""
    dead = visible.max(dim=1).values < dead_threshold
    """Per le celle che risultano dead mettiamo lo stato numerico di "-" che rappresenta il void"""
    idx = idx.masked_fill(dead, dead_tile)
    return idx