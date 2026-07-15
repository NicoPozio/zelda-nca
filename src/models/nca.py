# Neural Cellular Automata: la regola di update locale.
# Ogni cella osserva solo il vicinato 3x3 (percezione con filtri fissi), passa il
# vettore percepito a una piccola MLP (due Conv 1x1) e ne ricava un incremento
# Delta. L'update e' residuale (state = state + Delta) e stocastico.
#
# Padding replicato nella percezione: le stanze riempiono tutta la griglia fino al
# bordo, quindi un padding a zero creerebbe falsi gradienti Sobel sulle celle di
# muro perimetrali (quelle che definiscono l'enclosure). Il padding replicato da'
# derivata nulla su una regione costante, che e' il comportamento corretto.
#
# Difese contro l'instabilita' del BPTT: update residuale, ultimo Conv 1x1 a zero
# (Delta iniziale nullo), filtri Sobel normalizzati.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tiles import NUM_TILES


def _perception_filters(num_channels: int, use_laplacian: bool = False) -> torch.Tensor:
    """Banco di filtri fissi per la percezione.

    Sempre: identita', Sobel x, Sobel y. Con use_laplacian aggiunge il Laplaciano
    (derivata seconda, risponde a spot/angoli invece che a bordi direzionali).

    Restituisce (n_filtri * num_channels, 1, 3, 3) per una convoluzione depthwise
    (groups=num_channels): ogni canale viene convoluto con tutti i filtri.
    """
    identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y = sobel_x.t()
    kernels = [identity, sobel_x, sobel_y]
    if use_laplacian:
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) / 8.0
        kernels.append(laplacian)
    base = torch.stack(kernels).unsqueeze(1)              # (n_filtri, 1, 3, 3)
    return base.repeat(num_channels, 1, 1, 1)            # (n_filtri*num_channels, 1, 3, 3)


class NCA(nn.Module):

    def __init__(self, hidden_channels: int, mlp_hidden: int = 128,
                 update_prob: float = 0.5, use_laplacian: bool = False):
        """Regola di update dell'NCA.

        Args:
            hidden_channels: canali nascosti; lo stato ha NUM_TILES + hidden_channels canali.
            mlp_hidden: larghezza dello strato interno della MLP di update.
            update_prob: probabilita' che una cella si aggiorni a ogni passo.
            use_laplacian: se True aggiunge il Laplaciano ai filtri di percezione.
        """
        super().__init__()
        self.num_channels = NUM_TILES + hidden_channels
        self.update_prob = update_prob

        filters = _perception_filters(self.num_channels, use_laplacian)
        self.register_buffer("filters", filters)
        n_filters = filters.shape[0] // self.num_channels   # 3 oppure 4

        self.w1 = nn.Conv2d(n_filters * self.num_channels, mlp_hidden, kernel_size=1)
        self.w2 = nn.Conv2d(mlp_hidden, self.num_channels, kernel_size=1)

        # ultimo layer a zero: all'inizio Delta = 0, l'NCA parte da "non fare niente"
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Convoluzione depthwise coi filtri fissi, con padding replicato ai bordi."""
        padded = F.pad(state, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, self.filters, groups=self.num_channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Un passo di update: percezione, MLP, incremento residuale stocastico."""
        delta = self.w2(F.relu(self.w1(self.perceive(state))))
        n, _, h, w = state.shape
        alive = (torch.rand(n, 1, h, w, device=state.device) <= self.update_prob).float()
        return state + alive * delta

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, update_prob={self.update_prob}"