


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tiles import NUM_TILES


def _perception_filters(num_channels: int, use_laplacian: bool = False) -> torch.Tensor:
    """
    Filtri del NCA, restituisce un tensore di forma (n_filtri * num_channels, 1, 3, 3) 
    per una convoluzione depthwise
    (groups=num_channels), ogni canale viene convoluto con tutti i filtri.
    """
    identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

    """"Normalizziamo il filtro di sobel """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y = sobel_x.t()
    kernels = [identity, sobel_x, sobel_y]
    if use_laplacian:
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) / 8.0
        kernels.append(laplacian)
    base = torch.stack(kernels).unsqueeze(1)              

    return base.repeat(num_channels, 1, 1, 1)           


class NCA(nn.Module):

    def __init__(self, hidden_channels: int, mlp_hidden: int = 128,
                 update_prob: float = 0.5, use_laplacian: bool = False,
                 global_channels: int = 0):

        super().__init__()
        self.num_channels = NUM_TILES + hidden_channels
        self.update_prob = update_prob


        self.global_channels = global_channels
        if global_channels > hidden_channels:
            raise ValueError("global_channels non puo' superare hidden_channels")

        #Creazione filtri
        filters = _perception_filters(self.num_channels, use_laplacian)
        #Salvataggio filtri in un buffer persistente
        self.register_buffer("filters", filters)
        n_filters = filters.shape[0] // self.num_channels   

        #Implementiamo l'head MLP come una semplice convoluzione 1x1
        #Se usassimo nn.Linear allora dovremmo ricordarci di cambiare la struttura
        #dell'output post convoluzione
        self.w1 = nn.Conv2d(n_filters * self.num_channels, mlp_hidden, kernel_size=1)
        self.w2 = nn.Conv2d(mlp_hidden, self.num_channels, kernel_size=1)

        #I pesi e i bias dell'ultimo strato sono inizializzati a 0, l'NCA inizia non facendo niente
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Convoluzione depthwise coi filtri fissi, con padding replicato ai bordi"""
        padded = F.pad(state, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, self.filters, groups=self.num_channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        delta = self.w2(F.relu(self.w1(self.perceive(state))))
        n, _, h, w = state.shape
        alive = (torch.rand(n, 1, h, w, device=state.device) <= self.update_prob).float()
        state = state + alive * delta

        if self.global_channels > 0:
            g = state[:, -self.global_channels:]
            pooled = g.mean(dim=(2, 3), keepdim=True).expand_as(g)
            state = torch.cat([state[:, :-self.global_channels], pooled], dim=1)
        return state

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, update_prob={self.update_prob}"