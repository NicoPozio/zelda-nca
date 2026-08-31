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
    """Gruppi di filtri.

    I filtri standard sono identità, Sobel x, Sobel y. 
    Se use_laplacian è true allora si aggiunge anche il filtro di Laplace.
    (derivata seconda, risponde a spot/angoli invece che a bordi direzionali).
    Restituisce un tensore di forma (n_filtri * num_channels, 1, 3, 3) per una convoluzione depthwise
    (groups=num_channels), ogni canale viene convoluto con tutti i filtri.
    """
    identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

    """"Normalizziamo il filtro di sobel dividendo per 8.0f per combattere contro instabilità della BPTT"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y = sobel_x.t()
    kernels = [identity, sobel_x, sobel_y]
    if use_laplacian:
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) / 8.0
        kernels.append(laplacian)
    base = torch.stack(kernels).unsqueeze(1)              # (n_filtri, 1, 3, 3)
    #num_channels è il numero di canali (hidden e non)
    return base.repeat(num_channels, 1, 1, 1)            # (n_filtri*num_channels, 1, 3, 3)


class NCA(nn.Module):

    def __init__(self, hidden_channels: int, mlp_hidden: int = 128,
                 update_prob: float = 0.5, use_laplacian: bool = False,
                 global_channels: int = 0):
        """Regola di update dell'NCA.

        global_channels: numero di canali nascosti che, dopo ogni passo, vengono
            sostituiti dalla loro media spaziale e ridistribuiti a tutte le celle.
            E' un rilassamento minimo della localita': ogni cella continua a
            leggere solo il proprio 3x3, ma riceve in piu' un riassunto scalare
            dello stato globale, l'analogo di un segnale ormonale. Con 0 il
            modello e' esattamente quello locale.
        """
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
        n_filters = filters.shape[0] // self.num_channels   # 3 oppure 4

        #Implementiamo l'head MLP come una semplice convoluzione 1x1
        #Se usassimo nn.Linear allora dovremmo ricordarci di cambiare la struttura
        #dell'output post convoluzione
        self.w1 = nn.Conv2d(n_filters * self.num_channels, mlp_hidden, kernel_size=1)
        self.w2 = nn.Conv2d(mlp_hidden, self.num_channels, kernel_size=1)

        #i pesi e i bias dell'ultimo strato sono inizializzati a 0, l'NCA inizia non facendo niente
        """
        Non è un problema per l'aggiornamento del gradiente, stiamo solo dicendo al NCA alla prima iterazione 
        di non fare niente
        """
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Convoluzione depthwise coi filtri fissi, con padding replicato ai bordi."""
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