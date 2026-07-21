# U-Net come baseline per isolare la localita' dell'NCA.
#
# L'NCA ripara con una regola LOCALE (vicinato 3x3) applicata ITERATIVAMENTE.
# Confondere le due cose renderebbe il confronto inutile, quindi l'U-Net si usa in
# due modi che isolano una variabile per volta:
#   - iterativo  : stesso loop dell'NCA ma campo recettivo GLOBALE -> isola la localita'
#   - one-shot   : un solo passaggio -> isola l'iterativita'
#
# L'interfaccia e' la stessa dell'NCA: prende uno stato (N, C, H, W) e ne restituisce
# uno della stessa forma, cosi' pool, trainer e valutazione non cambiano. I canali
# nascosti non servono a un modello globale, ma si tengono per parita' di interfaccia.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tiles import NUM_TILES


def _conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, padding_mode="replicate"),
        nn.GroupNorm(1, cout),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1, padding_mode="replicate"),
        nn.GroupNorm(1, cout),
        nn.ReLU(),
    )


class UNet(nn.Module):

    def __init__(self, hidden_channels: int, base: int = 32, iterative: bool = True):
        """U-Net encoder-decoder con due livelli di pooling.

        Args:
            hidden_channels: come nell'NCA, per stato (NUM_TILES + hidden_channels).
            base: canali del primo livello (raddoppiano scendendo).
            iterative: se True e' pensato per il loop di riparazione; se False per
                un singolo passaggio. Non cambia l'architettura, solo come lo si usa;
                il flag e' salvato per chiarezza e per il logging.
        """
        super().__init__()
        self.num_channels = NUM_TILES + hidden_channels
        self.iterative = iterative

        self.enc1 = _conv(self.num_channels, base)
        self.enc2 = _conv(base, base * 2)
        self.bottleneck = _conv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv(base * 2, base)
        self.head = nn.Conv2d(base, self.num_channels, 1)

        # come l'NCA: ultimo layer a zero -> update residuale inizialmente nullo
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # padding a multiplo di 4 (due pooling), poi si ritaglia: 11x16 non e' divisibile
        _, _, h, w = state.shape
        ph, pw = (-h) % 4, (-w) % 4
        x = F.pad(state, (0, pw, 0, ph), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        delta = self.head(d1)

        delta = delta[:, :, :h, :w]                       # ritaglio al formato originale
        return state + delta                              # residuale, come l'NCA

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, iterative={self.iterative}"