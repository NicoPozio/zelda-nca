# Sample pool per il training dell'NCA.
# E' un buffer di stati persistenti: a ogni iterazione si pesca un batch, lo si fa
# evolvere per alcuni passi e lo si riscrive nel pool. Cosi' il training riparte da
# stati gia' parzialmente ricostruiti invece che da zero, il che stabilizza il BPTT
# e insegna all'NCA a mantenere stabile una stanza valida.
#
# A differenza del Growing NCA classico (un solo target), qui ci sono molte stanze
# diverse: ogni slot ricorda la propria stanza target, cosi' la loss confronta
# ciascuno stato con la stanza giusta.
from __future__ import annotations

import numpy as np
import torch

from src.models.encoding import to_nca_state


class SamplePool:

    def __init__(self, rooms, pool_size: int, hidden_channels: int,
                 device: str = "cpu", seed: int = 0):
        """Inizializza il pool pescando stanze a caso dal dataset.

        Args:
            rooms: array (M, H, W) di indici di tile, le stanze di training.
            pool_size: numero di slot del pool.
            hidden_channels: canali nascosti dello stato NCA.
            device: dove tenere i tensori del pool.
            seed: seme per la scelta casuale delle stanze.
        """
        self.rooms = torch.as_tensor(np.asarray(rooms), dtype=torch.long)
        self.pool_size = pool_size
        self.hidden_channels = hidden_channels
        self.device = device
        self.rng = np.random.default_rng(seed)

        idx = self.rng.integers(0, len(self.rooms), size=pool_size)
        self.target = self.rooms[idx].to(device)                       # (pool, H, W) indici
        self.state = to_nca_state(self.target, hidden_channels).to(device)  # (pool, C, H, W)

    def sample(self, batch_size: int):
        """Pesca un batch di slot dal pool.

        Returns:
            (slots, states, targets): gli indici degli slot (per riscriverli poi),
            gli stati NCA correnti e le stanze target corrispondenti.
        """
        slots = self.rng.choice(self.pool_size, size=batch_size, replace=False)
        slots = torch.as_tensor(slots, device=self.device)
        return slots, self.state[slots].clone(), self.target[slots].clone()

    def commit(self, slots, states):
        """Riscrive nel pool gli stati evoluti (staccati dal grafo)."""
        self.state[slots] = states.detach().to(self.device)

    def reseed(self, slots):
        """Sostituisce gli slot indicati con stanze fresche non danneggiate.

        Serve a tenere il pool ancorato a stanze valide ed evitare che tutti gli
        stati derivino verso configurazioni degeneri.
        """
        idx = self.rng.integers(0, len(self.rooms), size=len(slots))
        fresh = self.rooms[idx].to(self.device)
        self.target[slots] = fresh
        self.state[slots] = to_nca_state(fresh, self.hidden_channels).to(self.device)

    def state_dict(self):
        """Stato serializzabile del pool, per i checkpoint."""
        return {"state": self.state.cpu(), "target": self.target.cpu(),
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, sd):
        self.state = sd["state"].to(self.device)
        self.target = sd["target"].to(self.device)
        self.rng.bit_generator.state = sd["rng"]