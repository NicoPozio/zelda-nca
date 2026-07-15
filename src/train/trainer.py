# Loop di training dell'NCA, in regime di riparazione.
# A ogni iterazione: pesca un batch dal pool, ripulisce il campione peggiore
# (reseed), danneggia una parte del batch con danno stocastico, srotola l'NCA per
# un numero casuale di passi (BPTT), calcola la CrossEntropy sui canali visibili
# rispetto al target intatto, aggiorna i pesi e riscrive gli stati nel pool.
#
# Difese contro l'instabilita' del BPTT (le altre stanno in nca.py):
#  - clip della norma del gradiente prima dello step (contro l'exploding);
#    il paper usa una normalizzazione L2 per-variabile, il clip e' l'equivalente
#    piu' semplice e standard;
#  - unroll di lunghezza variabile in [bptt_min, bptt_max];
#  - il sample pool, che accorcia l'orizzonte effettivo del backprop.
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.models.encoding import to_nca_state, visible_channels
from src.damage.stochastic import erasure, tile_flip


class Trainer:

    def __init__(self, nca, pool, *, lr, grad_clip, bptt_min, bptt_max, batch_size,
                 damage_prob, damage_fractions, device="cpu", seed=0):
        """Args:
            nca: il modello NCA.
            pool: il SamplePool con le stanze di training.
            lr: learning rate di Adam.
            grad_clip: norma massima del gradiente prima dello step.
            bptt_min, bptt_max: estremi del numero di passi srotolati (compreso il max).
            batch_size: stanze per iterazione.
            damage_prob: frazione del batch a cui applicare danno stocastico.
            damage_fractions: lista di estensioni del danno tra cui campionare.
            device: 'cpu' o 'cuda'.
            seed: seme per danno, unroll e reseed.
        """
        self.nca = nca.to(device)
        self.pool = pool
        self.opt = Adam(self.nca.parameters(), lr=lr)
        self.grad_clip = grad_clip
        self.bptt_min = bptt_min
        self.bptt_max = bptt_max
        self.batch_size = batch_size
        self.damage_prob = damage_prob
        self.damage_fractions = list(damage_fractions)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.step = 0

    def _damage_batch(self, states, skip):
        """Danneggia una parte del batch con A1 o A2, saltando l'indice reseedato."""
        b = states.shape[0]
        candidates = [i for i in range(b) if i != skip]
        n_dmg = int(round(self.damage_prob * b))
        if n_dmg == 0 or not candidates:
            return states
        idxs = self.rng.choice(candidates, size=min(n_dmg, len(candidates)), replace=False)
        frac = float(self.rng.choice(self.damage_fractions))
        fn = erasure if self.rng.random() < 0.5 else tile_flip
        states = states.clone()
        damaged, _ = fn(states[idxs], self.rng, frac)
        states[idxs] = damaged
        return states

    def train_step(self):
        slots, states, targets = self.pool.sample(self.batch_size)
        states = states.to(self.device)
        targets = targets.to(self.device)

        # reseed del campione peggiore: rimette una stanza pulita, ripulendo il pool
        with torch.no_grad():
            per_sample = F.cross_entropy(
                visible_channels(states), targets, reduction="none"
            ).mean(dim=(1, 2))
        worst = int(per_sample.argmax())
        clean = to_nca_state(targets[worst:worst + 1], self.pool.hidden_channels)
        states[worst] = clean.to(self.device)[0]

        # danno stocastico sugli altri campioni
        states = self._damage_batch(states, skip=worst)

        # unroll BPTT per un numero di passi casuale
        n_steps = int(self.rng.integers(self.bptt_min, self.bptt_max + 1))
        for _ in range(n_steps):
            states = self.nca(states)

        loss = F.cross_entropy(visible_channels(states), targets)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nca.parameters(), self.grad_clip)
        self.opt.step()

        self.pool.commit(slots, states)
        self.step += 1
        return loss.item()

    def train(self, steps, log_every=100, checkpoint_every=0, ckpt_path=None, log_fn=print):
        for _ in range(steps):
            loss = self.train_step()
            if log_every and self.step % log_every == 0:
                log_fn(f"step {self.step}  loss {loss:.4f}")
            if checkpoint_every and ckpt_path and self.step % checkpoint_every == 0:
                self.save(ckpt_path)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def state_dict(self):
        return {"nca": self.nca.state_dict(), "opt": self.opt.state_dict(),
                "pool": self.pool.state_dict(), "step": self.step,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, sd):
        self.nca.load_state_dict(sd["nca"])
        self.opt.load_state_dict(sd["opt"])
        self.pool.load_state_dict(sd["pool"])
        self.step = sd["step"]
        self.rng.bit_generator.state = sd["rng"]