# Valutazione: danneggia le stanze, lascia riparare all'NCA, misura la topologia.
#
# Nota sulla natura dei danni. Un danno alle celle NON calpestabili (i muri) e'
# topologicamente neutro per costruzione: se il modello non lo ripara, le celle
# restano non calpestabili come il muro che c'era, e la connettivita' non cambia.
# Solo un danno al grafo calpestabile puo' spezzare la topologia. Per questo B1,
# B3 e B4 si misurano con la RSR, mentre B2 e' un test di sola fedelta' e va letto
# sulla tile accuracy.
from __future__ import annotations

import numpy as np
import torch

from src.damage.stochastic import erasure, tile_flip
from src.damage.targeted import TARGETED, articulation_points, kill_cells
from src.metrics.connectivity import rsr, tile_accuracy
from src.models.encoding import decode, to_nca_state


def _stochastic_damage(fn, fraction):
    """Adatta un danno stocastico alla firma comune (state, room, rng) -> (state, mask)."""
    def apply(state, room, rng):
        return fn(state, rng, fraction)
    return apply


def _targeted_damage(selector, **kwargs):
    """Adatta un selettore mirato alla firma comune: sceglie le celle, poi le uccide."""
    def apply(state, room, rng):
        return kill_cells(state, selector(room, rng, **kwargs))
    return apply


def _has_articulation(room):
    return bool(articulation_points(room).any())


def damage_suite(fractions=(0.2, 0.4, 0.6)):
    """Le condizioni di danno da valutare.

    Ogni voce e' (nome, estensione, funzione di danno, predicato di applicabilita',
    topologico). Il predicato serve a saltare le stanze su cui il danno sarebbe un
    no-op: B4 si applica solo dove esistono punti di articolazione, che nelle stanze
    aperte di Zelda spesso non ci sono.
    """
    always = lambda room: True
    suite = []
    for f in fractions:
        suite.append(("A1_erasure", f, _stochastic_damage(erasure, f), always, True))
        suite.append(("A2_tileflip", f, _stochastic_damage(tile_flip, f), always, True))
    suite.append(("B1_door", 1, _targeted_damage(TARGETED["B1_door"], n_doors=1), always, True))
    suite.append(("B2_wall", 5, _targeted_damage(TARGETED["B2_wall"], length=5), always, False))
    suite.append(("B3_isolation", 1, _targeted_damage(TARGETED["B3_isolation"]), always, True))
    suite.append(("B4_articulation", 1, _targeted_damage(TARGETED["B4_articulation"], k=1),
                  _has_articulation, True))
    return suite


@torch.no_grad()
def repair(nca, rooms, damage_fn, rng, steps, hidden_channels, device="cpu"):
    """Danneggia ogni stanza, fa girare l'NCA per `steps` passi, decodifica.

    Returns:
        Array (N, H, W) delle stanze riparate, in indici di tile.
    """
    nca.eval()
    out = []
    for room in rooms:
        state = to_nca_state(torch.as_tensor(np.asarray(room)).unsqueeze(0),
                             hidden_channels).to(device)
        state, _ = damage_fn(state, room, rng)
        for _ in range(steps):
            state = nca(state)
        out.append(decode(state)[0].cpu().numpy())
    return np.stack(out)


def evaluate(nca, rooms, steps, hidden_channels, seeds=(1, 2, 3), device="cpu",
             fractions=(0.2, 0.4, 0.6)):
    """Valuta l'NCA su tutta la suite di danni, ripetendo su piu' seed.

    Returns:
        Lista di dizionari, una riga per (danno, estensione, seed).
    """
    rows = []
    for name, extent, fn, applies, topological in damage_suite(fractions):
        subset = np.stack([r for r in rooms if applies(r)]) if any(applies(r) for r in rooms) else None
        if subset is None:
            continue
        for seed in seeds:
            rng = np.random.default_rng(seed)
            repaired = repair(nca, subset, fn, rng, steps, hidden_channels, device)
            rows.append({
                "damage": name,
                "extent": extent,
                "seed": seed,
                "topological": topological,
                "n_rooms": len(subset),
                "rsr": rsr(subset, repaired),
                "tile_acc": tile_accuracy(subset, repaired),
            })
    return rows


def aggregate(rows):
    """Aggrega le righe per (danno, estensione) in media e deviazione standard."""
    keys = sorted({(r["damage"], r["extent"]) for r in rows})
    out = []
    for damage, extent in keys:
        sel = [r for r in rows if r["damage"] == damage and r["extent"] == extent]
        out.append({
            "damage": damage,
            "extent": extent,
            "topological": sel[0]["topological"],
            "n_rooms": sel[0]["n_rooms"],
            "rsr_mean": float(np.mean([r["rsr"] for r in sel])),
            "rsr_std": float(np.std([r["rsr"] for r in sel])),
            "acc_mean": float(np.mean([r["tile_acc"] for r in sel])),
            "acc_std": float(np.std([r["tile_acc"] for r in sel])),
            "n_seeds": len(sel),
        })
    return out