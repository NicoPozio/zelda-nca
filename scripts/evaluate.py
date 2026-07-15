# Valuta un checkpoint sulla suite di danni e stampa la tabella dei risultati.
#
# Uso:
#   python scripts/evaluate.py eval.ckpt=/kaggle/working/runs/m2/last.pt
#   python scripts/evaluate.py eval.ckpt=... eval.split=val eval.steps=96
#
# Il default e' il test set: usalo solo per i numeri finali del report. Durante le
# ablation usa eval.split=val, altrimenti scegli gli iperparametri guardando il
# test e i numeri finali risultano ottimistici.
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.splits import train_val_test_split
from src.eval.runner import aggregate, evaluate
from src.models.nca import NCA


def pick_device(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def model_cfg_of(ckpt_path, fallback):
    """Legge la configurazione del modello dalla run che ha prodotto il checkpoint.

    Hydra salva il config risolto in <run_dir>/.hydra/config.yaml. Usarlo evita di
    dover ripetere a mano gli override in valutazione: se il checkpoint viene da
    una run con hidden_channels=24, il modello viene ricostruito con 24 e i pesi
    si caricano. Senza questo, un override dimenticato darebbe un errore di shape.
    """
    saved = os.path.join(os.path.dirname(ckpt_path), ".hydra", "config.yaml")
    if os.path.exists(saved):
        cfg = OmegaConf.load(saved)
        if "model" in cfg:
            print(f"config del modello letta da {saved}")
            return cfg.model
    print("nessun config salvato accanto al checkpoint: uso quello corrente")
    return fallback


def print_table(rows_agg):
    print(f"\n{'danno':<18}{'estensione':>11}{'topo':>6}{'stanze':>8}"
          f"{'RSR':>17}{'tile acc':>17}")
    print("-" * 77)
    for a in rows_agg:
        topo = "si" if a["topological"] else "no"
        print(f"{a['damage']:<18}{a['extent']:>11}{topo:>6}{a['n_rooms']:>8}"
              f"{a['rsr_mean']:>11.3f}±{a['rsr_std']:<5.3f}"
              f"{a['acc_mean']:>11.3f}±{a['acc_std']:<5.3f}")
    print("\nRSR = frazione di stanze che conservano accessi e connettivita' dell'intatta.")
    print("B2 e' segnato non-topologico: un danno ai muri non puo' spezzare la")
    print("connettivita' per costruzione, quindi va letto sulla tile accuracy.")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = pick_device(cfg.device)

    data = np.load(cfg.data.cache, allow_pickle=True)
    train, val, test = train_val_test_split(
        data["rooms"], cfg.data.val_fraction, cfg.data.test_fraction, seed=cfg.seed
    )
    rooms = {"train": train, "val": val, "test": test}[cfg.eval.split]

    mcfg = model_cfg_of(cfg.eval.ckpt, cfg.model)
    nca = NCA(hidden_channels=mcfg.hidden_channels,
              mlp_hidden=mcfg.mlp_hidden,
              update_prob=mcfg.update_prob,
              use_laplacian=mcfg.use_laplacian).to(device)
    ckpt = torch.load(cfg.eval.ckpt, map_location=device)
    nca.load_state_dict(ckpt["nca"])
    print(f"checkpoint: {cfg.eval.ckpt}  (step {ckpt['step']})  "
          f"hidden={mcfg.hidden_channels} mlp={mcfg.mlp_hidden} "
          f"update_prob={mcfg.update_prob} laplacian={mcfg.use_laplacian}")
    print(f"split '{cfg.eval.split}': {len(rooms)} stanze | device={device} "
          f"| {cfg.eval.steps} passi di riparazione")

    rows = evaluate(nca, rooms, steps=cfg.eval.steps,
                    hidden_channels=mcfg.hidden_channels,
                    seeds=tuple(cfg.eval.seeds), device=device,
                    fractions=tuple(cfg.train.damage_fractions))
    agg = aggregate(rows)
    print_table(agg)

    out = os.path.join(os.path.dirname(cfg.eval.ckpt), f"eval_{cfg.eval.split}.csv")
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print(f"\nsalvato in {out}")


if __name__ == "__main__":
    main()