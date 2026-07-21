# Entry-point di training con Hydra.
# Legge i config da conf/, costruisce dati, modello, pool e trainer, e lancia il
# training. Il device e' rilevato in automatico (cuda se disponibile).
#
# Uso:
#   python scripts/train.py                                       # run singola
#   python scripts/train.py train.smoke_steps=300                 # run corta di prova
#   python scripts/train.py -m model.hidden_channels=8,12,16,24   # ablation (multirun)
#   python scripts/train.py -m model.use_laplacian=false,true
#   python scripts/train.py -m model.mlp_hidden=64,128,256
#
# Nota Kaggle: Hydra crea una cartella di run nuova a ogni invocazione, quindi per
# riprendere una sessione interrotta bisogna puntare alla stessa cartella con
# hydra.run.dir=<percorso fisso su storage persistente>.
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.data.splits import train_val_test_split
from src.data.augment import augment
from src.models.nca import NCA
from src.train.pool import SamplePool
from src.train.trainer import Trainer
from src.models.factory import build_model


def pick_device(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)

    # dati: carico la cache, divido, aumento solo il train
    data = np.load(cfg.data.cache, allow_pickle=True)
    train, val, test = train_val_test_split(
        data["rooms"], cfg.data.val_fraction, cfg.data.test_fraction, seed=cfg.seed
    )
    if cfg.data.augment:
        train = augment(train)
    print(f"device={device} | train {len(train)}  val {len(val)}  test {len(test)}")

    # modello, pool, trainer
    nca = build_model(cfg.model).to(device)
    pool = SamplePool(train, pool_size=cfg.train.pool_size,
                      hidden_channels=cfg.model.hidden_channels,
                      device=device, seed=cfg.seed)
    trainer = Trainer(nca, pool, lr=cfg.train.lr, grad_clip=cfg.train.grad_clip,
                      bptt_min=cfg.train.bptt_min, bptt_max=cfg.train.bptt_max,
                      batch_size=cfg.train.batch_size, damage_prob=cfg.train.damage_prob,
                      damage_fractions=cfg.train.damage_fractions, device=device, seed=cfg.seed)

    # checkpoint e log nella cartella di output di questa run (unica per run e per
    # job del multirun: cosi' le ablation non si sovrascrivono il checkpoint)
    out_dir = HydraConfig.get().runtime.output_dir
    ckpt = os.path.join(out_dir, "last.pt")
    if os.path.exists(ckpt):
        trainer.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"ripreso da checkpoint allo step {trainer.step}")

    steps = cfg.train.smoke_steps if cfg.train.smoke_steps > 0 else cfg.train.steps
    writer = SummaryWriter(log_dir=out_dir)
    while trainer.step < steps:
        loss = trainer.train_step()
        writer.add_scalar("loss/train", loss, trainer.step)
        if cfg.train.log_every and trainer.step % cfg.train.log_every == 0:
            print(f"step {trainer.step}  loss {loss:.4f}")
        if cfg.train.checkpoint_every and trainer.step % cfg.train.checkpoint_every == 0:
            trainer.save(ckpt)

    trainer.save(ckpt)
    writer.close()
    print(f"fine allo step {trainer.step}. checkpoint in {ckpt}")


if __name__ == "__main__":
    main()