# Verifica che il progetto sia completo e funzionante prima di spendere GPU.
# Controlla i file attesi, gli import, la cache dei dati e fa una micro-run
# end-to-end di training e valutazione.
#
# Uso:  python scripts/check_setup.py
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")

FILES = [
    "pyproject.toml", "pytest.ini",
    "conf/config.yaml", "conf/model/nca.yaml", "conf/data/zelda.yaml", "conf/train/default.yaml",
    "scripts/prepare_data.py", "scripts/train.py",
    "src/__init__.py", "src/tiles.py",
    "src/data/__init__.py", "src/data/vglc.py", "src/data/symmetry.py",
    "src/data/dedup.py", "src/data/splits.py", "src/data/augment.py",
    "src/models/__init__.py", "src/models/encoding.py", "src/models/nca.py",
    "src/damage/__init__.py", "src/damage/stochastic.py", "src/damage/targeted.py",
    "src/metrics/__init__.py", "src/metrics/connectivity.py",
    "src/train/__init__.py", "src/train/pool.py", "src/train/trainer.py",
    "src/eval/__init__.py", "src/eval/runner.py",
    "tests/test_tiles.py", "tests/test_vglc.py", "tests/test_dedup.py",
    "tests/test_splits.py", "tests/test_augment.py", "tests/test_encoding.py",
    "tests/test_nca.py", "tests/test_pool.py", "tests/test_trainer.py",
    "tests/test_stochastic.py", "tests/test_targeted.py", "tests/test_connectivity.py",
]

MODULES = [
    "src.tiles", "src.data.vglc", "src.data.symmetry", "src.data.dedup",
    "src.data.splits", "src.data.augment", "src.models.encoding", "src.models.nca",
    "src.damage.stochastic", "src.damage.targeted", "src.metrics.connectivity",
    "src.train.pool", "src.train.trainer", "src.eval.runner",
]

ok = True


def check(label, condition, detail=""):
    global ok
    mark = "  ok  " if condition else " MANCA"
    print(f"[{mark}] {label}{('  -> ' + detail) if detail and not condition else ''}")
    if not condition:
        ok = False
    return condition


print("=== file attesi ===")
missing = [f for f in FILES if not os.path.exists(os.path.join(ROOT, f))]
for f in FILES:
    if f in missing:
        check(f, False)
if not missing:
    print(f"[  ok  ] tutti i {len(FILES)} file presenti")
else:
    ok = False

print("\n=== dati grezzi ===")
raw = os.path.join(ROOT, "data/raw")
txt = [f for f in os.listdir(raw) if f.endswith(".txt")] if os.path.isdir(raw) else []
check(f"dungeon .txt in data/raw ({len(txt)} trovati)", len(txt) > 0, "metti i tloz*_1.txt")

print("\n=== import dei moduli ===")
for m in MODULES:
    try:
        importlib.import_module(m)
        print(f"[  ok  ] {m}")
    except Exception as e:
        print(f"[ ERRORE] {m}  -> {type(e).__name__}: {e}")
        ok = False

if not ok:
    print("\nCi sono pezzi mancanti: sistema quelli prima di procedere.")
    sys.exit(1)

print("\n=== cache dei dati ===")
import numpy as np
cache = os.path.join(ROOT, "data/processed/rooms.npz")
if not os.path.exists(cache):
    print("[ MANCA] data/processed/rooms.npz  -> lancia: python scripts/prepare_data.py")
    sys.exit(1)
rooms = np.load(cache, allow_pickle=True)["rooms"]
check(f"stanze in cache: {rooms.shape}", rooms.ndim == 3 and rooms.shape[1:] == (11, 16))
check(f"indici di tile nel range 0-9 ({rooms.min()}-{rooms.max()})", rooms.min() >= 0 and rooms.max() <= 9)

print("\n=== micro-run end-to-end ===")
import torch
from src.data.splits import train_val_test_split
from src.data.augment import augment
from src.models.nca import NCA
from src.train.pool import SamplePool
from src.train.trainer import Trainer
from src.eval.runner import evaluate, aggregate

train, val, test = train_val_test_split(rooms, 0.15, 0.15, seed=42)
train = augment(train)
print(f"[  ok  ] split: train {len(train)} (augmentato)  val {len(val)}  test {len(test)}")

nca = NCA(hidden_channels=12)
n_par = sum(p.numel() for p in nca.parameters() if p.requires_grad)
print(f"[  ok  ] NCA costruito: {n_par} parametri addestrabili")

pool = SamplePool(train, pool_size=32, hidden_channels=12, device="cpu", seed=0)
tr = Trainer(nca, pool, lr=1e-3, grad_clip=1.0, bptt_min=4, bptt_max=6, batch_size=4,
             damage_prob=0.5, damage_fractions=[0.2, 0.4], device="cpu", seed=0)
losses = [tr.train_step() for _ in range(5)]
check(f"5 step di training, loss finite {[round(l, 2) for l in losses]}",
      all(np.isfinite(losses)))

rows = evaluate(nca, test[:4], steps=4, hidden_channels=12, seeds=(1,), device="cpu",
                fractions=(0.4,))
check(f"valutazione: {len(rows)} condizioni di danno misurate", len(rows) > 0)

print(f"\n[  ok  ] device disponibile: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("\nTutto a posto. Puoi lanciare la smoke run:")
print("    python scripts/train.py train.smoke_steps=300")