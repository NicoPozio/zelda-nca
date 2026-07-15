# Prepara il dataset di stanze a partire dai dungeon .txt.
# Orchestra la pipeline dati: estrae le stanze, le deduplica e salva il risultato
# in una cache .npz, cosi' il training non deve riparsare i .txt ogni volta.
#
# Uso:
#   python scripts/prepare_data.py
#   python scripts/prepare_data.py --raw-dir data/raw --out data/processed/rooms.npz
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.data.vglc import extract_rooms
from src.data.dedup import deduplicate


def main():
    parser = argparse.ArgumentParser(description="Estrae e deduplica le stanze in una cache .npz")
    parser.add_argument("--raw-dir", default="data/raw", help="cartella con i dungeon .txt")
    parser.add_argument("--out", default="data/processed/rooms.npz", help="file .npz di uscita")
    parser.add_argument("--dedup", default="symmetry", choices=["none", "exact", "symmetry"])
    args = parser.parse_args()

    rooms, sources = extract_rooms(args.raw_dir, to_visual=True)
    uniq, idx = deduplicate(rooms, mode=args.dedup, return_index=True)
    kept_sources = np.array([sources[i] for i in idx])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, rooms=uniq, sources=kept_sources)

    print(f"estratte {len(rooms)} stanze, {len(uniq)} uniche (dedup={args.dedup})")
    print(f"salvate in {args.out}  ->  array {uniq.shape}")


if __name__ == "__main__":
    main()