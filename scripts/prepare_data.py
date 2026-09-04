# Prepara il dataset di stanze a partire dai file .txt

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

    #Estrazione stanze e lista di percorsi room:i,j
    rooms, sources = extract_rooms(args.raw_dir, to_visual=True)
    #Deduplica le stanze, restituisce anche inidici delle stanze uniche
    uniq, idx = deduplicate(rooms, mode=args.dedup, return_index=True)
    #Crea un array numpy con la lista dei percorsi
    kept_sources = np.array([sources[i] for i in idx])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, rooms=uniq, sources=kept_sources)

    print(f"estratte {len(rooms)} stanze, {len(uniq)} uniche (dedup={args.dedup})")
    print(f"salvate in {args.out}  ->  array {uniq.shape}")


if __name__ == "__main__":
    main()