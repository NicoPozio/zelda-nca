# Topology-Preserving Level Repair with Neural Cellular Automata

Project for the course **Deep Learning & Applied AI (DLAI)**, a.y. 2025/26 — Sapienza University of Rome, Prof. Emanuele Rodolà.

This project asks whether a purely local update rule, trained only to reconstruct real rooms from the dungeons of the first *The Legend of Zelda*, can restore the **playability** of a damaged room rather than merely its appearance. Repair is evaluated with a pathfinding criterion instead of tile accuracy, and connectivity never appears in the cost function.

---

## Research Question

To what extent does a Neural Cellular Automaton with purely local rules restore the topological connectivity between a room's accesses and its main walkable area, and how does the outcome depend on the **target** of the damage rather than its **extent**?

---

## Models

Six models, each motivated by a limitation of the previous one:

**No-damage NCA.** The null hypothesis. Trained without ever seeing a corrupted room, so its only task is persistence. Establishes whether regenerative behaviour emerges from reconstruction alone.

**NCA.** The main model. 22-channel cell state (10 one-hot tiles, 12 hidden), depthwise perception with three fixed filters (identity and the two Sobel operators), a two-layer 1×1 MLP producing a residual increment under a Bernoulli(0.5) per-cell mask. 11,414 parameters. Trained with stochastic damage on half of each batch.

**Iterative U-Net.** Global receptive field, applied sixteen times. Isolates **locality**, since it shares the pool, the loss and the metric with the NCA and differs only in what each cell can see.

**One-shot U-Net.** The same network applied once. Isolates **iterativity**, since the NCA is both local and iterative and a single comparison would move two variables at once. 473,974 parameters, which also bound what capacity alone could achieve.

**BFS NCA.** One hidden channel supervised towards the geodesic distance to the nearest access, computed on the pristine room. Same architecture and parameter count as the NCA: the loss term is the only difference. Asks whether the bottleneck is representing the topology or acting on it.

**Global-channel NCA.** One hidden channel is replaced, after every step, by its spatial mean and redistributed to every cell. A minimal relaxation of locality: each cell still reads only its neighbourhood, but also receives a scalar summary of the global state, at the same parameter count.

---

## Project Structure

```
zelda-nca/
├── src/
│   ├── tiles.py                     # Alphabet, geometry, contextual walkability
│   ├── viz.py                       # Room, mask and component rendering
│   ├── data/
│   │   ├── vglc.py                  # Dungeon parsing and room slicing
│   │   ├── symmetry.py              # The four shape-preserving symmetries
│   │   ├── dedup.py                 # Deduplication up to symmetry
│   │   ├── splits.py                # Room-level train/val/test split
│   │   └── augment.py               # Training-set augmentation
│   ├── models/
│   │   ├── encoding.py              # State construction and decoding
│   │   ├── nca.py                   # The cellular rule
│   │   ├── unet.py                  # The global comparison model
│   │   ├── factory.py               # Architecture selection from config
│   │   └── aux_targets.py           # BFS distance field
│   ├── damage/
│   │   ├── stochastic.py            # A1 erasure, A2 tile flip (training)
│   │   └── targeted.py              # B1-B4 (evaluation only)
│   ├── metrics/
│   │   └── connectivity.py          # Partition signature, RSR
│   ├── train/
│   │   ├── pool.py                  # Sample pool of persistent states
│   │   └── trainer.py               # Training loop with BPTT
│   └── eval/
│       └── runner.py                # Damage suite and aggregation
├── conf/
│   ├── config.yaml                  # Hydra composition, seed, device
│   ├── model/                       # nca.yaml, unet.yaml
│   ├── data/zelda.yaml              # Cache, split fractions, augmentation
│   ├── train/default.yaml           # Training hyperparameters
│   └── eval/default.yaml            # Checkpoint, split, steps, damage extents
├── scripts/
│   ├── prepare_data.py              # Corpus to cached room array
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Evaluation entry point
│   └── check_setup.py               # Verifies files, imports and a micro-run
├── notebooks/
│   └── Zelda_NCA.ipynb              # Walkthrough of the whole project
├── tests/                           # 68 automated tests
├── data/raw/                        # The 9 VGLC dungeon files
└── report/report.pdf
```

> The notebook clones the repository, loads the published checkpoints and runs the evaluations. All logic lives in `src/`; the notebook only calls it.

> If GitHub fails to render the notebook, view it via [nbviewer](https://nbviewer.org/github/NicoPozio/zelda-nca/blob/main/notebooks/Zelda_NCA.ipynb).

---

## Setup

```bash
git clone https://github.com/NicoPozio/zelda-nca.git
cd zelda-nca
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/check_setup.py
```

`prepare_data.py` extracts 236 rooms from the corpus and caches the 136 that are unique up to symmetry. `check_setup.py` verifies the installation with a micro-run of training and evaluation.

Training requires a GPU; all experiments ran on a free Kaggle T4. A local CPU is sufficient for evaluation and analysis.

---

## Reproducing the Experiments

Every model is a configuration of the same entry point. A single run is 8000 iterations, roughly 18 minutes on a T4.

```bash
# NCA, the main model
python scripts/train.py seed=1

# No-damage NCA, the null hypothesis
python scripts/train.py seed=1 train.damage_prob=0

# One-shot U-Net
python scripts/train.py seed=1 model=unet model.iterative=false \
       train.bptt_min=1 train.bptt_max=1

# Iterative U-Net
python scripts/train.py seed=1 model=unet train.bptt_min=12 train.bptt_max=16

# BFS-supervised NCA
python scripts/train.py seed=1 train.aux_weight=1.0

# Global-channel NCA
python scripts/train.py seed=1 model.global_channels=1
```

Ablations vary a single hyperparameter around the main model:

```bash
python scripts/train.py -m seed=1,2,3 model.hidden_channels=8,16,24
python scripts/train.py -m seed=1,2,3 train.bptt_max=64,128
python scripts/train.py -m seed=1,2,3 model.update_prob=0.3,0.7
python scripts/train.py -m seed=1,2,3 model.mlp_hidden=64,256
python scripts/train.py -m seed=1,2,3 model.use_laplacian=true
```

Evaluation reads the model configuration from the checkpoint's own saved config, so the overrides do not have to be repeated:

```bash
# Ablations are reported on validation
python scripts/evaluate.py eval.ckpt=runs/<name>/last.pt eval.split=val

# Final comparisons on the held-out test split
python scripts/evaluate.py eval.ckpt=runs/<name>/last.pt eval.split=test
```

The inference regime must match the training one: `eval.steps=1` for the one-shot U-Net, `eval.steps=16` for the iterative one, and the default 96 for the cellular models.

---

## Pre-trained Weights

All 77 runs of the study are published as a Kaggle dataset. Attaching it removes the need to retrain: each run directory holds `last.pt`, the Hydra config that produced it, and the evaluation CSV files for both splits.

**Kaggle**: attach `niccolopozio/zelda-nca` via *Add Input*. The notebook detects it automatically and copies the runs into the working directory.

| Run prefix | Description | Seeds |
| :--- | :--- | :--- |
| `m2_*` | NCA, the main model | 8 |
| `m1_null_*` | No-damage NCA | 3 |
| `unet_oneshot_*` | One-shot U-Net | 3 |
| `unet_iter16_*` | Iterative U-Net | 3 |
| `aux10_*` | BFS-supervised NCA | 8 |
| `glob1_*` | Global-channel NCA | 8 |
| `h*`, `bp*`, `up*`, `mlp*`, `lap_*` | Ablations | 3 (8 for the Laplacian) |

---

## Results

Regeneration success rate on the held-out test split, averaged over training seeds. A0 applies no damage and sets the ceiling; A3, B1 and B3 destroy the **same number of cells** and differ only in which cells.

| Model | A0 none | A3 random | B1 access | B3 isolation |
| :--- | :--- | :--- | :--- | :--- |
| No-damage NCA | 0.967 | 0.328 | 0.122 | 0.189 |
| NCA | 0.973 | 0.482 | 0.100 | **0.675** |
| Iterative U-Net | 0.950 | **0.817** | 0.156 | **0.928** |
| One-shot U-Net | **1.000** | 0.150 | **0.506** | 0.222 |
| BFS NCA | 0.917 | 0.578 | 0.152 | 0.783 |
| Global-channel NCA | 0.971 | 0.268 | 0.273 | 0.477 |

**The target, not the extent.** At a matched number of destroyed cells the success rate varies **6.7×**: destroying walkable cells next to an access preserves the topology in 67.5% of rooms, random walkable cells in 48.2%, and access cells in 10.0%. The ordering follows how far the local neighbourhood agrees with the correct answer — an access is surrounded by walls, so a local rule writes a wall there.

**Fidelity is blind to function.** On access damage the repaired rooms are 98.9% correct tile-wise while only 10% remain playable. Across the whole damage suite tile accuracy varies by about one percentage point and the success rate by almost an order of magnitude.

**Degradation.** Under stochastic erasure the success rate falls from 0.358 when 5% of the room is destroyed to 0.031 at 20%, and approaches zero beyond that. The tile flip is considerably harder than the erasure at the same extent (0.035 against 0.358 at 5%): writing a wrong tile is worse than leaving a hole.

**Locality is the bottleneck.** Hidden channels (8 to 24), unroll length (64 to 128), update probability, MLP width, a Laplacian perception filter and explicit BFS supervision all leave access repair unchanged, while a single global pass repairs accesses **5.1×** better. No architecture dominates, though: on the locally consistent conditions the ordering reverses and the one-shot model is the worst of the three.

**A scalar summary recovers part of the gap.** Averaging one hidden channel over the grid raises access repair **2.7×** at the same parameter count, more than the iterative U-Net reaches with forty times the parameters, at a cost on the other conditions.

---

## Metric

The corpus annotates geometry but not affordances: bombable walls, pushable blocks and items that make water traversable are not marked, so an absolute reachability criterion fails on rooms that are perfectly playable in the game. The criterion is therefore defined **relative to the pristine room**, on two conditions:

1. the accesses of the repaired room are exactly those of the original;
2. the probe cells fall into the same connected components.

Probes are the main walkable component together with every access, including accesses that lie outside it. Both conditions are necessary: an access turned into floor leaves the area connected but the room without an exit.

The criterion is validated against rooms built by hand where the expected verdict is known, including a spiral of blocks around a central stair — a layout whose stair is unreachable by a naive search even in the pristine room. Under the relative criterion the pristine room passes, and a model that opens the wall and connects the stair fails, so it cannot game the metric by improving the topology.

---

## Reproducibility

Every run fixes its seed for weight initialisation, pool sampling, damage placement and unroll length. Inference is stochastic as well, since the per-cell update mask is drawn at every step, so evaluation fixes that seed too: without it, three evaluations of the same checkpoint gave access-repair values of 0.133, 0.083 and 0.150.

Configurations use three seeds, and eight paired seeds for the comparisons the conclusions rest on. All reported deviations are over training seeds, that is over independently trained models. Ablations are reported on validation and the final comparisons on the held-out test split, so that no hyperparameter was chosen by looking at the numbers in the report.

The test suite (`pytest`) covers 68 cases, including the properties the argument depends on: that a single step changes nothing outside the 3×3 neighbourhood, that the Sobel filters give exactly zero on a constant region, and that a room reconstructed to 95% of its tiles can still fail the topological criterion.

---

## References

- Mordvintsev, Randazzo, Niklasson & Levin. *Growing Neural Cellular Automata.* Distill, 2020.
- Niklasson, Mordvintsev, Randazzo & Levin. *Self-Organising Textures.* Distill, 2021.
- Randazzo, Mordvintsev, Niklasson, Levin & Greydanus. *Self-Classifying MNIST Digits.* Distill, 2020.
- Palm, González-Duque, Sudhakaran & Risi. *Variational Neural Cellular Automata.* ICLR 2022.
- Sudhakaran et al. *Growing 3D Artefacts and Functional Machines with Neural Cellular Automata.* ALIFE 2021.
- Earle, Snider, Fontaine, Nikolaidis & Togelius. *Illuminating Diverse Neural Cellular Automata for Level Generation.* GECCO 2022.
- Earle, Yildiz, Togelius & Hegde. *Pathfinding Neural Cellular Automata.* arXiv:2301.06820, 2023.
- Siper, Khalifa & Togelius. *Path of Destruction.* IEEE SSCI 2022.
- Summerville, Snodgrass, Mateas & Ontañón. *The VGLC: The Video Game Level Corpus.* Workshop on PCG, 2016.
- Summerville, Behrooz, Mateas & Jhala. *The Learning of Zelda: Data-Driven Learning of Level Topology.* FDG PCG Workshop, 2015.
- Summerville et al. *Procedural Content Generation via Machine Learning.* IEEE Trans. Games 10(3), 2018.
- Johnson, Yannakakis & Togelius. *Cellular Automata for Real-Time Generation of Infinite Cave Levels.* Workshop on PCG, 2010.
- Jain, Isaksen, Holmgård & Togelius. *Autoencoders for Level Generation, Repair, and Recognition.* ICCC 2016.
- Ronneberger, Fischer & Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
- Otte, Delfosse, Czech & Kersting. *Generative Adversarial Neural Cellular Automata.* arXiv:2108.04328, 2021.

Level data from the [Video Game Level Corpus](https://github.com/TheVGLC/TheVGLC).

---

## AI Usage

In accordance with course guidelines, Claude was used as a coding and writing assistant to: (i) write draft code from my specifications, which I then reviewed, tested and modified; (ii) debug the training and evaluation pipeline; (iii) produce the explanatory notebook; and (iv) rework the report manuscript and this README for English and concision.

The decisions that shape the project are mine: the relative topological criterion, the matched-extent control, the damage taxonomy, the domain rules the metric depends on — such as which tiles count as traversable in which context — and which ablations to run. Every number reported here was produced by runs I executed on Kaggle and checked against the released checkpoints. I understand the codebase and take responsibility for the submitted work.

---

## Author

**Niccolò Pozio** — Computer Science Student, Sapienza University of Rome
`pozio.2085512@studenti.uniroma1.it`
