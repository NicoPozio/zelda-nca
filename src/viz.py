"""
Rendering helpers for the explanatory notebook.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.tiles import CHAR_MAP, IDX_TO_CHAR, NUM_TILES

# One colour per tile, ordered by tile index. Walkable tiles are light, obstacles
# dark, so that traversability is readable at a glance.
TILE_COLOURS = {
    "F": "#e8dcc8",   # floor
    "B": "#8a7a63",   # block
    "M": "#c8705f",   # monster
    "P": "#3d6b8a",   # element 
    "O": "#7fa8c4",   # element + floor 
    "I": "#4a5a6b",   # element + block
    "D": "#d9a441",   # door
    "S": "#9c6bb0",   # stair
    "W": "#3a3226",   # wall
    "-": "#141414",   # void
}
_CMAP = ListedColormap([TILE_COLOURS[IDX_TO_CHAR[i]] for i in range(NUM_TILES)])


def render_room(room, ax=None, title=None, letters=False, highlight=None):
    """Draw a room as a colour grid.

    Args:
        room: (H, W) array of tile indices.
        ax: existing axes, or None to create one.
        title: optional axes title.
        letters: overlay the tile character on each cell.
        highlight: optional boolean mask; marked cells get a red outline.
    """
    room = np.asarray(room)
    if ax is None:
        _, ax = plt.subplots(figsize=(room.shape[1] * 0.22, room.shape[0] * 0.22))
    ax.imshow(room, cmap=_CMAP, vmin=0, vmax=NUM_TILES - 1, interpolation="nearest")

    if letters:
        for (i, j), v in np.ndenumerate(room):
            ax.text(j, i, IDX_TO_CHAR[int(v)], ha="center", va="center",
                    fontsize=5.5, color="white" if v in (8, 9) else "black")

    if highlight is not None:
        for i, j in zip(*np.nonzero(np.asarray(highlight))):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="#e03030", lw=1.4))

    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return ax


def render_mask(mask, ax=None, title=None):
    """Draw a boolean mask as a two-tone grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(mask.shape[1] * 0.22, mask.shape[0] * 0.22))
    ax.imshow(np.asarray(mask), cmap=ListedColormap(["#2b2b2b", "#7dc47d"]),
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return ax


def render_components(room, ax=None, title=None, passable_element_floor=True):
    """Draw connected components of walkable cells, one colour per component."""
    from src.metrics.connectivity import walkable_components
    labels, n = walkable_components(room, passable_element_floor)
    palette = ["#2b2b2b"] + ["#7dc47d", "#d9a441", "#c8705f", "#7fa8c4",
                             "#9c6bb0", "#8a7a63"] * 4
    if ax is None:
        _, ax = plt.subplots(figsize=(labels.shape[1] * 0.22, labels.shape[0] * 0.22))
    ax.imshow(labels, cmap=ListedColormap(palette[:max(n, 1) + 1]),
              vmin=0, vmax=max(n, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(f"{title} ({n} component{'s' if n != 1 else ''})", fontsize=8)
    return ax


def parse_room(rows):
    """Build a room from a list of character strings. Used for hand-made cases."""
    return np.array([[CHAR_MAP[c] for c in r] for r in rows], dtype=np.int64)