"""
Assignment 2 — Room Tiling with Squares

    python room_tiling_very_optimized.py --width 37 --height 21 --save out.png

"""

# from __future__ import annotations





import argparse
from dataclasses import dataclass
import time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# color mapping
# ---------------------------------------------------------------------------
SIZE_COLOR = {1: "red", 2: "blue", 3: "yellow", 4: "green"}
TILE_SIZES_DESC = (4, 3, 2, 1)

# ---------------------------------------------------------------------------
# This is Data class to record tile placements
# ---------------------------------------------------------------------------
@dataclass
class Tile:
    tile_id: int
    size: int
    row: int  
    col: int  

# ---------------------------------------------------------------------------
# Now lets start with the Spiral generator 
# ---------------------------------------------------------------------------

def spiral_indices(h: int, w: int):
    r = h // 2
    c = w // 2
    yield (r, c)
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    step = 1
    while True:
        yielded = False
        for d in range(4):
            dr, dc = dirs[d]
            for _ in range(step):
                r += dr
                c += dc
                if 0 <= r < h and 0 <= c < w:
                    yielded = True
                    yield (r, c)
            if d % 2 == 1:
                step += 1
        if not yielded:
            break


ANCHOR_OFFSETS = {}
for s in TILE_SIZES_DESC:
    offsets = [(k, l) for k in range(s) for l in range(s)]
    center_k = s // 2
    center_l = s // 2
    offsets.sort(key=lambda t: (abs(t[0] - center_k) + abs(t[1] - center_l), t[0], t[1]))
    ANCHOR_OFFSETS[s] = offsets

# ---------------------------------------------------------------------------
# Low-level placement checks (fast)
# ---------------------------------------------------------------------------

def can_place(grid: np.ndarray, r0: int, c0: int, s: int) -> bool:
    """Here we are Returning True if s x s square at top-left (r0,c0) is within bounds and empty.
    """
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + s > h or c0 + s > w:
        return False

    sub = grid[r0 : r0 + s, c0 : c0 + s]
    return np.all(sub == 0)


def place_tile(grid: np.ndarray, r0: int, c0: int, s: int, tile_id: int) -> Tile:
    grid[r0 : r0 + s, c0 : c0 + s] = tile_id
    return Tile(tile_id=tile_id, size=s, row=r0, col=c0)

# ---------------------------------------------------------------------------
# Optimized spiral greedy tiling algorithm
# ---------------------------------------------------------------------------

def spiral_tiling_optimized(width: int, height: int) -> Tuple[np.ndarray, List[Tile], float]:
    """
    This Returns the occupancy grid, list of Tile objects, and runtime.
    """
    t0 = time.perf_counter()
    grid = np.zeros((height, width), dtype=np.int32)
    tiles: List[Tile] = []
    next_id = 1

    total_cells = height * width
    filled = 0

    print("[Lets Start optimized spiral tiling ")
    print(f"       Room: {width} x {height} (W x H) — total cells: {total_cells})")

    for (r, c) in spiral_indices(height, width):
        if filled >= total_cells:
            break
        if grid[r, c] != 0:
            continue  

        
        placed_for_cell = False
        for s in TILE_SIZES_DESC:
            for (k, l) in ANCHOR_OFFSETS[s]:
                r0 = r - k
                c0 = c - l
                # The bounds check: r0 between 0 and height-s, c0 between 0 and width-s
                if r0 < 0 or c0 < 0 or r0 + s > height or c0 + s > width:
                    continue
                if can_place(grid, r0, c0, s):
                    tiles.append(place_tile(grid, r0, c0, s, next_id))
                    next_id += 1
                    filled += s * s
                    placed_for_cell = True
                    
                    print(f"The {s}x{s} tile id={next_id-1} at top-left (r={r0},c={c0}) covering (r={r},c={c}); filled={filled}/{total_cells}")
                    break
            if placed_for_cell:
                break
        

    
    remaining = np.argwhere(grid == 0)
    if remaining.size > 0:
        print(f"Then in Final pass: filling {len(remaining)} remaining cells with 1x1 tiles")
        for (r0, c0) in remaining:
            tiles.append(place_tile(grid, int(r0), int(c0), 1, next_id))
            next_id += 1
            filled += 1

    t1 = time.perf_counter()
    runtime = t1 - t0

    print("Finally Tiling finished.")
    counts = {s: 0 for s in TILE_SIZES_DESC}
    for t in tiles:
        counts[t.size] += 1
    for s in TILE_SIZES_DESC:
        print(f"   - {s}x{s}: {counts[s]}")
    print(f"Therefore TOTAL tiles: {len(tiles)} | Filled cells: {filled}/{total_cells} | time: {runtime:.4f}s")

    return grid, tiles, runtime

# ---------------------------------------------------------------------------
# Visualization with matplotlib
# ---------------------------------------------------------------------------

def visualize(grid: np.ndarray, tiles: List[Tile], width: int, height: int, save: str | None = None) -> None:
    print("Rendering plot...")
    fig_w = max(6, width * 0.25)
    fig_h = max(6, height * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for t in tiles:
        x = t.col
        y = height - (t.row + t.size)  
        rect = Rectangle((x, y), t.size, t.size,
                         facecolor=SIZE_COLOR[t.size], edgecolor='black', linewidth=0.6, alpha=0.9)
        ax.add_patch(rect)
        
        ax.text(x + t.size / 2, y + t.size / 2, f"{t.size}×{t.size}", ha='center', va='center', fontsize=6)

    #The Room boundary and grid
    ax.add_patch(Rectangle((0, 0), width, height, facecolor='none', edgecolor='black', linewidth=1.2))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, width + 1))
    ax.set_yticks(range(0, height + 1))
    ax.grid(True, color='lightgray', linewidth=0.5)

    # Legend: counts per size
    counts = {s: 0 for s in TILE_SIZES_DESC}
    for t in tiles:
        counts[t.size] += 1
    handles = []
    labels = []
    for s in TILE_SIZES_DESC:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=SIZE_COLOR[s]))
        labels.append(f"{s}×{s}: {counts[s]}")
    ax.legend(handles, labels, loc='upper right')

    ax.set_title(f"The Optimized Spiral Tiling — {width}×{height}")
    ax.set_xlabel('X (width)')
    ax.set_ylabel('Y (height)')

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"The plot is Saved to {save}")

    plt.show()
    print("The Visualization Done.")

# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Very-optimized spiral room tiling with 4/3/2/1 squares')
    p.add_argument('--width', type=int, required=True, help='Room width (columns)')
    p.add_argument('--height', type=int, required=True, help='Room height (rows)')
    p.add_argument('--save', type=str, default=None, help='Optional path to save PNG plot')
    return p.parse_args()


def main():
    args = parse_args()
    W = args.width
    H = args.height
    if W <= 0 or H <= 0:
        print('ERROR : width and height must be positive integers')
        return

    print('\n=== Assignment 2: Room Tiling  ===')
    print('The Tile sizes: 4×4 (green), 3×3 (yellow), 2×2 (blue), 1×1 (red)')
    print('The Algorithm: center-out spiral; for each unfilled cell try biggest tiles first; prefer centered anchors')

    grid, tiles, runtime = spiral_tiling_optimized(W, H)

    
    counts = {s: 0 for s in TILE_SIZES_DESC}
    for t in tiles:
        counts[t.size] += 1
    print('\n Final tile counts:')
    for s in TILE_SIZES_DESC:
        print(f'   {s}×{s}: {counts[s]}')
    print(f'   TOTAL tiles: {len(tiles)} | Runtime: {runtime:.4f}s')

    visualize(grid, tiles, W, H, save=args.save)


if __name__ == '__main__':
    main()
