#!/usr/bin/env python3
"""
Assignment 3 — Pyramid Building with Alternating Triangles

Run example:
    python triangle_pyramid_optimized.py --side 1.0 --depth 5 --save pyramid_d5.png
"""

# from __future__ import annotations




import argparse
from dataclasses import dataclass
from math import sqrt
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# -----------------------------
# Dataclass for a small triangle
# -----------------------------
@dataclass
class SmallTriangle:
    upright: bool       # True = apex up, False = apex down
    apex_x: float       # x coordinate of apex
    apex_y: float       # y coordinate of apex
    side: float         # side length

    def vertices(self) -> List[Tuple[float, float]]:
        """This function Returns 3 vertices (x,y) for the equilateral triangle.
        For upright: apex is top vertex; for inverted: apex is bottom vertex.
        """
        s = self.side
        h = (sqrt(3) / 2.0) * s
        x = self.apex_x
        y = self.apex_y
        if self.upright:
            return [(x, y), (x - s / 2.0, y - h), (x + s / 2.0, y - h)]
        else:
            return [(x - s / 2.0, y + h), (x + s / 2.0, y + h), (x, y)]

# -----------------------------
# Building pyramid geometry
# -----------------------------
def build_pyramid(side: float, depth: int) -> List[SmallTriangle]:
    """Now here we are Constructing list of triangles for pyramid depth D and side length s.
    """
    assert side > 0 and depth >= 1, "The side must be > 0 and depth >= 1"
    h = (sqrt(3) / 2.0) * side
    tris: List[SmallTriangle] = []

    # vertical shift so bottom base sits at y = 0
    vshift = h

    
    for r in range(1, depth + 1):
        # y coordinate of upright apices in this row
        y_u = (depth - r) * h + vshift

        
        x_u = [ (j - (r + 1) / 2.0) * side for j in range(1, r + 1) ]

        # adding upright triangles
        for xu in x_u:
            tris.append(SmallTriangle(upright=True, apex_x=xu, apex_y=y_u, side=side))

        # adding inverted triangles between adjacent uprights 
        if r >= 2:
            y_inv = y_u - h
            x_inv = [ 0.5 * (x_u[k] + x_u[k + 1]) for k in range(0, len(x_u) - 1) ]
            for xi in x_inv:
                tris.append(SmallTriangle(upright=False, apex_x=xi, apex_y=y_inv, side=side))

    return tris

# -----------------------------
# Plotting and summary
# -----------------------------
def plot_pyramid(triangles: List[SmallTriangle], side: float, depth: int, save_path: str | None = None,
                 upright_color: str = "tab:blue", inverted_color: str = "tab:orange"):
    h = (sqrt(3) / 2.0) * side

    # Compute bounds (tight)
    base_half_width = (depth - 1) * (side / 2.0)
    x_min = -base_half_width - side
    x_max = base_half_width + side
    y_min = 0.0
    y_max = depth * h + h

    fig_w = max(6.0, 0.45 * depth + 4.0)
    fig_h = max(6.0, 0.45 * depth + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # draw triangles
    for t in triangles:
        verts = t.vertices()
        poly = MplPolygon(verts, closed=True,
                          facecolor=(upright_color if t.upright else inverted_color),
                          edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.add_patch(poly)

    ax.set_title(f"Pyramid of Alternating Triangles — depth D={depth}, side s={side}")
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, color="lightgray", linewidth=0.6)

    # Legend proxies
    up_proxy = MplPolygon([[0,0],[0,0],[0,0]], color=upright_color)
    dn_proxy = MplPolygon([[0,0],[0,0],[0,0]], color=inverted_color)
    ax.legend([up_proxy, dn_proxy], ["Upright", "Inverted"], loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Finally Saved figure to: {save_path}")

    plt.show()

def print_summary(side: float, depth: int, triangles: List[SmallTriangle]):

    h = (sqrt(3) / 2.0) * side
    total = len(triangles)
    upright = sum(1 for t in triangles if t.upright)
    inverted = total - upright

    print("\n=== Assignment 3: Pyramid Building with Triangles ===")
    print(f"INPUT : side (s) = {side:.4f}, depth (D) = {depth}")
    print(f"The triangle height h = s * sqrt(3)/2 = {h:.4f}")
    print(f"The expected counts: upright = D(D+1)/2 = {depth*(depth+1)//2}, inverted = D(D-1)/2 = {depth*(depth-1)//2}, total = D^2 = {depth*depth}")
    print("\nThe counts computed from geometry:")
    print(f"  Upright  : {upright}")
    print(f"  Inverted : {inverted}")
    print(f"  TOTAL    : {total}\n")

    # show first few rows structure for clarity
    print("ROW PREVIEW : triangles per row (top->bottom):")
    for r in range(1, min(depth, 6) + 1):
        print(f"  Row {r}: {2*r - 1} triangles (upright={r}, inverted={r-1})")
    if depth > 6:
        print("  ... (remaining rows follow same pattern)")

# -----------------------------
# CLI + main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build and visualize a pyramid of alternating equilateral triangles.")
    p.add_argument("--side", type=float, required=True, help="Side length of each equilateral triangle (s > 0).")
    p.add_argument("--depth", type=int, required=True, help="Depth D (number of levels, D >= 1).")
    p.add_argument("--save", type=str, default=None, help="Optional path to save the PNG plot.")
    return p.parse_args()

def main():
    args = parse_args()
    s = args.side
    D = args.depth
    if s <= 0:
        print("ERROR : --side must be > 0")
        return
    if D < 1:
        print("ERROR : --depth must be >= 1")
        return

    # Build geometry (closed form) — O(D^2)
    triangles = build_pyramid(side=s, depth=D)

    # Structured prints for interview/demo
    print_summary(side=s, depth=D, triangles=triangles)

    # Visualize
    plot_pyramid(triangles, side=s, depth=D, save_path=args.save)

if __name__ == "__main__":
    main()
