# Vectra-Assignment
This assignment combines geometry and visualization tasks: computing polygon properties using vector math, tiling a rectangular room with minimal squares in a spiral pattern, and building a pyramid of alternating upright and inverted equilateral triangles. Each problem emphasizes optimization, structure, and clear plotting.

Geometry & Visualization Assignments

This repository contains solutions to a series of assignments focused on geometry, optimization, and visualization using Python. Each problem highlights structured algorithm design, efficient computation, and clear graphical output with matplotlib.

Assignment Overview
1. Polygon Properties with Geometry

Implemented the Shoelace Formula and 2D cross product to compute polygon areas.

Used vector norms to calculate edge lengths.

Explored shapely.geometry.Polygon for robust geometric operations.

Output: Polygon area, perimeter, and edge details.

2. Room Tiling with Squares (Spiral Fill Visualization)

Room of size M × N is tiled using square tiles of sizes 1×1, 2×2, 3×3, 4×4.

Larger tiles are placed first (optimization rule).

Tiling proceeds in a spiral motion from the center.

Any leftover central gaps are filled with 1×1 tiles.

Output: Matplotlib visualization with colored tiles and counts per tile size.

3. Pyramid Building with Triangles

Built a pyramid structure of equilateral triangles (side length s).

Pyramid has depth D (levels).

Alternating upright and inverted triangles per row, aligned properly.

Output: Matplotlib visualization with clear upright/inverted triangle distinction and summary counts.

Key Features

Optimized algorithms with minimal time complexity.

Structured dataclasses for geometry representation.

Precomputed anchor offsets for efficient tiling checks.

Professional visualization with legends, grids, and summaries.

Easy CLI-based execution with customizable input parameters.
