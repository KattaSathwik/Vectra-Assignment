import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

# ----------------------------
# 1. Input Data: Polygon Vertices
# ----------------------------
vertices = np.array([
    [9.05, 7.76],
    [12.5, 3.0],
    [10.0, 0.0],
    [5.0, 0.0],
    [2.5, 3.0]
])

print("\n--- Lets start Polygon Geometry Analysis ---")
print("The Vertices of the polygon:", vertices)

# Number of vertices
n = len(vertices)

# ----------------------------
# 2. Representing edges as vectors
# ----------------------------
edges = np.roll(vertices, -1, axis=0) - vertices
print("\nThe Edges (as vectors):")
for i, edge in enumerate(edges, start=1):
    print(f"Edge {i}: {edge}")

# ----------------------------
# 3. Computing area using Shoelace Formula
# ----------------------------
x = vertices[:, 0]
y = vertices[:, 1]
area_shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
print("\nNow The Polygon Area (Shoelace Formula):", area_shoelace)

# Now Comparing with Shapely
polygon = Polygon(vertices)
area_shapely = polygon.area
print("The Polygon Area (Shapely):", area_shapely)

# ----------------------------
# 4. Now Computing length of each edge using norms
# ----------------------------
edge_lengths = np.linalg.norm(edges, axis=1)
print("\nThe Edge Lengths:", edge_lengths)

# ----------------------------
# 5. Then Computing interior angles using dot product
# ----------------------------
angles = []
for i in range(n):
    # The Previous and next edge vectors
    prev_vec = vertices[i] - vertices[i-1]
    next_vec = vertices[(i+1) % n] - vertices[i]

    # Now Computing the angle using dot product formula
    dot_product = np.dot(prev_vec, next_vec)
    norm_product = np.linalg.norm(prev_vec) * np.linalg.norm(next_vec)
    cos_theta = dot_product / norm_product
    angle = np.degrees(np.arccos(cos_theta))
    angles.append(angle)

print("\nTherefor, The Interior Angles (degrees):", angles)

# ----------------------------
# 6. Now Checking if polygon is convex (all angles < 180) or concave
# ----------------------------
is_convex = all(angle < 180 for angle in angles)
print("\nIs the polygon convex?:", is_convex)

# ----------------------------
# 7. Then Computing centroid (average of vertices)
# ----------------------------
centroid_calc = np.mean(vertices, axis=0)
centroid_shapely = (polygon.centroid.x, polygon.centroid.y)
print("\nCentroid (Calculated):", centroid_calc)
print("Centroid (Shapely):", centroid_shapely)

# ----------------------------
# 8. Finally Visualization using Matplotlib
# ----------------------------
plt.figure(figsize=(6, 6))
plt.fill(x, y, alpha=0.3)  
plt.plot(x.tolist() + [x[0]], y.tolist() + [y[0]], 'b-')  


for i, (vx, vy) in enumerate(vertices):
    plt.text(vx, vy, f"V{i+1}", fontsize=10, color='blue')


plt.scatter(*centroid_calc, color='red', label='Centroid')
plt.legend()

# Annotating the angles near vertices
for i, (vx, vy) in enumerate(vertices):
    plt.text(vx + 0.2, vy + 0.2, f"{angles[i]:.1f}°", fontsize=8, color='green')

plt.title("Polygon Geometry Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')
plt.show()
