import json
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_json_map(json_path):
    # 1. Load the data
    with open(json_path, 'r') as f:
        data = json.load(f)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 2. Extract World Bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    bounds = data.get("bounds", {}).get("extents", [0, 10, 0, 10, 0, 10])
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_zlim(bounds[2], bounds[3])  # Mapping Y to Z (Height)
    ax.set_ylim(bounds[4], bounds[5])  # Mapping Z to Y (Depth)

    # 3. Helper to draw boxes from extents
    def draw_box(extents, color='gray', alpha=0.6):
        x1, x2, y1, y2, z1, z2 = extents
        # Define the 8 vertices of the cube
        # Note: We swap Y and Z here for visual height
        v = np.array([[x1, z1, y1], [x2, z1, y1], [x2, z2, y1], [x1, z2, y1],
                      [x1, z1, y2], [x2, z1, y2], [x2, z2, y2], [x1, z2, y2]])

        # Define the 6 faces (each face is a list of 4 indexes)
        faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                 [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                 [v[0], v[3], v[7], v[4]], [v[1], v[2], 6, v[5]]]  # Typo fix: v[6]

        # Correcting face index for the list above
        faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                 [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                 [v[1], v[2], v[6], v[5]], [v[4], v[7], v[3], v[0]]]

        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))

    # 4. Draw all blocks
    for block in data.get("blocks", []):
        ext = block["extents"]
        clr = block.get("color", [0.7, 0.7, 0.7])  # Default gray
        draw_box(ext, color=clr)

    # 5. Plot Start and Goal (if they exist)
    if "start" in data:
        s = data["start"]
        ax.scatter(s[0], s[2], s[1], color='lime', s=200, label='START', edgecolors='black')

    if "goal" in data:
        g = data["goal"]
        ax.scatter(g[0], g[2], g[1], color='red', s=200, label='GOAL', edgecolors='black', marker='*')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Z Axis (Depth)')
    ax.set_zlabel('Y Axis (Height)')
    plt.legend()
    plt.show()

# To use:
filename = '../util/mymap.json'
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
visualize_json_map(filename)