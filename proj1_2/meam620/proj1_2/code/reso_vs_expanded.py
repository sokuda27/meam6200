import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

from flightsim.axes3ds import Axes3Ds
from flightsim.world import World

from occupancy_map import OccupancyMap
from graph_search import graph_search

# Choose a test example file. You should write your own example files too!
# filename = 'test_empty.json'
filename = 'test_plate.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
resolution = world.world['resolution'] # (x,y,z) resolution of discretization, shape=(3,).
margin = world.world['margin']         # Scalar spherical robot size or safety margin.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

resolutions = [0.5, 0.3, 0.2, 0.1, 0.08]

astar_nodes = []
dijkstra_nodes = []

for r in resolutions:
    res = (r, r, r)

    _, nodes_a = graph_search(world, res, margin, start, goal, astar=True)
    _, nodes_d = graph_search(world, res, margin, start, goal, astar=False)

    astar_nodes.append(nodes_a)
    dijkstra_nodes.append(nodes_d)

plt.figure()
plt.plot(resolutions, astar_nodes, marker='o', label='A*')
plt.plot(resolutions, dijkstra_nodes, marker='o', label='Dijkstra')

plt.xlabel("Resolution (m)")
plt.ylabel("Nodes Expanded")
plt.title("Resolution vs Nodes Expanded")
plt.gca().invert_xaxis()  # optional (smaller res = more nodes)
plt.legend()
plt.grid(True)
plt.show()