import heapq
import numpy as np

from occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    # Return a tuple (path, nodes_expanded)

    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                delta = (dx, dy, dz)
                cost = np.linalg.norm(np.array([dx, dy, dz]) * resolution)
                directions.append((delta, cost))

    distances = {start_index: 0}
    parents = {start_index: None}
    visited = set()
    nodes_expanded = 0

    if astar:
        heuristic = np.linalg.norm((np.array(start_index) - np.array(goal_index)) * resolution)
        queue = [(heuristic, start_index)]
    else:
        queue = [(0, start_index)]

    while queue:
        current_f, current_index = heapq.heappop(queue)

        if current_index in visited:
            continue
        visited.add(current_index)
        nodes_expanded += 1

        if current_index == goal_index:
            break

        curr_dist = distances[current_index]

        for delta, cost in directions:
            neighbor = (current_index[0] + delta[0],
                        current_index[1] + delta[1],
                        current_index[2] + delta[2])

            if occ_map.is_valid_index(neighbor) and not occ_map.is_occupied_index(neighbor):
                new_dist = curr_dist + cost

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    parents[neighbor] = current_index

                    if astar:
                        heuristic = np.linalg.norm((np.array(neighbor) - np.array(goal_index)) * resolution)
                        heapq.heappush(queue, (new_dist + heuristic, neighbor))
                    else: heapq.heappush(queue, (new_dist, neighbor))

    path = []
    if goal_index not in parents:
        return None, nodes_expanded

    node = goal_index
    while node is not None:
        path.append(occ_map.index_to_metric_center(node))
        node = parents[node]

    path.reverse()

    path_metric = np.array(path)
    path_metric[0] = start
    path_metric[-1] = goal

    return path_metric, nodes_expanded
