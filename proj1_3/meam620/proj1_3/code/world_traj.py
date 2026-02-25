import numpy as np
from unicodedata import normalize

from graph_search import graph_search

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.
        # STUDENT CODE HERE
        self.speed = 2.5


    def normalize_grid_direction(v):
    # For voxel grids, reduce vector to step direction (-1,0,1)
        return (
        0 if v[0] == 0 else int(v[0] / abs(v[0])),
        0 if v[1] == 0 else int(v[1] / abs(v[1])),
        0 if v[2] == 0 else int(v[2] / abs(v[2]))
    )

    def clean_collinear(self, path):
        if len(path) < 3:
            return path

        cleaned = [path[0]]
        for i in range(1, len(path) - 1):
            A = cleaned[-1]
            B = path[i]
            C = path[i + 1]
            dir1 = self.normalize_grid_direction(B - A)
            dir2 = self.normalize_grid_direction(C - B)
            if dir1 != dir2:
                cleaned.append(B)
        cleaned.append(path[-1])
        return np.array(cleaned)

    # implement LOS checker?

    def quintic_3d(self, p0, v0, a0, pf, vf, af, T):

        M = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]
        ])

        coeffs = []

        for dim in range(3):
            b = np.array([
                p0[dim],
                v0[dim],
                a0[dim],
                pf[dim],
                vf[dim],
                af[dim]
            ])
            a = np.linalg.solve(M, b)
            coeffs.append(a)

        return np.array(coeffs)

    def eval_quintic_3d(self, coeffs, t):
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        x = np.zeros(3)
        x_dot = np.zeros(3)
        x_ddot = np.zeros(3)
        x_dddot = np.zeros(3)
        x_ddddot = np.zeros(3)

        for d in range(3):
            a0, a1, a2, a3, a4, a5 = coeffs[d]

            x[d] = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4+ a5 * t5
            x_dot[d] = a1 + 2 * a2 * t + 3 * a3 * t2 + 4 * a4 * t3+ 5 * a5 * t4
            x_ddot[d] = 2 * a2 + 6 * a3 * t + 12 * a4 * t2 + 20 * a5 * t3
            x_dddot[d] = 6 * a3 + 24 * a4 * t + 60 * a5 * t2
            x_ddddot[d] = 24 * a4 + 120 * a5 * t

        return x, x_dot, x_ddot, x_dddot, x_ddddot

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        cleaned_path = self.clean_collinear(self.path)

        n_points = len(cleaned_path)
        start_times = np.zeros(n_points)
        seg_times = np.zeros(n_points-1)
        start_times[0] = 0
        for i in range(n_points - 1):
            distance = np.linalg.norm(cleaned_path[i+1] - cleaned_path[i])
            seg_times[i] = distance/self.speed
            start_times[i+1] = start_times[i] + seg_times[i]

        if t >= start_times[-1]:
            x = self.points[-1]
            return {
                'x': x,
                'x_dot': np.zeros(3),
                'x_ddot': np.zeros(3),
                'x_dddot': np.zeros(3),
                'x_ddddot': np.zeros(3),
                'yaw': 0,
                'yaw_dot': 0
            }

        curr_seg = 0
        for i in range(len(cleaned_path) - 1):
            if start_times[i] <= t < start_times[i + 1]:
                curr_seg = i
                break

        t0 = start_times[curr_seg]
        dt = t - t0

        # conditions
        vi = []
        vj = []
        a = np.zeros(3)
        if curr_seg == 0:
            vi.append(cleaned_path[curr_seg]/seg_times[curr_seg])
            pi = cleaned_path[curr_seg]
            vj.append((cleaned_path[curr_seg+2]-cleaned_path[curr_seg])/ (seg_times[curr_seg]+seg_times[curr_seg+1]))
            pj = cleaned_path[curr_seg+1]
        else:
            vi = vj[-1]
            pi = cleaned_path[curr_seg]
            vj.append((cleaned_path[curr_seg+2]-cleaned_path[curr_seg]) / (seg_times[curr_seg]+seg_times[curr_seg+1]))
            pj = cleaned_path[curr_seg+1]

        coeffs = self.quintic_3d(pi, vi, a, pj, vj, a, seg_times[curr_seg])

        x, x_dot, x_ddot, x_dddot, x_ddddot = self.eval_quintic_3d(coeffs, dt)

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
