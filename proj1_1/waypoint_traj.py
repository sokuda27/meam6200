import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self.speed = 2.5
        self.points = points

        n_points = len(points)
        start_times = np.zeros(n_points)
        start_times[0] = 0
        for i in range(n_points - 1):
            distance = np.linalg.norm(self.points[i+1] - self.points[i])
            start_times[i+1] = start_times[i] + distance/self.speed
        self.times = start_times

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
        start_times = self.times
        if t >= self.times[-1]:
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
        for i in range(len(self.points) - 1):
            if start_times[i] <= t < start_times[i + 1]:
                curr_seg = i
                break

        t0 = start_times[curr_seg]
        segment_total = start_times[curr_seg + 1] - start_times[curr_seg]
        dt = t - t0

        start_pos = self.points[curr_seg]
        end_pos = self.points[curr_seg + 1]
        diff = end_pos - start_pos
        vect = diff / np.linalg.norm(diff)
        x_dot = self.speed*vect
        x = start_pos + x_dot*dt

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
