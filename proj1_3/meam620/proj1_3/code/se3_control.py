import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        K_p = np.diag([6.5, 6.5, 12.])
        K_d = np.diag([4., 4., 5.0])

        K_r = np.diag([1000.0, 1000.0, 100.0])
        K_w = np.diag([100.0, 100.0, 20.0])

        e_pos = flat_output['x'] - state['x']
        e_vel = flat_output['x_dot'] - state['v']
        rdes_ddot = flat_output['x_ddot'] + K_p @ e_pos + K_d @ e_vel

        F_des = self.mass * rdes_ddot + np.array([0, 0, self.g*self.mass])
        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = R @ np.array([0, 0, 1])
        u1 = b3 @ F_des
        u1 = np.clip(u1, 0.0, 5 * self.mass * self.g)

        b3_des = F_des / np.linalg.norm(F_des)
        a = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0])
        b2 = np.cross(b3_des, a)
        b2_des = b2 / np.linalg.norm(b2)
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack((b1_des, b2_des, b3_des))
        e_R_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([
            e_R_matrix[2, 1],
            e_R_matrix[0, 2],
            e_R_matrix[1, 0]
        ])

        # F_des_dot = self.mass * flat_output['x_dddot']
        # b3_des_dot = (F_des_dot - (F_des_dot @ b3_des)*b3_des) / np.linalg.norm(F_des)
        # a_dot = np.array([-np.sin(flat_output['yaw']*flat_output['yaw_dot']), np.cos(flat_output['yaw']*flat_output['yaw_dot']), 0])
        # cross_dot = np.cross(b3_des, a_dot) + np.cross(b3_des_dot, a)
        # b2_des_dot = (cross_dot - np.dot(cross_dot, b2_des)*b2_des) / np.linalg.norm(cross_dot)
        # b1_des_dot = np.cross(b2_des_dot, b3_des) + np.cross(b3_des_dot, b2_des)
        #
        # R_des_dot = np.column_stack((b1_des_dot, b2_des_dot, b3_des_dot))
        # omega_hat = R_des.T @ R_des_dot
        # w_des = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
        w_des = np.zeros(3)
        e_w = state['w'] - w_des

        u2 = self.inertia @ (-K_r@e_R -K_w@e_w)

        cmd_thrust = u1
        cmd_moment = u2

        l = self.arm_length
        km = self.k_drag / self.k_thrust

        A = np.array([
            [1, 1, 1, 1],
            [0, l, 0, -l],
            [-l, 0, l, 0],
            [km, -km, km, -km]
        ])

        U = np.array([u1, u2[0], u2[1], u2[2]])
        F = np.linalg.solve(A, U)
        w_sq = F / self.k_thrust
        w_sq = np.clip(w_sq, 0, None)
        omega = np.sqrt(w_sq)

        cmd_thrust = u1
        cmd_moment = u2
        cmd_motor_speeds = np.clip(omega, self.rotor_speed_min, self.rotor_speed_max)
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
