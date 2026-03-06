import inspect
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Force interactive window

from scipy.spatial.transform import Rotation

from flightsim.animate import animate
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate
from flightsim.world import World

from se3_control import SE3Control
from world_traj import WorldTraj


def main():

    # -------------------------------------------------
    # Load world
    # -------------------------------------------------
    filename = '../util/mymap.json'

    file = (
            Path(inspect.getsourcefile(lambda: 0)).parent.resolve()
            / '..'
            / 'util'
            / filename
    ).resolve()

    world = World.from_file(file)
    start = world.world['start']
    goal = world.world['goal']

    # -------------------------------------------------
    # Create quad + controller
    # -------------------------------------------------
    quadrotor = Quadrotor(quad_params)
    controller = SE3Control(quad_params)

    # -------------------------------------------------
    # Plan trajectory
    # -------------------------------------------------
    print("Planning trajectory...")
    t0 = time.time()
    traj = WorldTraj(world, start, goal)
    print(f"Planning time: {time.time() - t0:.2f} s")

    # -------------------------------------------------
    # Initial state
    # -------------------------------------------------
    initial_state = {
        'x': start,
        'v': (0, 0, 0),
        'q': (0, 0, 0, 1),
        'w': (0, 0, 0)
    }

    # -------------------------------------------------
    # Simulate
    # -------------------------------------------------
    print("Simulating...")
    sim_time, state, control, flat, exit = simulate(
        initial_state,
        quadrotor,
        controller,
        traj,
        t_final=60
    )

    print("Simulation finished.")

    # -------------------------------------------------
    # Animate
    # -------------------------------------------------
    print("Launching animation window...")
    fig = plt.figure("Flight Animation")

    R = Rotation.from_quat(state['q']).as_matrix()

    # Option 1: View live animation
    anim = animate(sim_time, state['x'], R, world=world, filename=None)

    # Option 2: Save to mp4 instead
    # animate(sim_time, state['x'], R, world=world, filename='flight.mp4')

    print("Done.")


if __name__ == "__main__":
    main()