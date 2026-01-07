from __future__ import annotations

import numpy as np

from leapfrog.nbody import (
    NBodySystem,
    integrate_leapfrog,
    shift_system_to_com_frame,
    trajectory_to_com_frame,
)
from leapfrog.viz import animate_trajectories_3d, show_animation

# -----------------------------------------------------------------------------
# Pick which demo you want to run (PyCharm friendly - just press Run).
# -----------------------------------------------------------------------------
DEMO = "random"  # options: "figure8", "random"

# Simulation parameters (dimensionless for these demos - we use grav_const=1)
DT = 0.005
T_STOP = 20.0

# For random demo
RANDOM_N = 4
RANDOM_SEED = 0


def make_figure8_system() -> NBodySystem:
    """3-body figure-eight choreography (equal masses, planar).

    This is a standard set of initial conditions commonly used for the 3-body
    figure-eight orbit. Units here are dimensionless, and we set grav_const=1.

    Important property: total momentum is (approximately) zero so COM stays near
    the origin. We still shift to COM frame explicitly for robustness.
    """
    m = np.array([1.0, 1.0, 1.0], dtype=float)

    r = np.array(
        [
            [0.97000436, -0.24308753, 0.0],
            [-0.97000436, 0.24308753, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    v = np.array(
        [
            [0.466203685, 0.43236573, 0.0],
            [0.466203685, 0.43236573, 0.0],
            [-0.93240737, -0.86473146, 0.0],
        ],
        dtype=float,
    )

    return NBodySystem(m=m, r=r, v=v, grav_const=1.0, softening=0.0)


def make_random_system(n: int, *, seed: int) -> NBodySystem:
    """Random N-body system (no central mass).

    This is purely a "many bodies flying around" demo.
    We shift to COM frame so the plot stays centered.
    """
    rng = np.random.default_rng(seed)

    m = rng.uniform(0.5, 2.0, size=n)
    r = rng.normal(0.0, 1.0, size=(n, 3))
    v = rng.normal(0.0, 0.3, size=(n, 3))

    system = NBodySystem(m=m, r=r, v=v, grav_const=1.0, softening=1e-3)

    # Shift initial conditions so COM position and COM velocity are zero.
    shift_system_to_com_frame(system)
    return system


def main() -> None:
    # -------------------------------------------------------------------------
    # Build system
    # -------------------------------------------------------------------------
    if DEMO == "figure8":
        system = make_figure8_system()
        # explicit COM shift to remove any rounding-induced drift
        shift_system_to_com_frame(system)
        title = "3-body figure-eight (COM frame)"
    elif DEMO == "random":
        system = make_random_system(RANDOM_N, seed=RANDOM_SEED)
        title = f"Random N-body (N={RANDOM_N}, COM frame)"
    else:
        msg = f"Unknown DEMO={DEMO!r}"
        raise ValueError(msg)

    # -------------------------------------------------------------------------
    # Integrate
    # -------------------------------------------------------------------------
    _, traj = integrate_leapfrog(system, dt=DT, t_stop=T_STOP, record=True)

    # Convert recorded trajectory into COM coordinates per frame (robust centering)
    traj_com = trajectory_to_com_frame(traj, system.m)

    # -------------------------------------------------------------------------
    # Visualize
    # -------------------------------------------------------------------------
    handles = animate_trajectories_3d(
        traj_com,
        interval_ms=10,
        trail=True,
        title=title,
    )
    show_animation(handles)


if __name__ == "__main__":
    main()
