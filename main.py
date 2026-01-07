from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from leapfrog.nbody import NBodySystem, integrate
from leapfrog.viz import animate_trajectories_3d, show_animation


@dataclass(slots=True)
class RandomSystemConfig:
    """Configuration class."""

    n_objects: int = 3
    central_mass: float = 3.0
    min_radius_au: float = 7.0
    max_radius_au: float = 15.0
    min_vy: float = 0.005
    max_vy: float = 0.009
    min_vz: float = 0.0009
    max_vz: float = 0.002
    min_mass: float = 0.0001
    max_mass: float = 0.0003
    seed: int | None = None


def build_random_system(cfg: RandomSystemConfig) -> NBodySystem:
    # Your old code used random.seed(time.time()) repeatedly :contentReference[oaicite:9]{index=9},
    # which makes reproducibility hard and can produce weird correlations.
    # Here: one RNG, optional seed.
    seed = cfg.seed if cfg.seed is not None else int(time.time())
    rng = np.random.default_rng(seed)

    n = 1 + cfg.n_objects

    m = np.empty((n,), dtype=float)
    r = np.zeros((n, 3), dtype=float)
    v = np.zeros((n, 3), dtype=float)

    # central body
    m[0] = cfg.central_mass
    r[0] = [0.0, 0.0, 0.0]
    v[0] = [0.0, 0.0, 0.0]

    # orbiting bodies
    radii = rng.uniform(cfg.min_radius_au, cfg.max_radius_au, size=cfg.n_objects)
    vy = rng.uniform(cfg.min_vy, cfg.max_vy, size=cfg.n_objects)
    vz = rng.uniform(cfg.min_vz, cfg.max_vz, size=cfg.n_objects)
    masses = rng.uniform(cfg.min_mass, cfg.max_mass, size=cfg.n_objects)

    m[1:] = masses
    r[1:, 0] = radii
    v[1:, 1] = vy
    v[1:, 2] = vz

    return NBodySystem(m=m, r=r, v=v, softening=0.0)


def main() -> None:
    # all units in convention of Gauss grav. constant k = 0.01720
    # [M] = M_sun, [r] = AU, [v] = AU/d
    #
    # --- Original demos preserved ---
    #
    # lf = leapfrog.Leapfrog(m = [1.9891e30, 5.97219e24],
    #               r = [[0., 0., 0.], [sp.constants.au, 0., 0.]],
    #               v = [[0., 0., 0.], [0., 40.*3600, 0.]])
    #
    # lf = leapfrog.Leapfrog(m = [0.10000000000000000e+01, 0.28583678719451197e-03],
    #               r = [[0., 0., 0.], [0.69905175661092100e+00, -0.95522604421347300e+01, -0.43673105309042500e-01]],
    #               v = [[0., 0., 0.], [0.55414811016520800e-02, 0.29850559025851500e-03, -0.22127242395468300e-03]])
    #
    # --- Random demo ---

    system = build_random_system(RandomSystemConfig(n_objects=3, central_mass=3.0))

    stop = 100000 / 2.0  # 365. * 86400
    step = 20.0  # 24.*3600

    _, positions = integrate(system, dt=step, t_stop=stop, record=True)

    handles = animate_trajectories_3d(
        positions,
        interval_ms=20,
        # was interval=1 in your old code :contentReference[oaicite:11]{index=11}, but modern backends prefer >1
        trail=True,
        title="N-body leapfrog demo",
    )

    # IMPORTANT: keep `handles` referenced until after plt.show()
    show_animation(handles)


if __name__ == "__main__":
    main()
