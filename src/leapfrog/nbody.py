from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayF = np.ndarray


@dataclass(slots=True)
class NBodySystem:
    """General Newtonian N-body system.

    Units are user-defined, but by default we keep your original astronomical convention:
    - [M] = M_sun
    - [r] = AU
    - [v] = AU / day
    - G = k^2 where k is Gauss' gravitational constant

    You can also set grav_const=1.0 for dimensionless demos (e.g. figure-eight).
    """

    m: ArrayF  # (n,)
    r: ArrayF  # (n, 3)
    v: ArrayF  # (n, 3)

    grav_const: float = 0.017202098950 ** 2
    softening: float = 0.0  # optional epsilon to avoid singularities


def _as_float_array(x: ArrayF, *, shape: tuple[int, ...] | None = None) -> ArrayF:
    arr = np.asarray(x, dtype=float)
    if shape is not None and arr.shape != shape:
        msg = f"Expected shape {shape}, got {arr.shape}"
        raise ValueError(msg)
    return arr


def center_of_mass(m: ArrayF, r: ArrayF) -> ArrayF:
    """Center-of-mass position R_cm, shape (3,)."""
    m = _as_float_array(m)
    r = _as_float_array(r)
    m_tot = float(np.sum(m))
    if m_tot == 0.0:
        msg = "Total mass is zero."
        raise ValueError(msg)
    return (m[:, None] * r).sum(axis=0) / m_tot


def center_of_mass_velocity(m: ArrayF, v: ArrayF) -> ArrayF:
    """Center-of-mass velocity V_cm, shape (3,)."""
    m = _as_float_array(m)
    v = _as_float_array(v)
    m_tot = float(np.sum(m))
    if m_tot == 0.0:
        msg = "Total mass is zero."
        raise ValueError(msg)
    return (m[:, None] * v).sum(axis=0) / m_tot


def shift_system_to_com_frame(system: NBodySystem) -> None:
    """In-place shift: r -= R_cm and v -= V_cm.

    If total momentum is conserved (it should be, up to floating error),
    then the COM should remain near the origin afterwards.
    """
    r_cm = center_of_mass(system.m, system.r)
    v_cm = center_of_mass_velocity(system.m, system.v)
    system.r = system.r - r_cm[None, :]
    system.v = system.v - v_cm[None, :]


def compute_accelerations(
        r: ArrayF,
        m: ArrayF,
        *,
        grav_const: float,
        softening: float = 0.0,
) -> ArrayF:
    """Vectorized accelerations for all bodies.

    a_i = -G * sum_{j!=i} m_j * (r_i - r_j) / |r_i - r_j|^3
    """
    dr = r[:, None, :] - r[None, :, :]  # (n, n, 3)
    dist2 = np.sum(dr * dr, axis=2)  # (n, n)

    if softening > 0.0:
        dist2 = dist2 + softening * softening

    # remove self interaction
    np.fill_diagonal(dist2, np.inf)

    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))  # 1 / |r|^3
    return -grav_const * np.sum((m[None, :, None] * inv_dist3[:, :, None]) * dr, axis=1)


def integrate_leapfrog(
        system: NBodySystem,
        *,
        dt: float,
        t_stop: float,
        record: bool = True,
        dtype: type = np.float64,
) -> tuple[ArrayF, ArrayF]:
    """Leapfrog integrator (kick-drift-kick).

    Returns:
      times: (steps,)
      positions: (steps, n, 3) if record=True else (1, n, 3)

    """
    m = _as_float_array(system.m).astype(dtype, copy=False)
    r = _as_float_array(system.r).astype(dtype, copy=True)
    v = _as_float_array(system.v).astype(dtype, copy=True)

    if r.ndim != 2 or r.shape[1] != 3:
        msg = f"r must have shape (n, 3), got {r.shape}"
        raise ValueError(msg)
    if v.shape != r.shape:
        msg = f"v must have same shape as r. r={r.shape}, v={v.shape}"
        raise ValueError(msg)
    if m.ndim != 1 or m.shape[0] != r.shape[0]:
        msg = f"m must have shape (n,), got {m.shape}, expected n={r.shape[0]}"
        raise ValueError(msg)

    steps = int(np.ceil(t_stop / dt))
    times = (np.arange(steps, dtype=dtype) * dt)

    traj = np.empty((steps, r.shape[0], 3), dtype=dtype) if record else np.empty((1, r.shape[0], 3), dtype=dtype)

    a = compute_accelerations(r, m, grav_const=system.grav_const, softening=system.softening).astype(dtype, copy=False)

    for k in range(steps):
        if record:
            traj[k] = r
        elif k == steps - 1:
            traj[0] = r

        # kick half
        v = v + 0.5 * dt * a

        # drift full
        r = r + dt * v

        # update acceleration at r_{n+1}
        a = compute_accelerations(
            r=r,
            m=m,
            grav_const=system.grav_const,
            softening=system.softening,
        ).astype(dtype, copy=False)

        # kick half
        v = v + 0.5 * dt * a

    system.m = m
    system.r = r
    system.v = v
    return times, traj


def trajectory_to_com_frame(positions: ArrayF, m: ArrayF) -> ArrayF:
    """Return a new trajectory in COM coordinates for each frame.

    positions: (steps, n, 3)
    m: (n, ),
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        msg = f"positions must have shape (steps, n, 3), got {pos.shape}"
        raise ValueError(msg)

    m = _as_float_array(m)
    m_tot = float(np.sum(m))
    if m_tot == 0.0:
        msg = "Total mass is zero."
        raise ValueError(msg)

    # R_cm(t) = sum_i m_i r_i(t) / M
    r_cm = (pos * m[None, :, None]).sum(axis=1) / m_tot  # (steps, 3)
    return pos - r_cm[:, None, :]
