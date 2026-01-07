from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

ArrayF = np.ndarray


@dataclass(slots=True)
class NBodySystem:
    """Simple N-body Newtonian gravitation in Gauss-like units.

    Convention used by your original code:
    - [M] = M_sun
    - [r] = AU
    - [v] = AU / day
    - G is set to k^2 where k is Gauss gravitational constant (≈ 0.01720209895)
      so that AU, day, solar mass system behaves nicely.

    This mirrors your original constant:
      self.G = 0.017202098950**2
    """

    m: ArrayF  # shape: (n,)
    r: ArrayF  # shape: (n, 3)
    v: ArrayF  # shape: (n, 3)
    gauss_g: float = 0.017202098950 ** 2
    softening: float = 0.0  # AU, optional gravitational softening to avoid singularities


def _as_float_array(x: ArrayF, *, shape: tuple[int, ...] | None = None) -> ArrayF:
    arr = np.asarray(x, dtype=float)
    if shape is not None and arr.shape != shape:
        msg = f"Expected shape {shape}, got {arr.shape}"
        raise ValueError(msg)
    return arr


def compute_accelerations(r: ArrayF, m: ArrayF, *, gauss_g: float, softening: float = 0.0) -> ArrayF:
    """Compute vectorized accelerations for all bodies.

    a_i = -gauss_g * sum_{j!=i} m_j * (r_i - r_j) / |r_i - r_j|^3
    """
    # r: (n, 3)
    # dr[i, j, :] = r[i, :] - r[j, :]
    dr = r[:, None, :] - r[None, :, :]  # (n, n, 3)

    dist2 = np.sum(dr * dr, axis=2)  # (n, n)
    if softening > 0.0:
        dist2 = dist2 + softening * softening

    # Avoid self-interaction: set diagonal to inf so 1/dist^3 becomes 0
    np.fill_diagonal(dist2, np.inf)

    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))  # 1 / |r|^3

    # Weight by masses on j index
    # (n, n, 1) * (n, n, 3) -> (n, n, 3), sum over j -> (n, 3)
    return -gauss_g * np.sum((m[None, :, None] * inv_dist3[:, :, None]) * dr, axis=1)


def integrate(
        system: NBodySystem,
        *,
        dt: float,
        t_stop: float,
        record: bool = True,
        dtype: type = np.float64,
) -> tuple[ArrayF, ArrayF]:
    """Leapfrog (kick-drift-kick), but written in modern, vectorized style.

    Returns:
      times: (steps, )
      positions: (steps, n, 3) if record=True else (1, n, 3) with final position only

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

    if record:  # noqa: SIM108
        traj = np.empty((steps, r.shape[0], 3), dtype=dtype)
    else:
        traj = np.empty((1, r.shape[0], 3), dtype=dtype)

    # initial acceleration
    a = compute_accelerations(r, m, gauss_g=system.gauss_g, softening=system.softening).astype(dtype, copy=False)

    for k in range(steps):
        # Record BEFORE stepping (matches your old "append r_zero then update") :contentReference[oaicite:5]{index=5}
        if record:
            traj[k] = r
        elif k == steps - 1:
            traj[0] = r

        # kick (half)
        v = v + 0.5 * dt * a

        # drift
        r = r + dt * v

        # update acceleration at new position
        a = compute_accelerations(r, m, gauss_g=system.gauss_g, softening=system.softening).astype(dtype, copy=False)

        # kick (half)
        v = v + 0.5 * dt * a

    # update system (so caller can keep integrating)
    system.m = m
    system.r = r
    system.v = v

    return times, traj


def iter_leapfrog_positions(
        system: NBodySystem,
        *,
        dt: float,
        steps: int,
        dtype: type = np.float64,
) -> Iterator[ArrayF]:
    """Iterate leaprforg position.

    Generator version: yields positions (n, 3) each step.

    Useful if you want truly streaming animation later.
    """
    m = _as_float_array(system.m).astype(dtype, copy=False)
    r = _as_float_array(system.r).astype(dtype, copy=True)
    v = _as_float_array(system.v).astype(dtype, copy=True)

    a = compute_accelerations(r, m, gauss_g=system.gauss_g, softening=system.softening).astype(dtype, copy=False)

    for _ in range(steps):
        yield r

        v = v + 0.5 * dt * a
        r = r + dt * v
        a = compute_accelerations(r, m, gauss_g=system.gauss_g, softening=system.softening).astype(dtype, copy=False)
        v = v + 0.5 * dt * a

    system.m = m
    system.r = r
    system.v = v
