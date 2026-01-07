from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # required for 3D projection

_BACKEND = os.environ.get("GARGANTUA_MPL_BACKEND", "").strip()
if _BACKEND:
    mpl.use(_BACKEND, force=True)
else:
    # Try stable interactive backends first; fallback to Agg.
    for candidate in ("TkAgg", "QtAgg", "Agg"):
        # noinspection PyBroadException
        try:
            mpl.use(candidate, force=True)
            break
        except Exception:  # pragma: no cover  # noqa: BLE001, S112
            continue
ArrayF = np.ndarray


@dataclass(slots=True)
class AnimationHandles:
    """Keep references alive.

    If you don't keep the FuncAnimation referenced, you can get a blank window
    (or an animation that never starts) depending on backend and environment.
    """

    fig: plt.Figure
    ax: plt.Axes
    anim: FuncAnimation


def _compute_bounds(positions: ArrayF, pad_fraction: float = 0.05) -> tuple[ArrayF, ArrayF]:
    """Compute boundaries.

    positions: (steps, n, 3)
    returns (min_xyz, max_xyz) with padding
    """
    min_xyz = np.min(positions, axis=(0, 1))
    max_xyz = np.max(positions, axis=(0, 1))
    span = np.maximum(max_xyz - min_xyz, 1e-12)
    pad = pad_fraction * span
    return min_xyz - pad, max_xyz + pad


def animate_trajectories_3d(  # noqa: C901
        positions: ArrayF,
        *,
        interval_ms: int = 20,
        trail: bool = True,
        show_axes: bool = True,
        title: str | None = None,
) -> AnimationHandles:
    """Animate trajectories in 3D.

    positions: array (steps, n, 3)

    Replaces your old Vis.visualize() + update() logic :contentReference[oaicite:7]{index=7}
    with a simpler, more reliable FuncAnimation for modern Matplotlib.

    Key fixes for "blank screen":
    - explicit frames=steps
    - blit=False (3D does not blit well)
    - keep a reference to the animation object (returned)
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        msg = f"positions must have shape (steps, n, 3), got {pos.shape}"
        raise ValueError(msg)

    steps, n, _ = pos.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if title:
        ax.set_title(title)

    # bounds based on full trajectory
    min_xyz, max_xyz = _compute_bounds(pos)
    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])

    if not show_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Artists: points + optional trails
    points = [ax.plot([pos[0, i, 0]], [pos[0, i, 1]], [pos[0, i, 2]], marker="o", linestyle="")[0] for i in range(n)]
    trails = None
    if trail:
        trails = [ax.plot([pos[0, i, 0]], [pos[0, i, 1]], [pos[0, i, 2]])[0] for i in range(n)]

    def init() -> list:
        # set initial data
        for i in range(n):
            points[i].set_data([pos[0, i, 0]], [pos[0, i, 1]])
            # noinspection PyUnresolvedReferences
            points[i].set_3d_properties([pos[0, i, 2]])
            if trails is not None:
                trails[i].set_data([pos[0, i, 0]], [pos[0, i, 1]])
                # noinspection PyUnresolvedReferences
                trails[i].set_3d_properties([pos[0, i, 2]])
        return points + (trails if trails is not None else [])

    def update(frame: int) -> list:
        # frame in [0..steps-1]
        for i in range(n):
            points[i].set_data([pos[frame, i, 0]], [pos[frame, i, 1]])
            # noinspection PyUnresolvedReferences
            points[i].set_3d_properties([pos[frame, i, 2]])
            if trails is not None:
                trails[i].set_data(pos[: frame + 1, i, 0], pos[: frame + 1, i, 1])
                # noinspection PyUnresolvedReferences
                trails[i].set_3d_properties(pos[: frame + 1, i, 2])
        return points + (trails if trails is not None else [])

    anim = FuncAnimation(
        fig,
        update,
        frames=steps,  # important
        init_func=init,
        interval=interval_ms,
        blit=False,  # important for 3D
        repeat=True,
    )

    return AnimationHandles(fig=fig, ax=ax, anim=anim)


def show_animation(_: AnimationHandles, *, block: bool = True) -> None:
    """Call this to show the animation and keep references alive in user code."""
    plt.show(block=block)
