# Leapfrog N-Body Simulator

Educational gravitational N-body simulator using a symplectic leapfrog
integrator, vectorized NumPy, and real-time 3D visualization with
Matplotlib.

The project focuses on numerical correctness, physical clarity, and easy
experimentation.

------------------------------------------------------------------------

## Features

- Newtonian N-body gravity
- Symplectic leapfrog (kick-drift-kick)
- Vectorized force computation
- Center-of-mass (COM) coordinate support
- Real-time 3D animation
- Preserved legacy implementation for reference
- LaTeX technical documentation

------------------------------------------------------------------------

## How to Run

1. Install dependencies:
```shell
pip install -r requirements.txt
```

2. Open `main.py` in PyCharm.

3. Select the demo by editing:
```python
DEMO = "figure8"  # or `random`
```
4. Press Run ▶

------------------------------------------------------------------------

## Demos

### figure8

Exact 3-body figure-eight orbit.
Stable and periodic motion.
Good for validating integrator correctness.

### random_nbody

Random chaotic N-body system.
Bodies scatter, form temporary binaries, and escape.
Good for stress-testing dynamics and performance.

------------------------------------------------------------------------

## Center of Mass

The simulation supports working in the center-of-mass frame:

- Initial conditions can be shifted into COM coordinates.
- Trajectories can be visualized in COM frame.

Benefits:

- No translational drift in visualization.
- Easier interpretation of dynamics.
- Better numerical diagnostics.

------------------------------------------------------------------------

## Physics Model

Newtonian gravity:

r¨ᵢ = -G Σⱼ≠ᵢ mⱼ (rᵢ - rⱼ) / |rᵢ - rⱼ|³

Integrated using leapfrog for long-term stability.

------------------------------------------------------------------------

## Documentation

Mathematical documentation is provided in LaTeX and covers:

- Physical model
- Units and constants
- Vectorization
- Leapfrog derivation
- Center-of-mass handling
- Numerical properties

------------------------------------------------------------------------

## License

See LICENSE.