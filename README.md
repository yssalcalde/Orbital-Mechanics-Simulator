# Orbital Mechanics Simulator

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-required-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-required-11557C)

A Python simulator for studying numerical integration methods in the
restricted Newtonian two-body problem. The project combines an interactive
3D visualizer with quantitative benchmarks of Explicit Euler, Implicit Euler,
classical fourth-order Runge-Kutta (RK4), and Velocity Verlet.

> **Project context:** developed as an aerospace-engineering student project
> focused on orbital mechanics, numerical stability, and scientific
> visualization.

## Features

- Newtonian point-mass gravitational dynamics.
- Explicit Euler, Implicit Euler, RK4, and Velocity Verlet integrators.
- Interactive 3D orbital visualization.
- Selectable system file, integration method, and time step.
- Black space background with stars and mouse/touchpad zoom.
- Pause, reset, and steps-per-frame controls.
- Energy and angular-momentum conservation diagnostics.
- Position-error, convergence, and runtime benchmarks.
- CSV-based system definitions using SI units.

## Demonstration

The visualizer can be started with the Earth-Moon example in
[data/Moon-Earth.txt](data/Moon-Earth.txt).

[Orbital simulator demonstration](assets\AnimationEarthMoonVerlet.gif)

## Installation

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run the project using the virtual
environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Running the simulator

Start the graphical application:

```powershell
.\.venv\Scripts\python.exe main.py
```

The setup window lets you choose:

- an input system file;
- Explicit Euler, Implicit Euler, RK4, or Velocity Verlet;
- the physical time step in seconds.

The visualizer provides:

- **Pause** and **Reset** buttons;
- **Steps/frame**, which controls how many physical updates occur per
  displayed frame;
- **dt multiplier**, which changes the physical time step while running;
- mouse-wheel or touchpad scrolling for camera zoom;
- mouse dragging for rotating the 3D view.

`dt multiplier` affects the numerical simulation. Zoom affects only the camera.
Increasing the physical time step can make the animation faster, but can also
reduce numerical accuracy or destabilize an integrator.

## Running the benchmark

Run the Earth-Moon benchmark:

```powershell
.\.venv\Scripts\python.exe analysis\benchmark_errors.py
```

The benchmark simulates ten lunar orbital periods using a deliberately large
two-hour step. It generates:

- [benchmark_errors.png](assets/benchmark_errors.png): distance, relative
  specific-energy error, angular-momentum error, and orbital-plane plots;
- [benchmark_convergence.png](assets/benchmark_convergence.png): final
  position error and runtime for several time steps;
- [benchmark_summary.csv](assets/benchmark_summary.csv): raw numerical
  results.

The convergence study compares each method with an RK4 reference solution
using a 60-second time step. The tested steps are 14,400, 7,200, 3,600,
1,800, and 900 seconds.

## Numerical interpretation

The benchmark demonstrates the trade-offs between the methods:

- **Explicit Euler** is inexpensive but gains energy and causes the orbit to
  expand.
- **Implicit Euler** dissipates energy and causes the orbit to contract. It
  also requires a Newton-Raphson solve and can fail to converge for a large
  step.
- **RK4** provides excellent short- and medium-term accuracy, but it is not
  symplectic, so long-term energy conservation is not guaranteed.
- **Velocity Verlet** is second-order accurate but symplectic. Its energy
  error generally remains bounded and oscillatory, making it well suited to
  long-term conservative orbital simulations.

For this reason, the method with the smallest short-term position error is not
always the best method for a very long simulation.

## Input data format

System files use comma-separated values and SI units:

```text
type,name,mass_kg,distance_m,velocity_m_s,inclination_deg,color,size
central,Earth,5.972e24,0,0,0,blue,15
body,Moon,7.3477e22,3.844e8,1.022e3,5.145,gray,5
```

Distances are measured in metres, velocities in metres per second, masses in
kilograms, and inclinations in degrees. The current loader expects one
`central` row and at least one `body` row.

## Physical model and limitations

The current engine implements a restricted two-body model:

- the central body is fixed at the origin;
- orbiting bodies are treated as point masses;
- only the central body's gravity is applied;
- orbiting bodies do not gravitationally interact with one another;
- the model does not include atmospheric drag, radiation pressure, tides,
  oblateness, or relativistic effects.

The model is therefore appropriate for comparing integrators and studying
Keplerian orbital motion, but it is not a high-fidelity ephemeris or complete
N-body solar-system simulator.

## Project structure

```text
.
├── analysis/
│   └── benchmark_errors.py    # Accuracy, conservation, convergence, runtime
├── assets/                    # Generated benchmark figures and CSV results
├── data/                      # Example system definitions
├── src/
│   ├── physics.py             # Planet model and numerical integrators
│   ├── gui.py                 # Tkinter setup window
│   └── visualizer.py          # Matplotlib 3D animation
├── main.py                    # GUI entry point
├── requirements.txt
└── README.md
```

## License

This project is released under the [MIT License](LICENSE). You may use,
modify, and redistribute the code, provided that the original copyright and
license notice are retained. The software is provided without warranty.

## Possible future work

- General N-body dynamics and barycentric coordinates.
- Orbital-element calculations.
- Physical collision detection using body radii.
- J2 perturbation, atmospheric drag, and radiation pressure.
- Analytical Kepler-orbit comparisons.
- Spacecraft transfers such as Hohmann and Lambert trajectories.
- Automated unit and regression tests.
