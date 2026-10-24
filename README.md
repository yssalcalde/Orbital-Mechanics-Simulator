# Orbital Mechanics Simulator

This project compares numerical integrators for a Newtonian Earth-Moon orbit.
The benchmark runs ten lunar orbital periods with a deliberately large
two-hour time step:

```powershell
.\.venv\Scripts\python.exe analysis\benchmark_errors.py
```

It saves the comparison figure to
[assets/benchmark_errors.png](assets/benchmark_errors.png). The upper plot
shows the Earth-Moon distance; the lower plot shows the relative specific
energy error.

## Why symplectic integration matters

Explicit Euler and classical RK4 do not preserve the Hamiltonian structure of
orbital motion. Euler quickly gains or loses orbital energy, while RK4 has
small local errors but can accumulate a secular energy drift over many
periods. Velocity Verlet is only second-order accurate per time step, but it
is symplectic: its energy error usually remains bounded and oscillates around
the correct value instead of continuously diverging. That bounded long-term
behavior is why symplectic integrators are essential for long-duration orbital
mechanics simulations.

The benchmark uses the restricted two-body model implemented in
[src/physics.py](src/physics.py): Earth is fixed, and the Moon is treated as a
point mass acted on only by Earth's Newtonian gravity.
