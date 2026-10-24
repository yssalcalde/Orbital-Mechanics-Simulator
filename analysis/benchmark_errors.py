"""Quantitative comparison of orbital integration methods."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physics import G, Planet


DATA_FILE = ROOT / "data" / "Moon-Earth.txt"
ASSETS = ROOT / "assets"
MAIN_FIGURE = ASSETS / "benchmark_errors.png"
CONVERGENCE_FIGURE = ASSETS / "benchmark_convergence.png"
SUMMARY_FILE = ASSETS / "benchmark_summary.csv"
TIME_STEP = 7_200.0
REFERENCE_TIME_STEP = 60.0
METHODS = ("euler", "euler_implicit", "rk4", "verlet")
METHOD_LABELS = {
    "euler": "Explicit Euler",
    "euler_implicit": "Implicit Euler",
    "rk4": "RK4",
    "verlet": "Velocity Verlet",
}


@dataclass
class System:
    central_mass: float
    planet: Planet


@dataclass
class SimulationResult:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    energies: np.ndarray
    angular_momenta: np.ndarray
    runtime: float
    failure: str | None


def load_system(path: Path) -> System:
    """Load one central body and one orbiting body from a CSV data file."""
    with path.open(newline="", encoding="utf-8") as data_file:
        rows = list(csv.DictReader(data_file))

    central_rows = [row for row in rows if row["type"].strip().lower() == "central"]
    body_rows = [row for row in rows if row["type"].strip().lower() == "body"]
    if len(central_rows) != 1 or len(body_rows) != 1:
        raise ValueError("The benchmark data must contain one central row and one body row.")

    central = central_rows[0]
    body = body_rows[0]
    return System(
        central_mass=float(central["mass_kg"]),
        planet=Planet(
            name=body["name"],
            mass=float(body["mass_kg"]),
            distance=float(body["distance_m"]),
            velocity=float(body["velocity_m_s"]),
            inclination=float(body["inclination_deg"]),
            color=body["color"],
            size=int(body["size"]),
        ),
    )


def clone_planet(planet: Planet) -> Planet:
    """Create a planet with the same initial state."""
    return Planet(
        planet.name,
        planet.mass,
        np.linalg.norm(planet.position),
        np.linalg.norm(planet.velocity),
        np.degrees(planet.inclination),
        planet.color,
        planet.size,
    )


def simulate(
    method: str,
    system: System,
    dt: float,
    duration: float,
    store_history: bool = True,
) -> SimulationResult:
    """Simulate one method and retain physical diagnostics."""
    central_position = np.zeros(3, dtype=np.float64)
    planet = clone_planet(system.planet)
    steps = int(np.ceil(duration / dt))
    times = np.arange(steps + 1, dtype=np.float64) * dt
    positions = np.full((steps + 1, 3), np.nan)
    velocities = np.full((steps + 1, 3), np.nan)
    energies = np.full(steps + 1, np.nan)
    angular_momenta = np.full((steps + 1, 3), np.nan)
    positions[0] = planet.position
    velocities[0] = planet.velocity
    energies[0] = planet.specific_energy(system.central_mass, central_position)
    angular_momenta[0] = planet.specific_angular_momentum(central_position)
    start = time.perf_counter()
    failure = None

    for step in range(1, steps + 1):
        try:
            planet.update(dt, system.central_mass, central_position, method)
            positions[step] = planet.position
            velocities[step] = planet.velocity
            energies[step] = planet.specific_energy(system.central_mass, central_position)
            angular_momenta[step] = planet.specific_angular_momentum(central_position)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failure = f"{type(error).__name__}: {error}"
            times = times[: step + 1]
            positions = positions[: step + 1]
            velocities = velocities[: step + 1]
            energies = energies[: step + 1]
            angular_momenta = angular_momenta[: step + 1]
            break

    runtime = time.perf_counter() - start
    if not store_history:
        positions = positions[-1:]
        velocities = velocities[-1:]
        energies = energies[-1:]
        angular_momenta = angular_momenta[-1:]
        times = times[-1:]
    return SimulationResult(
        times, positions, velocities, energies, angular_momenta, runtime, failure
    )


def relative_error(values: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Return the relative norm error for a vector history."""
    denominator = np.linalg.norm(initial)
    return np.linalg.norm(values - initial, axis=1) / denominator


def plot_main_benchmark(
    system: System,
    duration: float,
    orbital_period: float,
) -> dict[str, SimulationResult]:
    """Plot radius, energy, angular momentum, and orbital plane histories."""
    results = {
        method: simulate(method, system, TIME_STEP, duration)
        for method in METHODS
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    distance_axis, energy_axis, momentum_axis, orbit_axis = axes.flat
    central_position = np.zeros(3, dtype=np.float64)
    initial_energy = system.planet.specific_energy(system.central_mass, central_position)
    initial_momentum = system.planet.specific_angular_momentum(central_position)

    for method, result in results.items():
        label = METHOD_LABELS[method]
        days = result.times / 86_400.0
        distance_axis.plot(days, np.linalg.norm(result.positions, axis=1) / 1e6, label=label)
        energy_error = np.abs((result.energies - initial_energy) / initial_energy)
        momentum_error = relative_error(result.angular_momenta, initial_momentum)
        energy_axis.semilogy(days, energy_error, label=label)
        momentum_axis.semilogy(days, momentum_error, label=label)
        orbit_axis.plot(
            result.positions[:, 0] / 1e6,
            result.positions[:, 1] / 1e6,
            label=label,
        )
        status = "completed" if result.failure is None else f"stopped: {result.failure}"
        print(f"{label}: {status}; runtime={result.runtime:.3f} s")

    distance_axis.set_title("Distance from central body")
    distance_axis.set_xlabel("Time (days)")
    distance_axis.set_ylabel("Distance (10^6 m)")
    energy_axis.set_title("Relative specific-energy error")
    energy_axis.set_xlabel("Time (days)")
    energy_axis.set_ylabel("|(E(t) - E(0)) / E(0)|")
    momentum_axis.set_title("Relative angular-momentum error")
    momentum_axis.set_xlabel("Time (days)")
    momentum_axis.set_ylabel("|h(t) - h(0)| / |h(0)|")
    orbit_axis.set_title("Orbital plane")
    orbit_axis.set_xlabel("x (10^6 m)")
    orbit_axis.set_ylabel("y (10^6 m)")
    orbit_axis.set_aspect("equal", adjustable="box")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Earth-Moon integration comparison: 10 periods, dt={TIME_STEP / 3600:.0f} h "
        f"(period={orbital_period / 86_400:.2f} days)"
    )
    figure.tight_layout()
    figure.savefig(MAIN_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return results


def run_convergence_study(system: System, duration: float) -> None:
    """Measure final position error and runtime for several time steps."""
    time_steps = (14_400.0, 7_200.0, 3_600.0, 1_800.0, 900.0)
    rows: list[dict[str, str | float]] = []
    figure, (error_axis, runtime_axis) = plt.subplots(1, 2, figsize=(13, 5))

    for method in METHODS:
        errors = []
        runtimes = []
        for dt in time_steps:
            result = simulate(method, system, dt, duration, store_history=False)
            reference_duration = result.times[-1]
            reference = simulate(
                "rk4", system, REFERENCE_TIME_STEP, reference_duration, store_history=False
            )
            position_error = np.linalg.norm(result.positions[-1] - reference.positions[-1])
            errors.append(position_error)
            runtimes.append(result.runtime)
            rows.append(
                {
                    "method": method,
                    "time_step_s": dt,
                    "final_position_error_m": position_error,
                    "runtime_s": result.runtime,
                    "status": "completed" if result.failure is None else result.failure,
                }
            )
        error_axis.loglog(time_steps, errors, "o-", label=METHOD_LABELS[method])
        runtime_axis.loglog(time_steps, runtimes, "o-", label=METHOD_LABELS[method])

    error_axis.set_title("Convergence against RK4 reference")
    error_axis.set_xlabel("Time step (s)")
    error_axis.set_ylabel("Final position error (m)")
    runtime_axis.set_title("Runtime")
    runtime_axis.set_xlabel("Time step (s)")
    runtime_axis.set_ylabel("Runtime (s)")
    for axis in (error_axis, runtime_axis):
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(CONVERGENCE_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)

    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the benchmark and save figures and numerical results."""
    system = load_system(DATA_FILE)
    distance = np.linalg.norm(system.planet.position)
    orbital_period = 2 * np.pi * np.sqrt(distance**3 / (G * system.central_mass))
    duration = 10 * orbital_period
    ASSETS.mkdir(exist_ok=True)
    plot_main_benchmark(system, duration, orbital_period)
    run_convergence_study(system, duration)
    print(f"Saved {MAIN_FIGURE}")
    print(f"Saved {CONVERGENCE_FIGURE}")
    print(f"Saved {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
