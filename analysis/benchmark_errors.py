"""Compare orbital integrators on a ten-period Earth-Moon simulation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physics import G, Planet


DATA_FILE = ROOT / "data" / "Moon-Earth.txt"
OUTPUT_FILE = ROOT / "assets" / "benchmark_errors.png"
TIME_STEP = 7_200.0
METHODS = ("euler", "euler_implicit", "rk4", "verlet")


def load_system(path: Path) -> tuple[float, Planet]:
    """Load the fixed central body and orbiting body from a CSV data file."""
    with path.open(newline="", encoding="utf-8") as data_file:
        rows = list(csv.DictReader(data_file))

    central_rows = [row for row in rows if row["type"].strip().lower() == "central"]
    body_rows = [row for row in rows if row["type"].strip().lower() == "body"]
    if len(central_rows) != 1 or len(body_rows) != 1:
        raise ValueError("The benchmark data must contain one central row and one body row.")

    central_mass = float(central_rows[0]["mass_kg"])
    body = body_rows[0]
    planet = Planet(
        name=body["name"],
        mass=float(body["mass_kg"]),
        distance=float(body["distance_m"]),
        velocity=float(body["velocity_m_s"]),
        inclination=float(body["inclination_deg"]),
        color=body["color"],
        size=int(body["size"]),
    )
    return central_mass, planet


def run_method(
    method: str,
    central_mass: float,
    initial_planet: Planet,
    steps: int,
    central_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Run one method and return distance, relative-energy-error, and failure."""
    planet = Planet(
        initial_planet.name,
        initial_planet.mass,
        np.linalg.norm(initial_planet.position),
        np.linalg.norm(initial_planet.velocity),
        np.degrees(initial_planet.inclination),
        initial_planet.color,
        initial_planet.size,
    )
    initial_energy = planet.specific_energy(central_mass, central_position)
    distances = np.full(steps + 1, np.nan)
    energy_errors = np.full(steps + 1, np.nan)
    distances[0] = np.linalg.norm(planet.position)
    failure = None

    for step in range(1, steps + 1):
        try:
            planet.update(TIME_STEP, central_mass, central_position, method)
            energy = planet.specific_energy(central_mass, central_position)
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failure = f"{type(error).__name__}: {error}"
            break

        distances[step] = np.linalg.norm(planet.position)
        energy_errors[step] = abs((energy - initial_energy) / initial_energy)

    return distances, energy_errors, failure


def main() -> None:
    """Generate and save the benchmark figure."""
    central_mass, planet = load_system(DATA_FILE)
    central_position = np.zeros(3, dtype=np.float64)
    initial_distance = np.linalg.norm(planet.position)
    orbital_period = 2 * np.pi * np.sqrt(initial_distance**3 / (G * central_mass))
    total_time = 10 * orbital_period
    steps = int(np.ceil(total_time / TIME_STEP))
    times = np.arange(steps + 1) * TIME_STEP / 86_400.0

    figure, (distance_axis, energy_axis) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for method in METHODS:
        distances, energy_errors, failure = run_method(
            method, central_mass, planet, steps, central_position
        )
        distance_axis.plot(times, distances / 1e6, label=method)
        energy_axis.semilogy(times, energy_errors, label=method)
        if failure:
            print(f"{method}: stopped after a numerical failure ({failure})")
        else:
            print(f"{method}: completed {steps} steps")

    distance_axis.set_title("Earth-Moon benchmark: distance from Earth")
    distance_axis.set_ylabel("Distance (10^6 m)")
    distance_axis.grid(True, alpha=0.3)
    distance_axis.legend()

    energy_axis.set_title("Relative specific energy error")
    energy_axis.set_xlabel("Time (days)")
    energy_axis.set_ylabel("|(E(t) - E(0)) / E(0)|")
    energy_axis.grid(True, alpha=0.3, which="both")
    energy_axis.legend()

    figure.suptitle(
        f"Ten lunar orbital periods, dt = {TIME_STEP / 3600:.0f} h "
        f"(period = {orbital_period / 86_400:.2f} days)"
    )
    figure.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    print(f"Saved benchmark figure to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()