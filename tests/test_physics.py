import numpy as np
import pytest

from src.physics import (
    G,
    Planet,
    analytical_circular_state,
    circular_orbital_velocity,
)


def test_circular_orbital_velocity_matches_kepler_solution() -> None:
    mass = 5.972e24
    radius = 3.844e8
    assert circular_orbital_velocity(mass, radius) == pytest.approx(
        np.sqrt(G * mass / radius)
    )


def test_analytical_circular_state_matches_integrated_rk4_orbit() -> None:
    mass = 5.972e24
    radius = 7.0e6
    inclination = 23.0
    period = 2 * np.pi * np.sqrt(radius**3 / (G * mass))
    planet = Planet(
        "test",
        1.0,
        radius,
        circular_orbital_velocity(mass, radius),
        inclination,
        "white",
        1,
    )
    central_position = np.zeros(3)
    steps = 1000
    dt = period / steps
    for _ in range(steps):
        planet.update(dt, mass, central_position, "rk4")

    expected_position, expected_velocity = analytical_circular_state(
        mass, radius, inclination, period
    )
    np.testing.assert_allclose(planet.position, expected_position, rtol=1e-8, atol=1e-2)
    np.testing.assert_allclose(planet.velocity, expected_velocity, rtol=1e-8, atol=2e-6)


def test_planet_rejects_invalid_initial_conditions() -> None:
    with pytest.raises(ValueError, match="mass"):
        Planet("invalid", 0, 1, 1, 0, "white", 1)
    with pytest.raises(ValueError, match="distance"):
        Planet("invalid", 1, 0, 1, 0, "white", 1)
