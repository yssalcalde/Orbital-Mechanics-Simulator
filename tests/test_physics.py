import numpy as np
import pytest

from src.physics import (
    G,
    Planet,
    analytical_circular_state,
    analytical_kepler_state,
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


def test_elliptical_kepler_state_returns_to_periapsis_after_one_period() -> None:
    mass = 5.972e24
    semi_major_axis = 1.0e7
    eccentricity = 0.5
    inclination = 18.0
    period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (G * mass))
    initial_position, initial_velocity = analytical_kepler_state(
        mass, semi_major_axis, eccentricity, inclination, 0.0
    )
    final_position, final_velocity = analytical_kepler_state(
        mass, semi_major_axis, eccentricity, inclination, period
    )
    np.testing.assert_allclose(final_position, initial_position, rtol=1e-12, atol=3e-9)
    np.testing.assert_allclose(final_velocity, initial_velocity, rtol=1e-12, atol=3e-9)
    assert np.linalg.norm(initial_position) == pytest.approx(
        semi_major_axis * (1 - eccentricity)
    )


def test_rk4_matches_elliptical_kepler_solution() -> None:
    mass = 5.972e24
    semi_major_axis = 1.0e7
    eccentricity = 0.5
    inclination = 18.0
    periapsis = semi_major_axis * (1 - eccentricity)
    periapsis_speed = np.sqrt(
        G * mass * (1 + eccentricity) / (semi_major_axis * (1 - eccentricity))
    )
    period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (G * mass))
    planet = Planet(
        "elliptical",
        1.0,
        periapsis,
        periapsis_speed,
        inclination,
        "white",
        1,
    )
    central_position = np.zeros(3)
    initial_energy = planet.specific_energy(mass, central_position)
    initial_momentum = planet.specific_angular_momentum(central_position)
    for _ in range(1000):
        planet.update(period / 1000, mass, central_position, "rk4")

    expected_position, expected_velocity = analytical_kepler_state(
        mass, semi_major_axis, eccentricity, inclination, period
    )
    np.testing.assert_allclose(planet.position, expected_position, rtol=1e-8, atol=0.5)
    np.testing.assert_allclose(planet.velocity, expected_velocity, rtol=1e-8, atol=6e-4)
    assert planet.specific_energy(mass, central_position) == pytest.approx(
        initial_energy, rel=1e-9
    )
    np.testing.assert_allclose(
        planet.specific_angular_momentum(central_position),
        initial_momentum,
        rtol=1e-9,
        atol=1e-5,
    )
