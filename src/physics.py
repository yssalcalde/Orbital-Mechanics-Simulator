from __future__ import annotations

from typing import Final

import numpy as np

G: Final = 6.67430e-11

class Planet:
    """Point-mass body orbiting a fixed central mass."""

    def __init__(
        self,
        name: str,
        mass: float,
        distance: float,
        velocity: float,
        inclination: float,
        color: str,
        size: int,
    ) -> None:
        if mass <= 0:
            raise ValueError("Planet mass must be positive.")
        if distance <= 0:
            raise ValueError("Initial distance must be positive.")

        self.name = name
        self.mass = float(mass)
        self.color = color
        self.size = int(size)

        self.inclination = np.radians(float(inclination))
        self.position = np.array([distance, 0.0, 0.0], dtype=np.float64)
        self.velocity = np.array([
            0.0,
            velocity * np.cos(self.inclination),
            velocity * np.sin(self.inclination)
        ], dtype=np.float64)
        self.trajectory: list[np.ndarray] = [self.position.copy()]

    def compute_acceleration(
        self,
        position: np.ndarray,
        central_mass: float,
        central_position: np.ndarray,
    ) -> np.ndarray:
        """Return gravitational acceleration at ``position``."""
        r_vector = position - central_position
        r_norm = np.linalg.norm(r_vector)
        if r_norm == 0:
            return np.zeros(3, dtype=np.float64)
        return -G * central_mass * r_vector / r_norm**3

    def specific_energy(
        self, central_mass: float, central_position: np.ndarray
    ) -> float:
        """Return specific orbital energy, ``v²/2 - GM/r``."""

        v_sq = np.dot(self.velocity, self.velocity)
        r_vector = self.position - central_position
        r_norm = np.linalg.norm(r_vector)

        if r_norm == 0:
            raise ValueError("Specific energy is undefined at zero distance.")

        return 0.5 * v_sq - G * central_mass / r_norm
    def specific_angular_momentum(self, central_position: np.ndarray) -> np.ndarray:
        """Return the specific angular-momentum vector, ``r × v``."""
        r_vector = self.position - central_position
        return np.cross(r_vector, self.velocity)

    def update(
        self,
        dt: float,
        central_mass: float,
        central_position: np.ndarray,
        method: str = "rk4",
    ) -> None:
        """Advance one step using the selected numerical integration method."""
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
        if central_mass <= 0:
            raise ValueError("Central mass must be positive.")

        method= method.lower()
        if method == "euler":
            self.update_euler(dt, central_mass, central_position)
        elif method == "euler_implicit":
            self.update_euler_implicit(dt, central_mass, central_position)
        elif method == "rk4":
            self.update_rk4(dt, central_mass, central_position)
        elif method == "verlet":
            self.update_verlet(dt, central_mass, central_position)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'euler', 'euler_implicit', 'rk4', or 'verlet'.")

        self.trajectory.append(self.position.copy())
    def update_euler(
        self, dt: float, central_mass: float, central_position: np.ndarray
    ) -> None:
        """Advance with first-order explicit Euler."""
        acceleration = self.compute_acceleration(self.position, central_mass, central_position)
        self.position += self.velocity * dt
        self.velocity += acceleration * dt

    def update_euler_implicit(
        self, dt: float, central_mass: float, central_position: np.ndarray
    ) -> None:
        """Implicit Euler solved via Newton-Raphson using an analytical 6x6 Jacobian matrix."""
        def equations(state):
            pos = state[:3]
            vel = state[3:]
            acc = self.compute_acceleration(pos, central_mass, central_position)

            eq_pos = pos -(self.position + vel * dt)
            eq_vel = vel -(self.velocity + acc * dt)
            return np.concatenate((eq_pos, eq_vel))

        def jacobian(state):
            pos= state[:3]
            r_vec = pos - central_position
            r_norm = np.linalg.norm(r_vec)
            if r_norm == 0:
                return np.zeros((6, 6), dtype=np.float64)

            #Analitic derivative d(acc)/d(pos)
            d_acc_d_pos = -G * central_mass * (np.eye(3) / (r_norm ** 3) - 3 * np.outer(r_vec, r_vec) / (r_norm ** 5))

            #Build the Jacobian matrix 6x6:

            jac = np.zeros((6, 6), dtype=np.float64)
            jac[:3,:3] = np.eye(3)
            jac[:3,3:] =  -dt *np.eye(3)
            jac[3:,:3] = -dt * d_acc_d_pos
            jac[3:,3:] = np.eye(3)
            return jac
        state = np.concatenate((self.position, self.velocity))
        converged = False
        for _ in range(12):  # Newton-Raphson iterations
            f_val = equations(state)
            j_mat = jacobian(state)
            delta = np.linalg.solve(j_mat, -f_val)
            state += delta
            if np.linalg.norm(delta) < 1e-10:
                converged = True
                break
        if not converged:
            raise RuntimeError("Implicit Euler Newton-Raphson iteration did not converge.")
        self.position = state[:3]
        self.velocity = state[3:]

    def update_rk4(
        self, dt: float, central_mass: float, central_position: np.ndarray
    ) -> None:
        """Fourth-order classical Runge-Kutta integrator."""
        k1_x = self.velocity
        k1_v = self.compute_acceleration(self.position, central_mass, central_position)

        k2_x = self.velocity + 0.5 * dt * k1_v
        k2_v = self.compute_acceleration(self.position + 0.5 * dt * k1_x, central_mass, central_position)

        k3_x = self.velocity + 0.5 * dt * k2_v
        k3_v = self.compute_acceleration(self.position + 0.5 * dt * k2_x, central_mass, central_position)

        k4_x = self.velocity + dt * k3_v
        k4_v = self.compute_acceleration(self.position + dt * k3_x, central_mass, central_position)

        self.position += (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        self.velocity += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    def update_verlet(
        self, dt: float, central_mass: float, central_position: np.ndarray
    ) -> None:
        """Second-order Velocity Verlet integrator (symplectic, preserves phase space volume)."""

        acceleration = self.compute_acceleration(self.position, central_mass, central_position)
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        new_acceleration = self.compute_acceleration(self.position, central_mass, central_position)
        self.velocity += 0.5 * (acceleration + new_acceleration) * dt

def circular_orbital_velocity(central_mass: float, distance: float) -> float:
    """Calculate the circular orbital velocity for a given central mass and distance."""
    if central_mass <= 0:
        raise ValueError("Central mass must be positive.")
    if distance <= 0:
        raise ValueError("Orbital distance must be positive.")
    return np.sqrt(G * central_mass / distance)


def analytical_circular_state(
    central_mass: float,
    distance: float,
    inclination: float,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact position and velocity for a circular orbit.

    The initial state is at ``(distance, 0, 0)`` with the velocity direction
    used by :class:`Planet`. ``inclination`` is expressed in degrees.
    """
    if central_mass <= 0:
        raise ValueError("Central mass must be positive.")
    if distance <= 0:
        raise ValueError("Orbital distance must be positive.")
    if time < 0:
        raise ValueError("Time must not be negative.")

    inclination_rad = np.radians(float(inclination))
    speed = circular_orbital_velocity(central_mass, distance)
    angular_rate = speed / distance
    angle = angular_rate * time
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    cos_inclination, sin_inclination = (
        np.cos(inclination_rad),
        np.sin(inclination_rad),
    )
    position = distance * np.array(
        [cos_angle, sin_angle * cos_inclination, sin_angle * sin_inclination]
    )
    velocity = speed * np.array(
        [-sin_angle, cos_angle * cos_inclination, cos_angle * sin_inclination]
    )
    return position, velocity


def analytical_kepler_state(
    central_mass: float,
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an elliptical Kepler orbit at time ``time``.

    The orbit starts at periapsis on the x-axis. ``inclination`` rotates the
    perifocal y-axis into the y-z plane, matching :class:`Planet`.
    """
    if central_mass <= 0:
        raise ValueError("Central mass must be positive.")
    if semi_major_axis <= 0:
        raise ValueError("Semi-major axis must be positive.")
    if not 0 <= eccentricity < 1:
        raise ValueError("Eccentricity must satisfy 0 <= e < 1.")
    if time < 0:
        raise ValueError("Time must not be negative.")

    mu = G * central_mass
    mean_motion = np.sqrt(mu / semi_major_axis**3)
    mean_anomaly = mean_motion * time
    eccentric_anomaly = mean_anomaly
    for _ in range(20):
        correction = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if abs(correction) < 1e-14:
            break

    cos_e, sin_e = np.cos(eccentric_anomaly), np.sin(eccentric_anomaly)
    radius = semi_major_axis * (1 - eccentricity * cos_e)
    position = np.array(
        [
            semi_major_axis * (cos_e - eccentricity),
            semi_major_axis * np.sqrt(1 - eccentricity**2) * sin_e,
            0.0,
        ]
    )
    velocity_scale = np.sqrt(mu * semi_major_axis) / radius
    velocity = velocity_scale * np.array(
        [-sin_e, np.sqrt(1 - eccentricity**2) * cos_e, 0.0]
    )
    inclination_rad = np.radians(float(inclination))
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(inclination_rad), -np.sin(inclination_rad)],
            [0.0, np.sin(inclination_rad), np.cos(inclination_rad)],
        ]
    )
    return rotation @ position, rotation @ velocity
