import numpy as np

G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2

class Planet:
    def __init__(self, name, mass, distance, velocity, inclination, color, size):
        if mass <= 0:
            raise ValueError("Planet mass must be positive.")
        if distance <= 0:
            raise ValueError("Initial distance must be positive.")

        self.name = name
        self.mass = float(mass)
        self.color = color
        self.size = int(size)

        # Inclination in degrees, converted to radians for calculations
        self.inclination = np.radians(float(inclination))

        #Initial position in the nodal place X axis ; ||r||=distance
        self.position= np.array([distance, 0.0,0.0], dtype=np.float64)

        #Orbital velocity with exact rotation in the YZ plane ; ||v|| = velocity
        self.velocity = np.array([
            0.0,
            velocity * np.cos(self.inclination),
            velocity * np.sin(self.inclination)
        ], dtype=np.float64)

        self.trajectory = [self.position.copy()]  # Store the trajectory of the planet

    def compute_acceleration(self, position, central_mass, central_position):
        # Calculate the vector from the planet to the central mass
        r_vector = position - central_position
        r_norm = np.linalg.norm(r_vector)


        if r_norm == 0:
            return np.zeros(3, dtype=np.float64)  # Avoid division by zero if the planet is at the central mass position
        return -G * central_mass*r_vector / (r_norm ** 3)
    def specific_energy(self, central_mass, central_position):
        '''E = K + U = 1/2 * ||v||^2 - G * M / ||r||'''

        v_sq = np.dot(self.velocity, self.velocity)
        r_vector = self.position - central_position
        r_norm = np.linalg.norm(r_vector)

        if r_norm == 0:
            raise ValueError("Specific energy is undefined at zero distance.")

        return 0.5 * v_sq - G * central_mass / r_norm
    def  specific_angular_momentum(self, central_position):
        '''h = r x v'''

        r_vector = self.position - central_position
        return np.cross(r_vector, self.velocity)

    def update (self, dt, central_mass, central_position, method ="rk4"):
        '''Updaate a dt step using the specified algorithm :'euler', 'euler_implicit', 'rk4' or 'verlet' '''
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
            raise ValueError("Unknown method: {}. Choose from 'euler', 'euler_implicit', 'rk4', or 'verlet'.".format(method))

        self.trajectory.append(self.position.copy())
    ####Add each method

    def update_euler(self, dt, central_mass, central_position):
        '''Update the planet's position and velocity using the explicit Euler method.First-order Explicit Euler (energy non-conserving, artificially expands orbit).'''
        acceleration = self.compute_acceleration(self.position, central_mass, central_position)
        self.position += self.velocity * dt
        self.velocity += acceleration * dt

    def update_euler_implicit(self, dt, central_mass, central_position):
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

    def update_rk4(self, dt, central_mass, central_position):
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

    def update_verlet(self, dt, central_mass, central_position):
        """Second-order Velocity Verlet integrator (symplectic, preserves phase space volume)."""

        acceleration = self.compute_acceleration(self.position, central_mass, central_position)
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        new_acceleration = self.compute_acceleration(self.position, central_mass, central_position)
        self.velocity += 0.5 * (acceleration + new_acceleration) * dt

def circular_orbital_velocity(central_mass, distance):
    """Calculate the circular orbital velocity for a given central mass and distance."""
    if central_mass <= 0:
        raise ValueError("Central mass must be positive.")
    if distance <= 0:
        raise ValueError("Orbital distance must be positive.")
    return np.sqrt(G * central_mass / distance)
