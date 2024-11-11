import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1.0                # Length of the domain
Nx = 50                # Number of cells
dx = L / Nx            # Cell size
c = 1.0                # Advection speed

# Define time step for different Courant numbers
CFL_stable = 0.5
CFL_unstable = 2.0

# Function to initialize a square wave
def initialize_u(Nx):
    u = np.zeros(Nx)
    u[Nx // 4: Nx // 2] = 1.0  # Initial condition: square wave in the middle
    return u

# Upwind scheme function
def upwind_scheme(u, c, dx, dt):
    unew = np.copy(u)
    for i in range(1, len(u)):
        unew[i] = u[i] - (c * dt / dx) * (u[i] - u[i - 1])
    return unew

# Simulation function
def simulate_advection(CFL, Nx, c, dx, steps):
    dt = CFL * dx / c
    u = initialize_u(Nx)
    results = [u.copy()]  # Store initial condition
    for _ in range(steps):
        u = upwind_scheme(u, c, dx, dt)
        results.append(u.copy())
    return results

# Simulation parameters
steps = 25  # Number of time steps to simulate

# Run simulations for stable and unstable cases
results_stable = simulate_advection(CFL_stable, Nx, c, dx, steps)
results_unstable = simulate_advection(CFL_unstable, Nx, c, dx, steps)

# Plot results for both stable and unstable cases
def plot_results(results, title):
    plt.figure(figsize=(12, 6))
    for i, u in enumerate(results[::5]):
        plt.plot(np.linspace(0, L, Nx), u, label=f'Step {i * 5}')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()

plot_results(results_stable, "Advection with CFL = 0.5 (Stable)")
plot_results(results_unstable, "Advection with CFL = 2.0 (Unstable)")
