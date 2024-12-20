import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Euler import LaxFriedrichsFlow  # Import the class with Lax-Friedrichs advection
from particles import Particles

def LaxFriedrichsSimulation():
    # PARAMETERS
    vol_size = [4, 2, 2]  # box size
    vol_res = [64, 32, 32]  # volume resolution
    dt = 1 / 48  # time step
    tmax = 50  # max time

    jet_velocity = [1, 0, 0]  # jet velocity

    nozzle_cen = [2 - 1.7, 1 - 0.034, 1 + 0.066]  # nozzle center
    nozzle_len = 0.5  # nozzle length
    nozzle_rad = 0.5  # nozzle radius

    n_particles = 50  # number of particles

    # INITIALIZATION
    lax = LaxFriedrichsFlow(*vol_size, *vol_res)
    lax.dt = dt

    # Initialize scalar field
    psi = np.ones_like(lax.px, dtype='float')

    # Velocity field (jet velocity)
    vx = np.full_like(psi, jet_velocity[0])
    vy = np.full_like(psi, jet_velocity[1])
    vz = np.full_like(psi, jet_velocity[2])

    # PARTICLES
    particle = Particles()
    particle.x = np.array([0], dtype=float)
    particle.y = np.array([0], dtype=float)
    particle.z = np.array([0], dtype=float)

    # Setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(particle.x, particle.y, particle.z, s=1)
    ax.set_xlim(0, vol_size[0])
    ax.set_ylim(0, vol_size[1])
    ax.set_zlim(0, vol_size[2])
    plt.ion()
    plt.show()

    # MAIN ITERATION
    itermax = int(np.ceil(tmax / dt))
    for iter in range(itermax):
        t = iter * dt

        # Lax-Friedrichs advection
        psi = lax.LaxFriedrichsAdvection(psi, vx, vy, vz)
        vx, vy, vz = lax.enforce_incompressibility(vx, vy, vz)

        # Particle birth
        rt = np.random.rand(n_particles) * 2 * np.pi
        newx = nozzle_cen[0] * np.ones_like(rt)
        newy = nozzle_cen[1] + 0.9 * nozzle_rad * np.cos(rt)
        newz = nozzle_cen[2] + 0.9 * nozzle_rad * np.sin(rt)

        particle.x = np.concatenate([particle.x, newx])
        particle.y = np.concatenate([particle.y, newy])
        particle.z = np.concatenate([particle.z, newz])

        # Advect particles
        particle.staggered_advect(lax, vx, vy, vz, dt)

        # Keep particles within domain
        mask = ((particle.x > 0) & (particle.x < vol_size[0]) &
                (particle.y > 0) & (particle.y < vol_size[1]) &
                (particle.z > 0) & (particle.z < vol_size[2]))
        particle.keep(mask)

        # Update plot
        scatter._offsets3d = (particle.x, particle.y, particle.z)
        plt.title(f'Iteration = {iter}')
        plt.show()
        plt.pause(0.01)

if __name__ == "__main__":
    LaxFriedrichsSimulation()
