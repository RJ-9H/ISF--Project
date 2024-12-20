import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ISF import ISF
from particles import Particles
def main():
    # PARAMETERS
    vol_size = [4, 2, 2]  # box size
    vol_res = [64, 32, 32]  # volume resolution
    hbar = 0.1  # Planck constant
    dt = 1 / 48  # time step
    tmax = 50  # max time

    jet_velocity = [1, 0, 0]  # jet velocity

    nozzle_cen = [2-1.7, 1-.034 , 1+.066]  # nozzle center
    nozzle_len = 0.5  # nozzle length
    nozzle_rad = 0.5  # nozzle radius

    n_particles = 50 # number of particles

    # INITIALIZATION
    isf = ISF(*vol_size, *vol_res)
    isf.hbar = hbar
    isf.dt = dt
    isf.BuildSchroedinger()

    # Set nozzle
    isJet = ((np.abs(isf.px - nozzle_cen[0]) <= nozzle_len / 2) &
             ((isf.py - nozzle_cen[1]) ** 2 + (isf.pz - nozzle_cen[2]) ** 2 <= nozzle_rad ** 2))

    # Initialize psi
    psi1 = np.ones_like(isf.px, dtype = 'complex')
    psi2 = 0.01 * psi1
    psi1, psi2 = ISF.Normalize(psi1, psi2)

    # Constrain velocity
    kvec = np.array(jet_velocity) / isf.hbar
    omega = np.sum(np.array(jet_velocity) ** 2) / (2 * isf.hbar)
    phase = (kvec[0] * isf.px +
             kvec[1] * isf.py +
             kvec[2] * isf.pz)

    # Initial velocity constraint iteration
    for _ in range(10):
        amp1 = np.abs(psi1)
        amp2 = np.abs(psi2)
        psi1[isJet] = amp1[isJet] * np.exp(1j * phase[isJet])
        psi2[isJet] = amp2[isJet] * np.exp(1j * phase[isJet])
        psi1, psi2 = isf.PressureProject(psi1, psi2)

    # SET PARTICLES
    particle = Particles()
    particle.x = np.array([0], dtype=float)
    particle.y = np.array([0], dtype=float)
    particle.z = np.array([0], dtype=float)

    # Setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(particle.x, particle.y, particle.z, s = 1)
    ax.set_xlim(0, vol_size[0])
    ax.set_ylim(0, vol_size[1])
    ax.set_zlim(0, vol_size[2])
    plt.ion()
    plt.show()

    # MAIN ITERATION
    itermax = int(np.ceil(tmax / dt))
    for iter in range(itermax):
        t = iter * dt

        # Incompressible Schroedinger flow
        psi1, psi2 = isf.SchroedingerFlow(psi1, psi2)
        psi1, psi2 = isf.Normalize(psi1, psi2)
        psi1, psi2 = isf.PressureProject(psi1, psi2)

        # Constrain velocity
        phase = (kvec[0] * isf.px +
                 kvec[1] * isf.py +
                 kvec[2] * isf.pz -
                 omega * t)

        amp1 = np.abs(psi1)
        amp2 = np.abs(psi2)
        psi1[isJet] = amp1[isJet] * np.exp(1j * phase[isJet])
        psi2[isJet] = amp2[isJet] * np.exp(1j * phase[isJet])
        psi1, psi2 = isf.PressureProject(psi1, psi2)

        # Particle birth
        rt = np.random.rand(n_particles) * 2 * np.pi
        newx = nozzle_cen[0] * np.ones_like(rt)
        newy = nozzle_cen[1] + 0.9 * nozzle_rad * np.cos(rt)
        newz = nozzle_cen[2] + 0.9 * nozzle_rad * np.sin(rt)

        particle.x = np.concatenate([particle.x, newx])
        particle.y = np.concatenate([particle.y, newy])
        particle.z = np.concatenate([particle.z, newz])

        # Advect and show particles
        vx, vy, vz = isf.VelocityOneForm(psi1, psi2, isf.hbar)
        vx, vy, vz = isf.staggered_sharp(vx, vy, vz)


        particle.staggered_advect(isf, vx, vy, vz, isf.dt)

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
    main()