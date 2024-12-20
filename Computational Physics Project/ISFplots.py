import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ISF import ISF
from particles import Particles

# PARAMETERS
vol_size = [4, 2, 2]  # box size
vol_res = [64, 32, 32]  # volume resolution
hbar = 0.1  # Planck constant
dt = 1 / 48  # time step
tmax = 50  # max time

jet_velocity = [1, 0, 0]  # jet velocity

nozzle_cen = [2 - 1.7, 1 - 0.034, 1 + 0.066]  # nozzle center
nozzle_len = 0.5  # nozzle length
nozzle_rad = 0.5  # nozzle radius

n_particles = 50  # number of particles

# INITIALIZATION
isf = ISF(*vol_size, *vol_res)
isf.hbar = hbar
isf.dt = dt
isf.BuildSchroedinger()

# Set nozzle
isJet = ((np.abs(isf.px - nozzle_cen[0]) <= nozzle_len / 2) &
         ((isf.py - nozzle_cen[1]) ** 2 + (isf.pz - nozzle_cen[2]) ** 2 <= nozzle_rad ** 2))

# Initialize psi
psi1 = np.ones_like(isf.px, dtype='complex')
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

# Compute velocity field
vx, vy, vz = isf.VelocityOneForm(psi1, psi2, isf.hbar)
vx, vy, vz = isf.staggered_sharp(vx, vy, vz)

# Generate velocity contour plot for xy slice
midplane_idx = vol_res[2] // 2  # Index for the midplane in z-direction
velocity_magnitude = np.sqrt(vx[:, :, midplane_idx]**2 + vy[:, :, midplane_idx]**2)

x = np.linspace(0, vol_size[0], vol_res[0])
y = np.linspace(0, vol_size[1], vol_res[1])
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, velocity_magnitude.T, cmap='viridis')
plt.colorbar(contour, label='Velocity Magnitude')
plt.title('Velocity Contour in XY-plane at z = Midplane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Vorticity calculation function
def compute_vorticity(vx, vy, vz, dx, dy, dz):
    dvz_dy, dvz_dx = np.gradient(vz, dy, dx, axis=(1, 0))
    dvy_dz, dvy_dx = np.gradient(vy, dz, dx, axis=(2, 0))
    dvx_dz, dvx_dy = np.gradient(vx, dz, dy, axis=(2, 1))

    wx = dvy_dz - dvz_dy
    wy = dvz_dx - dvx_dz
    wz = dvx_dy - dvy_dx

    return wx, wy, wz

# Generate grid
x = np.linspace(0, vol_size[0], vol_res[0])
y = np.linspace(0, vol_size[1], vol_res[1])
z = np.linspace(0, vol_size[2], vol_res[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Compute vorticity
dx, dy, dz = vol_size[0] / vol_res[0], vol_size[1] / vol_res[1], vol_size[2] / vol_res[2]
wx, wy, wz = compute_vorticity(vx, vy, vz, dx, dy, dz)
vorticity_magnitude = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)

# 1. 3D Quiver Plot of Velocity
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
subsample = slice(None, None, 4)  # Subsample for clarity
ax.quiver(X[subsample], Y[subsample], Z[subsample],
          vx[subsample], vy[subsample], vz[subsample],
          length=0.5, normalize=True, color='blue', alpha=.1)
ax.set_title('3D Velocity Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 2. Vorticity Contour in the Midplane
midplane_idx = vol_res[2] // 2
plt.figure(figsize=(8, 6))
contour = plt.contourf(X[:, :, midplane_idx], Y[:, :, midplane_idx],
                       vorticity_magnitude[:, :, midplane_idx], cmap='plasma')
plt.colorbar(contour, label='Vorticity Magnitude')
plt.title('Vorticity Contour in XY-plane at z = Midplane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 3. Phase Distribution Heatmap
phase_psi1 = np.angle(psi1)
plt.figure(figsize=(8, 6))
plt.contourf(X[:, :, midplane_idx], Y[:, :, midplane_idx],
             phase_psi1[:, :, midplane_idx], cmap='twilight')
plt.colorbar(label='Phase (radians)')
plt.title('Phase Distribution of Psi1 in XY-plane at z = Midplane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 4. Density Distribution
density = np.abs(psi1) ** 2 + np.abs(psi2) ** 2
plt.figure(figsize=(8, 6))
plt.contourf(X[:, :, midplane_idx], Y[:, :, midplane_idx],
             density[:, :, midplane_idx], cmap='inferno')
plt.colorbar(label='Density')
plt.title('Density Distribution in XY-plane at z = Midplane')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
