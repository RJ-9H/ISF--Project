import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Euler import LaxFriedrichsFlow  # Use LaxFriedrichsFlow for flow
from particles import Particles  # Import Particles class

# Vorticity calculation function
def compute_vorticity(vx, vy, vz, dx, dy, dz):
    dvz_dy, dvz_dx = np.gradient(vz, dy, dx, axis=(1, 0))
    dvy_dz, dvy_dx = np.gradient(vy, dz, dx, axis=(2, 0))
    dvx_dz, dvx_dy = np.gradient(vx, dz, dy, axis=(2, 1))
    wx = dvy_dz - dvz_dy
    wy = dvz_dx - dvx_dz
    wz = dvx_dy - dvy_dx
    return wx, wy, wz

# PARAMETERS
vol_size = [4, 2, 2]  # box size
vol_res = [64, 32, 32]  # volume resolution
dt = 1 / 48  # time step

jet_velocity = [1, 0, 0]  # jet velocity
nozzle_cen = [2 - 1.7, 1 - 0.034, 1 + 0.066]  # nozzle center
nozzle_len = 0.5  # nozzle length
nozzle_rad = 0.5  # nozzle radius
n_particles = 50  # number of particles

# INITIALIZATION
lax = LaxFriedrichsFlow(*vol_size, *vol_res)
lax.dt = dt
psi = np.ones(lax.px.shape, dtype='float')

# Velocity field (jet velocity)
vx = np.full_like(psi, jet_velocity[0])
vy = np.full_like(psi, jet_velocity[1])
vz = np.full_like(psi, jet_velocity[2])

# Generate grid
x = np.linspace(0, vol_size[0], vol_res[0])
y = np.linspace(0, vol_size[1], vol_res[1])
z = np.linspace(0, vol_size[2], vol_res[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Compute vorticity
dx, dy, dz = vol_size[0] / vol_res[0], vol_size[1] / vol_res[1], vol_size[2] / vol_res[2]
wx, wy, wz = compute_vorticity(vx, vy, vz, dx, dy, dz)
vorticity_magnitude = np.sqrt(wx**2 + wy**2 + wz**2)

# Velocity Field (3D Quiver Plot)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
subsample = slice(None, None, 4)
ax.quiver(X[subsample], Y[subsample], Z[subsample],
          vx[subsample], vy[subsample], vz[subsample],
          length=0.5, normalize=True, color='blue')
ax.set_title('3D Velocity Field (Lax-Friedrichs)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Vorticity Contour in Midplane
midplane_idx = vol_res[2] // 2
plt.figure(figsize=(8, 6))
contour = plt.contourf(X[:, :, midplane_idx], Y[:, :, midplane_idx],
                       vorticity_magnitude[:, :, midplane_idx], cmap='plasma')
plt.colorbar(contour, label='Vorticity Magnitude')
plt.title('Vorticity Contour in XY-plane at z = Midplane (Lax-Friedrichs)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Velocity Contour in Midplane (X and Y velocities)
plt.figure(figsize=(8, 6))
contour_vx = plt.contourf(X[:, :, midplane_idx], Y[:, :, midplane_idx],
                          vx[:, :, midplane_idx], cmap='viridis')
plt.colorbar(contour_vx, label='Velocity in X direction')
plt.title('Velocity Contour in XY-plane at z = Midplane (X-component)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Optionally, add contours for other velocity components (vy, vz) if needed
