import numpy as np

class Particles:

    #Represents particles that can be advected by a staggered velocity
    #field on a TorusDEC grid using the RK4 method.

    
    def __init__(self):
  
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
    
    def staggered_advect(self, torus, vx, vy, vz, dt):
     
        # Compute RK4 stages
        k1x, k1y, k1z = self.staggered_velocity(
            self.x, self.y, self.z, 
            torus, vx, vy, vz
        )
        
        k2x, k2y, k2z = self.staggered_velocity(
            self.x + k1x * dt/2, 
            self.y + k1y * dt/2, 
            self.z + k1z * dt/2, 
            torus, vx, vy, vz
        )
        
        k3x, k3y, k3z = self.staggered_velocity(
            self.x + k2x * dt/2, 
            self.y + k2y * dt/2, 
            self.z + k2z * dt/2, 
            torus, vx, vy, vz
        )
        
        k4x, k4y, k4z = self.staggered_velocity(
            self.x + k3x * dt, 
            self.y + k3y * dt, 
            self.z + k3z * dt, 
            torus, vx, vy, vz
        )
        
        # Update positions using RK4 weighted average
        self.x += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        self.y += dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
        self.z += dt/6 * (k1z + 2*k2z + 2*k3z + k4z)
    
    def keep(self, ind):
        #Remove particles not in the specified indices

        self.x = self.x[ind]
        self.y = self.y[ind]
        self.z = self.z[ind]
    
    @staticmethod
    def staggered_velocity(px, py, pz, torus, vx, vy, vz):
        #Evaluate velocity at (px, py, pz) in the grid torus
        # Periodic wrapping of positions
        px = np.mod(px, torus.sizex)
        py = np.mod(py, torus.sizey)
        pz = np.mod(pz, torus.sizez)
        
        # Compute grid indices 
        ix = np.floor(px / torus.dx).astype(int)
        iy = np.floor(py / torus.dy).astype(int)
        iz = np.floor(pz / torus.dz).astype(int)
        
        # Periodic index shifts
        ixp = (ix + 1) % torus.resx
        iyp = (iy + 1) % torus.resy
        izp = (iz + 1) % torus.resz
        
        # Compute local coordinates within grid cell
        wx = px - ix * torus.dx
        wy = py - iy * torus.dy
        wz = pz - iz * torus.dz
        
        ux = (
            (1-wz) * ((1-wy) * vx[ix, iy, iz] + wy * vx[ix, iyp, iz]) +
            wz * ((1-wy) * vx[ix, iy, izp] + wy * vx[ix, iyp, izp])
        )
        
        uy = (
            (1-wz) * ((1-wx) * vy[ix, iy, iz] + wx * vy[ixp, iy, iz]) +
            wz * ((1-wx) * vy[ix, iy, izp] + wx * vy[ixp, iy, izp])
        )
        
        uz = (
            (1-wy) * ((1-wx) * vz[ix, iy, iz] + wx * vz[ixp, iy, iz]) +
            wy * ((1-wx) * vz[ix, iyp, iz] + wx * vz[ixp, iyp, iz])
        )
        
        return ux, uy, uz