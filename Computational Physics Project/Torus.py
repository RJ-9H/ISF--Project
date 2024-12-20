import numpy as np 
import scipy.fft as fft

class TorusDEC:
    def __init__(self, sizex, sizey, sizez, resx, resy, resz):
        #Size is size of the grid 
        #res is number of grid points in each dimension 
        self.sizex, self.sizey, self.sizez = sizex, sizey, sizez
        self.resx, self.resy, self.resz =round(resx), round(resy), round(resz) 

        #dx, dy, dz is the edge length 
        self.dx = self.sizex/self.resx
        self.dy = self.sizey/self.resy
        self.dz = self.sizez /self.resz

        #ix, iy, iz is the 1-D index array 
        self.ix = np.arange(self.resx)
        self.iy = np.arange(self.resy)
        self.iz = np.arange(self.resz)

        #3D index array 
        self.iix, self.iiy, self.iiz = np.meshgrid(self.ix, self.iy, self.iz, indexing= 'ij')
        self.px = (self.iix )*self.dx 
        self.py = (self.iiy )*self.dy
        self.pz = (self.iiz)*self.dz
    def derivativeofFunction(self, f):
        ixp = (self.ix +1) % self.resx
        iyp = (self.iy +1) % self.resy
        izp = (self.iz +1) % self.resz
        vx = f[ixp, :, :] - f
        vy = f[:, iyp, :] - f
        vz = f[:, :, izp] - f
        return vx,vy,vz
    #1 form to 2 form differential
    def derivative_of_one_form(self,vx, vy, vz):
        ixp = (self.ix +1) % self.resx
        iyp = (self.iy +1) % self.resy
        izp = (self.iz +1) % self.resz

        wx =vy- vy[:, :, izp] + vz[:, iyp, :] - vz
        wy =  vz -  vz[ixp, :, :]  + vx[:, :, vz] - vz
        wz = vx - vx[:, iyp, :] + vy[ixp,:,:] - vy
        return wx, wy, wz

    # 2-form to 3-form
    def derivative_of_two_form(self, wx, wy, wz):
        ixp = (self.ix +1) % self.resx
        iyp = (self.iy +1) % self.resy
        izp = (self.iz +1) % self.resz

        f = wx[ixp, : , : ] -wx
        f+= wy[ : , iyp, : ] -wy
        f+= wz[:, :, izp] -wz
        return f 

    def div(self, vx, vy, vz):
        ixm = ((self.ix -1 ) % self.resx) 
        iym = ((self.iy -1 ) % self.resy) 
        izm = ((self.iz -1) % self.resz) 
        f = (vx - vx[ixm, :, :]) / (self.dx**2)
        f += (vy - vy[:, iym, :]) / (self.dy**2)
        f += (vz - vz[:, :, izm]) / (self.dz**2)
        return f

    def sharp(self, vx, vy, vz):
        ixm = ((self.ix -1 ) % self.resx) 
        iym = ((self.iy -1 ) % self.resy) 
        izm = ((self.iz -1) % self.resz)

        ux = 0.5 * (vx[ixm, :, :] + vx) / self.dx
        uy = 0.5 * (vy[:, iym, :] + vy) / self.dy
        uz = 0.5 * (vz[:, :, izm] + vz) / self.dz
        return ux, uy, uz

    # Staggered Sharp Operator
    def staggered_sharp(self, vx, vy, vz):
        ux = vx/self.dx
        uy = vy/self.dy
        uz = vz/self.dz
        return ux, uy, uz

    # Poisson Solver
    def poisson_solve(self, f):
        f = fft.fftn(f)
        sx = np.sin(np.pi * self.iix / self.resx) / self.dx
        sy = np.sin(np.pi * self.iiy / self.resy) / self.dy
        sz = np.sin(np.pi * self.iiz / self.resz) / self.dz

        denom = sx ** 2 + sy ** 2 + sz ** 2
        denom[0, 0, 0] = .1  # Avoid division by zero
        f_hat = -.25/denom
        f_hat[0, 0, 0] = 0  # Ensure zero mean
        f = f*f_hat
        f = fft.ifftn(f)
        return f
