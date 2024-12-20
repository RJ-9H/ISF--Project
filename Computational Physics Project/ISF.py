import numpy as np
import numpy.fft as fft
from Torus import TorusDEC
from typing import Tuple, Optional, List

class ISF(TorusDEC):
    #class for simulating incompressible 
    #schroedinger flow
    def __init__(self, sizex, sizey, sizez, resx, resy, resz):
        super().__init__(sizex, sizey, sizez, resx, resy, resz)
        self.hbar = None
        self.dt = None
        self.SchroedingerMask = None
    def BuildSchroedinger(self):
        nx = self.resx
        ny = self.resy 
        nz = self.resz 
        fac = -4 * (np.pi)**2 * self.hbar
        kx = (self.iix  - nx/2)/ self.sizex
        ky = (self.iiy - ny/2)/ self.sizey 
        kz = (self.iiz - nz/2) / self.sizez

        lambdaCoef = fac * (kx**2 + ky**2 + kz**2)
        self.SchroedingerMask = np.exp(1j * lambdaCoef * self.dt/2)

    def SchroedingerFlow(self, psi1, psi2):
        psi1_ft = fft.fftshift((fft.fftn(psi1)))
        psi2_ft = fft.fftshift((fft.fftn(psi2)))
        psi1_ft = psi1_ft * self.SchroedingerMask
        psi2_ft =  psi2_ft*self.SchroedingerMask
        psi1 = fft.ifftn(fft.fftshift(psi1_ft))
        psi2 = fft.ifftn(fft.fftshift(psi2_ft))
        return psi1, psi2

    def PressureProject(self, psi1: np.ndarray, psi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #Pressure projection of 2-component wave function.

        vx, vy, vz = self.VelocityOneForm(psi1, psi2)
        div = self.div(vx, vy, vz)  
        q = self.poisson_solve(div)  
        return self.gauge_transform(psi1, psi2, -q)
    
    def VelocityOneForm(self, psi1: np.ndarray, psi2: np.ndarray, hbar: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #Extracts velocity 1-form from (psi1, psi2).

        if hbar is None:
            hbar = 1.0
       
        ixp = (self.ix + 1) % self.resx
        iyp = (self.iy + 1) % self.resy
        izp = (self.iz + 1) % self.resz
        vx = np.angle(np.conj(psi1) * psi1[ixp, :, :] +
                      np.conj(psi2) * psi2[ixp, :, :]) * hbar
        
        vy = np.angle(np.conj(psi1) * psi1[:, iyp, :] +
                      np.conj(psi2) * psi2[:, iyp, :]) * hbar
        
        vz = np.angle(np.conj(psi1) * psi1[:, :, izp] +
                      np.conj(psi2) * psi2[:, :, izp]) * hbar
        
        return vx, vy, vz
    
    def add_circle(self, psi: np.ndarray, center: List[float], normal: List[float], r: float, d: float) -> np.ndarray:

        #Adds a vortex ring to a 1-component wave function psi.
        rx = self.px - center[0]
        ry = self.py - center[1]
        rz = self.pz - center[2]
        
        normal = np.array(normal) / np.linalg.norm(normal)
        alpha = np.zeros_like(rx)
        
        z = rx * normal[0] + ry * normal[1] + rz * normal[2]
        
        in_cylinder = rx**2 + ry**2 + rz**2 - z**2 < r**2
        in_layer_p = (z > 0) & (z <= d/2) & in_cylinder
        in_layer_m = (z <= 0) & (z >= -d/2) & in_cylinder
        
        alpha[in_layer_p] = -np.pi * (2 * z[in_layer_p] / d - 1)
        alpha[in_layer_m] = -np.pi * (2 * z[in_layer_m] / d + 1)
        
        return psi * np.exp(1j * alpha)
    
    @staticmethod
    def gauge_transform(psi1: np.ndarray, psi2: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #Multiplies exp(i*q) to (psi1, psi2).

        eiq = np.exp(1j * q)
        return psi1 * eiq, psi2 * eiq
    

    @staticmethod
    def Normalize(psi1: np.ndarray, psi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #Normalizes (psi1, psi2).

        psi_norm = np.sqrt(np.abs(psi1)**2 + np.abs(psi2)**2)
        return psi1 / psi_norm, psi2 / psi_norm
    