import numpy as np
from Torus import TorusDEC

class LaxFriedrichsFlow(TorusDEC):
    def __init__(self, sizex, sizey, sizez, resx, resy, resz):
        super().__init__(sizex, sizey, sizez, resx, resy, resz)
        self.dt = None

    def LaxFriedrichsAdvection(self, field, vx, vy, vz):
        # Compute first-order derivatives
        dphi_dx, dphi_dy, dphi_dz = self.derivativeofFunction(field)
        dphi_dx /= self.dx
        dphi_dy /= self.dy
        dphi_dz /= self.dz

        # Compute fluxes (central difference approximation)
        flux_x = 0.5 * (field + self.shift(field, axis=0)) * vx
        flux_y = 0.5 * (field + self.shift(field, axis=1)) * vy
        flux_z = 0.5 * (field + self.shift(field, axis=2)) * vz

        # Compute Lax-Friedrichs term (artificial viscosity)
        lax_term = -0.5 * self.dt * (
            (self.shift(field, axis=0) - field) / self.dx +
            (self.shift(field, axis=1) - field) / self.dy +
            (self.shift(field, axis=2) - field) / self.dz
        )

        # Update field
        field += self.dt * (-dphi_dx * vx - dphi_dy * vy - dphi_dz * vz) + flux_x + flux_y + flux_z -lax_term

        return field

    def shift(self, field, axis):
        """
        Shift the field by one step along the given axis using parent methods.
        """
        shifted_field = np.empty_like(field)
        if axis == 0:  # x-axis
            shifted_field = np.roll(field, shift=1, axis=0)
        elif axis == 1:  # y-axis
            shifted_field = np.roll(field, shift=1, axis=1)
        elif axis == 2:  # z-axis
            shifted_field = np.roll(field, shift=1, axis=2)
        return shifted_field

    def enforce_incompressibility(self, vx, vy, vz):
        """
        Project the velocity field to be incompressible using parent class methods.
        """
        # Compute divergence of velocity field
        div_v = self.div(vx, vy, vz)

        # Solve Poisson equation for pressure
        p = np.real(self.poisson_solve(div_v))

        # Compute gradient of pressure using parent method
        dp_dx, dp_dy, dp_dz = self.derivativeofFunction(p)
        dp_dx /= self.dx
        dp_dy /= self.dy
        dp_dz /= self.dz

        # Subtract pressure gradient to update velocities
        vx -= dp_dx
        vy -= dp_dy
        vz -= dp_dz

        return vx, vy, vz
