"""Audited existing FFT grid plus matched Gaussian restriction and right inverse."""
import math

import torch

from research.skew_full_validation_20261020.core import Grid


class GaussianRestriction:
    """C=P_low transfer G_delta; L retains only low modes, with C L=I there.

    The lift is a declared intervention, NOT recovery of the true fine state.
    It holds all modes in ker(C) fixed. Strong Gaussian inversion is rejected.
    """
    def __init__(self, high, low, delta, max_lift_gain=100.):
        if high.length != low.length or high.n <= low.n or delta <= 0:
            raise ValueError("matching domains, high.n>low.n, delta>0 required")
        if high.dtype != low.dtype or high.device != low.device:
            raise ValueError("grid dtype/device must match")
        self.high, self.low, self.delta = high, low, float(delta)
        self.filter = torch.exp(-delta * delta * high.k2 / 24)
        exponent = delta * delta * low.k2 / 24
        self.lift_gain = float(torch.exp(exponent[low.mask]).max())
        if not math.isfinite(self.lift_gain) or self.lift_gain > max_lift_gain:
            raise ValueError("ILL_CONDITIONED_GAUSSIAN_LIFT")
        self.inverse_filter = torch.exp(exponent) * low.mask

    def restrict(self, z):
        return self.low.project(self.high.transfer(
            self.high.ifft(self.high.fft(z) * self.filter), self.low))

    def lift(self, eta):
        x = self.low.ifft(self.low.fft(eta) * self.inverse_filter)
        return self.high.project(self.low.transfer(x, self.high))

    def hidden(self, z):
        return z - self.lift(self.restrict(z))


def rk4_step(rhs, x, dt):
    if dt <= 0:
        raise ValueError("positive dt required")
    k1 = rhs(x)
    k2 = rhs(x + dt * k1 / 2)
    k3 = rhs(x + dt * k2 / 2)
    k4 = rhs(x + dt * k3)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class VorticityStepper:
    """Independent wrapper: original grid/RHS, optional fixed vorticity forcing."""
    def __init__(self, grid, nu, dt, source=None, forcing=None):
        if nu <= 0 or dt <= 0:
            raise ValueError("nu, dt must be positive")
        self.grid, self.nu, self.dt, self.source = grid, nu, dt, source
        self.forcing = torch.zeros((grid.n, grid.n), device=grid.device, dtype=grid.dtype) \
            if forcing is None else grid.project(forcing)

    def rhs(self, x):
        return self.grid.rhs(x, self.nu, self.source) + self.forcing

    def __call__(self, x):
        return self.grid.project(rk4_step(self.rhs, x, self.dt))


def kolmogorov_vorticity_forcing(grid, amplitude, mode):
    """f_omega=A cos(2pi*m*y/L); amplitude units are s^-2."""
    if mode <= 0 or mode > grid.cut:
        raise ValueError("forcing mode outside retained spectrum")
    y = torch.arange(grid.n, device=grid.device, dtype=grid.dtype) * grid.length / grid.n
    return grid.project(amplitude * torch.cos(2 * math.pi * mode * y[:, None] / grid.length)
                        .expand(grid.n, grid.n))
