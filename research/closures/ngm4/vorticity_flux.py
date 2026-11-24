"""Gaussian gradient expansion in the frozen solver's collocation convention.

Ghat=exp(-delta^2*k^2/24). For filtered factors, the exact product kernel
is exp(-delta^2*p.q/12)-1. Its first two terms give C2 and C4 below.
For P33 inputs on N100, quadratic products are alias-free AFTER P33;
unprojected flux samples are not claimed to be exact Galerkin products.
"""
import torch


def flux2(state, delta):
    # Keep the original expression and operation ordering exactly.
    c = delta**2 / 12.
    gx, gy = state['omega_x'], state['omega_y']
    return c*(state['u_x']*gx+state['u_y']*gy), c*(state['v_x']*gx+state['v_y']*gy)


def fourth_increment(solver, state, delta):
    d = solver.derivative
    # Hessians from the same projected first derivatives used by original C2.
    wx = d(state['omega_x'], 'x')
    wy = d(state['omega_y'], 'y')
    wxy = d(state['omega_x'], 'y')
    factor = delta**4 / 288.
    result = []
    for a in ('u', 'v'):
        result.append(factor*(d(state[a+'_x'], 'x')*wx
                              + 2*d(state[a+'_x'], 'y')*wxy
                              + d(state[a+'_y'], 'y')*wy))
    return tuple(result)


def flux4(solver, state, delta):
    a, b = flux2(state, delta), fourth_increment(solver, state, delta)
    return a[0]+b[0], a[1]+b[1]


def source(solver, flux):
    return -solver.project(solver.derivative(flux[0], 'x')+solver.derivative(flux[1], 'y'))


def multiply(solver, field, response):
    h = (torch.fft.fft2(field)*response*solver.dealias).clone()
    h[..., 0, 0] = 0
    return torch.fft.ifft2(h).real
