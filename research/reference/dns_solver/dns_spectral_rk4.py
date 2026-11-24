"""Same strict 2/3 Galerkin/RK4 algebra as audited Grid, in Fourier coordinates."""
import numpy as np
import torch
from rp_common import ROOT
from research.skew_full_validation_20261020.core import Grid
from research.skew_full_validation_20261020.reference import initial


class SpectralRK4:
    def __init__(self, n, length, nu, device='cpu'):
        self.grid = Grid(n, length, device, torch.float64)
        self.nu = nu
        g = self.grid
        self.multipliers = torch.stack((1j*g.ky*g.inv, -1j*g.kx*g.inv, 1j*g.kx, 1j*g.ky))
        self.max_speed = torch.zeros((), device=device, dtype=torch.float64)

    def initial(self, seed, cfg):
        return self.grid.fft(initial(self.grid, seed, cfg)) * self.grid.mask

    def rhs(self, h):
        g = self.grid
        h = h*g.mask
        u, v, gx, gy = g.ifft(self.multipliers*h)
        self.max_speed = torch.maximum(self.max_speed, (u.abs()+v.abs()).amax())
        out = (-g.fft(u*gx+v*gy)-self.nu*g.k2*h)*g.mask
        p = h.abs().square()
        return out, -self.nu*(g.k2*p).sum(), -self.nu*p.sum()

    def step(self, h, dt):
        a, za, ea = self.rhs(h)
        b, zb, eb = self.rhs(h+dt*a/2)
        c, zc, ec = self.rhs(h+dt*b/2)
        d, zd, ed = self.rhs(h+dt*c)
        return ((h+dt*(a+2*b+2*c+d)/6)*self.grid.mask,
                dt*(za+2*zb+2*zc+zd)/6, dt*(ea+2*eb+2*ec+ed)/6)


def labels(solver, low, h, delta, nu, legacy=True):
    """Independent flux-product and semi-discrete RHS definitions on one F."""
    high = solver.grid
    gain = torch.exp(-delta**2 * high.k2/24)

    def transfer_h(v):
        # Reuse the audited, forward-normalized Fourier transfer implementation.
        return high.transfer(high.ifft(v), low)

    def F(a):
        return low.project(transfer_h(high.fft(a)*gain))

    w = high.ifft(h)
    u, v, _, _ = high.ifft(solver.multipliers*h)
    wb, ub, vb = F(w), F(u), F(v)
    qx, qy = low.project(F(u*w)-ub*wb), low.project(F(v*w)-vb*wb)
    source_flux = -low.div(qx, qy)
    dot = low.project(transfer_h(solver.rhs(h)[0]*gain))
    resolved = low.rhs(wb, nu)
    source_gap = dot-resolved
    result = dict(reference_omega=wb, reference_dot=dot, reference_source=source_flux,
                  reference_source_rhs_gap=source_gap, reference_qx=qx, reference_qy=qy,
                  reference_residual=dot-resolved-source_flux)
    if legacy:
        from spnsgs.filters import filter_vorticity_snapshot
        raw = [high.transfer(x, low).cpu().numpy() for x in (u, v, w)]
        lab = filter_vorticity_snapshot(*raw, low.length/low.n, low.length/low.n, delta)
        old = low.project(torch.as_tensor(-lab['div_q_sgs'], dtype=low.dtype, device=low.device))
        result['legacy_reference_source'] = old
        result['legacy_reference_residual'] = dot-resolved-old
    return {k: x.detach().cpu().numpy() for k, x in result.items()}


def raw_diagnostics(solver, h):
    g = solver.grid
    p = h.abs().square()
    modes = g.modes.abs()
    tail = ((modes[:, None].maximum(modes[None, :])) >= .8*g.cut) & g.mask
    shell = torch.round(torch.sqrt(g.k2)*g.length/(2*np.pi)).long()
    spectrum = torch.bincount(shell.flatten(), weights=p.flatten(), minlength=int(shell.max())+1)
    return dict(E=float(.5*(p*g.inv).sum()), Z=float(.5*p.sum()),
                tail_fraction=float(p[tail].sum()/p.sum().clamp_min(1e-30))), spectrum.cpu().numpy()
