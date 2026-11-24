"""FFT conventions, exact bilinear-coefficient Galerkin source, file safety."""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scalar(x):
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


def rms(x):
    return x.square().mean().sqrt()


def relative(x, target):
    return scalar(rms(x - target) / rms(target).clamp_min(1e-20))


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def timed(fn, device, count=1):
    sync(device); start = time.perf_counter()
    result = None
    for _ in range(count):
        result = fn()
    sync(device)
    return result, (time.perf_counter() - start) / count


class Grid:
    """[y,x], physical FFT frequencies; strict 2/3 excludes alias boundary."""
    def __init__(self, n, length=2*math.pi, device='cpu', dtype=torch.float64):
        self.n, self.length, self.device, self.dtype = n, length, torch.device(device), dtype
        self.k = torch.fft.fftfreq(n, d=length/n, device=device, dtype=dtype)*2*math.pi
        self.ky, self.kx = torch.meshgrid(self.k, self.k, indexing='ij')
        self.k2 = self.kx.square()+self.ky.square()
        self.inv = torch.where(self.k2 > 0, 1/self.k2.clamp_min(1e-30), 0.)
        self.cut = (n-1)//3
        self.modes = torch.fft.fftfreq(n, d=1/n, device=device, dtype=dtype).round().long()
        self.mask = ((self.modes[:, None].abs() <= self.cut) &
                     (self.modes[None, :].abs() <= self.cut))
        self.mask[0, 0] = False

    def fft(self, w):
        return torch.fft.fft2(w, norm='forward')

    def ifft(self, h):
        return torch.fft.ifft2(h, norm='forward').real

    def project(self, w):
        return self.ifft(self.fft(w)*self.mask)

    def grad(self, w):
        h = self.fft(w)
        return self.ifft(1j*self.kx*h), self.ifft(1j*self.ky*h)

    def div(self, qx, qy, project=True):
        h = 1j*self.kx*self.fft(qx)+1j*self.ky*self.fft(qy)
        return self.ifft(h*self.mask if project else h)

    def lap(self, w):
        return self.ifft(-self.k2*self.fft(w))

    def state(self, w):
        h = self.fft(w)*self.mask
        ph = h*self.inv
        psi = self.ifft(ph)
        u, v = self.ifft(1j*self.ky*ph), self.ifft(-1j*self.kx*ph)
        ux, uy = self.grad(u); vx, vy = self.grad(v)
        gx, gy = self.ifft(1j*self.kx*h), self.ifft(1j*self.ky*h)
        return dict(omega=self.ifft(h), psi=psi, u=u, v=v, omega_x=gx, omega_y=gy,
                    laplacian_omega=self.ifft(-self.k2*h), u_x=ux, u_y=uy,
                    v_x=vx, v_y=vy,
                    strain=((ux-vy).square()+(uy+vx).square()+1e-12).sqrt())

    def rhs(self, w, nu, source=None):
        s = self.state(w)
        out = -self.project(s['u']*s['omega_x']+s['v']*s['omega_y'])+nu*s['laplacian_omega']
        if source is not None:
            out = out+source(s['omega'])
        return self.project(out)

    def step(self, w, dt, nu, source=None):
        k1 = self.rhs(w, nu, source)
        k2 = self.rhs(w+dt*k1/2, nu, source)
        k3 = self.rhs(w+dt*k2/2, nu, source)
        k4 = self.rhs(w+dt*k3, nu, source)
        return self.project(w+dt*(k1+2*k2+2*k3+k4)/6)

    def transfer(self, w, other):
        """No Nyquist retained: spectral crop/zero padding of resolved fields."""
        h = self.fft(w)
        out = torch.zeros(w.shape[:-2]+(other.n, other.n), device=w.device, dtype=h.dtype)
        limit = min((self.n-1)//2, (other.n-1)//2)
        ks = torch.arange(-limit, limit+1, device=w.device)
        out[..., ks[:, None] % other.n, ks[None, :] % other.n] = h[..., ks[:, None] % self.n, ks[None, :] % self.n]
        return other.ifft(out)

    def energy(self, w):
        s = self.state(w)
        return scalar(.5*(s['u'].square()+s['v'].square()).mean()), scalar(.5*w.square().mean())


def unpack(x):
    # Frobenius-isometric symmetric coordinates, plus skew scalar.
    return x[0], x[1]/math.sqrt(2), x[2], x[3]


def pack_matrix(x):
    a, b, d, _ = unpack(x)
    return torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)


def edge_bounds(x, length):
    h = length/x.shape[-1]
    dx = (x.roll(-1, -1)-x)/h
    dy = (x.roll(-1, -2)-x)/h
    return dict(M_x=torch.linalg.vector_norm(dx[:3], dim=0).amax(),
                M_y=torch.linalg.vector_norm(dy[:3], dim=0).amax(),
                phi_x=dx[3].abs().amax(), phi_y=dy[3].abs().amax())


def feasible_repair(x, budget, length, skew=True):
    """Global contraction, preserving continuous bilinear PSD and convex edge caps."""
    x = x.clone()
    center = torch.zeros_like(x); center[0] = center[2] = budget['kappa']/2
    eig = torch.linalg.eigvalsh(pack_matrix(x))
    k = budget['kappa']
    low = (eig.amin()-k/2).abs().clamp_min(1e-30)
    high = (eig.amax()-k/2).abs().clamp_min(1e-30)
    scale_M = torch.minimum(torch.ones_like(low), torch.minimum((k/2)/low, (k/2)/high))
    e = edge_bounds(x, length)
    for key in ['M_x', 'M_y']:
        scale_M = torch.minimum(scale_M, torch.as_tensor(budget['L_M']/math.sqrt(2), device=x.device, dtype=x.dtype)/e[key].clamp_min(1e-30))
    repaired_M = center[:3]+scale_M*(x[:3]-center[:3])
    sp = torch.minimum(torch.ones_like(scale_M), budget['Phi']/x[3].abs().amax().clamp_min(1e-30))
    for key in ['phi_x', 'phi_y']:
        sp = torch.minimum(sp, torch.as_tensor(budget['L_phi']/math.sqrt(2), device=x.device, dtype=x.dtype)/e[key].clamp_min(1e-30))
    repaired_phi = x[3]*sp if skew else x[3]*0
    return torch.cat([repaired_M,repaired_phi[None]],0)


class BilinearSource:
    """Exact Galerkin moments of continuous periodic bilinear M,phi.

    Hat-function Fourier coefficient: DFT(nodes)[k mod c] sinc(k/c)^2.
    Only |coefficient k| <= 2K enters P div(C grad w), K=LES cutoff.
    Padding length >= 4K+1 avoids all aliases into |k|<=K in products.
    Pointwise PSD pertains to the bilinear interpolant, not its Fourier truncation.
    """
    def __init__(self, grid, control):
        self.grid, self.c = grid, control
        self.pad = Grid(2*grid.n, grid.length, grid.device, grid.dtype)
        modes = self.pad.modes
        self.idx = modes % control
        weight = torch.sinc(modes.to(grid.dtype)/control).square()
        coeffmask = (modes[:, None].abs() <= 2*grid.cut) & (modes[None, :].abs() <= 2*grid.cut)
        self.weight = weight[:, None]*weight[None, :]*coeffmask

    def coefficients_fourier(self, x):
        h = torch.fft.fft2(x, norm='forward')
        return h[..., self.idx[:, None], self.idx[None, :]]*self.weight

    def __call__(self, w, x):
        wp = self.grid.transfer(self.grid.project(w), self.pad)
        gx, gy = self.pad.grad(wp)
        a, b, d, phi = unpack(self.pad.ifft(self.coefficients_fourier(x)))
        fx, fy = a*gx+(b-phi)*gy, (b+phi)*gx+d*gy
        S = self.pad.div(fx, fy, project=False)
        return self.grid.project(self.pad.transfer(S, self.grid))

    def collocated_coefficients(self, x, n):
        # Exact bilinear interpolation at arbitrary uniform sample positions.
        pos = torch.arange(n, device=x.device, dtype=x.dtype)*self.c/n
        ix = pos.floor().long() % self.c; t = pos-pos.floor()
        ix1 = (ix+1) % self.c
        row = x[..., ix]*(1-t)+x[..., ix1]*t
        return row[..., ix, :]*(1-t)[:, None]+row[..., ix1, :]*t[:, None]


def coefficient_metrics(x, length, budget):
    e = edge_bounds(x, length)
    eig = torch.linalg.eigvalsh(pack_matrix(x))
    gx = (x.roll(-1, -1)-x)/(length/x.shape[-1])
    gy = (x.roll(-1, -2)-x)/(length/x.shape[-1])
    k = budget['kappa']
    violation = max(0., -scalar(eig.min()), scalar(eig.max())-k,
                    scalar(x[3].abs().max())-budget['Phi'],
                    scalar(e['M_x'])-budget['L_M']/math.sqrt(2),
                    scalar(e['M_y'])-budget['L_M']/math.sqrt(2),
                    scalar(e['phi_x'])-budget['L_phi']/math.sqrt(2),
                    scalar(e['phi_y'])-budget['L_phi']/math.sqrt(2))
    return dict(eigen_min=scalar(eig.min()), eigen_max=scalar(eig.max()),
                eigen_quantiles=torch.quantile(eig.reshape(-1), torch.tensor([0., .25, .5, .75, 1.], device=x.device, dtype=x.dtype)).cpu().tolist(),
                M_F_RMS=scalar(x[:3].square().sum(0).mean().sqrt()),
                M_F_max=scalar(x[:3].square().sum(0).max().sqrt()),
                phi_RMS=scalar(rms(x[3])), phi_max=scalar(x[3].abs().max()),
                M_gradient_continuous_bound=math.hypot(scalar(e['M_x']),scalar(e['M_y'])),
                phi_gradient_continuous_bound=math.hypot(scalar(e['phi_x']),scalar(e['phi_y'])),
                additional_velocity_bound=math.hypot(scalar(e['phi_x']),scalar(e['phi_y'])),
                edge_violation=violation,
                combined_cost=scalar(x.square().sum(0).mean()+gx.square().sum(0).mean()+gy.square().sum(0).mean()),
                continuous_scope='convex bilinear PSD; piecewise W1inf a.e.; per-axis cap L/sqrt(2); not smooth C1')


def source_metrics(grid, w, pred, target, full_norm=None):
    err = relative(pred, target)
    a, b = pred-pred.mean(), target-target.mean()
    corr = scalar((a*b).mean()/(rms(a)*rms(b)).clamp_min(1e-30))
    ph, th = grid.fft(pred), grid.fft(target)
    radius = torch.sqrt(grid.k2)
    bands = {}
    for name, lo, hi in [('low',0.,7.),('middle',7.,14.),('high',14.,1000.)]:
        mask = (radius>=lo)&(radius<hi)&grid.mask
        den = th[mask].abs().square().sum()
        bands[name] = scalar((ph[mask]-th[mask]).abs().square().sum().sqrt()/den.sqrt().clamp_min(1e-20))
    ref_budget = scalar((w*target).mean()); budget = scalar((w*pred).mean())
    full = float(full_norm) if full_norm is not None else scalar(rms(target))
    proj2 = scalar(target.square().mean())
    tail2 = max(0.,full*full-proj2)
    return dict(source_relative_L2=err,source_correlation=corr,band_errors=bands,
                reference_enstrophy_budget=ref_budget,predicted_enstrophy_budget=budget,
                budget_absolute_error=abs(budget-ref_budget),
                raw_source_RMS=full,projection_tail_RMS=math.sqrt(tail2),
                projection_tail_fraction=math.sqrt(tail2)/max(full,1e-20),
                relative_L2_full=math.sqrt(scalar((pred-target).square().mean())+tail2)/max(full,1e-20))
