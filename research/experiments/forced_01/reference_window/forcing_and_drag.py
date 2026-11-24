"""Forced adapters. The extant Galerkin/RK4 and closure engines do the stepping."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENGINE = ROOT / "A0B1 full/jcp_dissipation_20261024/decisive/reference_platform_v2_20261025"
PRODUCTION = ROOT / "research/full_aug_parameter_pc_full_v1_20261116"
sys.path[:0] = [str(ROOT), str(ENGINE), str(PRODUCTION)]

from rp_spectral import Grid, SpectralRK4, labels
from research.full_response_20261020.operators import kolmogorov_vorticity_forcing
import pc_common as common
import pc_engine as closure


def forcing(grid, cfg, delta=0.0):
    """Vorticity forcing; historical operator's amplitude is for vorticity."""
    n = cfg["forcing_mode"]
    k = 2 * math.pi * n / grid.length
    gain = math.exp(-delta**2 * k**2 / 24)
    return kolmogorov_vorticity_forcing(grid, -cfg["F0"] * k * gain, n)


def rms_reynolds(kinetic_energy, cfg):
    """Use U_rms = sqrt(2K) and report both declared length scales."""
    speed = math.sqrt(2 * kinetic_energy)
    forcing_length = cfg["length"] / (2 * math.pi * cfg["forcing_mode"])
    return {
        "U_rms": speed,
        "Re_rms_unit_length": speed * cfg["reynolds"]["length_scale"] / cfg["nu"],
        "Re_rms_forcing_length": speed * forcing_length / cfg["nu"],
    }


class ForcedDNS(SpectralRK4):
    def __init__(self, n, cfg, device):
        super().__init__(n, cfg["length"], cfg["nu"], device)
        self.cfg = cfg
        self.alpha = cfg["alpha"]
        g = self.grid
        self.f_hat = g.fft(forcing(g, cfg)) * g.mask
        self.quadrature = None
        self.rhs_calls = 0
        self.tail_mask = (g.modes.abs()[:, None].maximum(g.modes.abs()[None, :])
                          >= .8 * g.cut) & g.mask
        self.shell = g.k2.sqrt().round().long()
        self.shell_count = int(self.shell[g.mask].max()) + 1

    def rhs(self, h):
        self.rhs_calls += 1
        rhs, visc_z, visc_k = super().rhs(h)
        g = self.grid
        h = h * g.mask
        power = h.abs().square()
        work = (h.conj() * self.f_hat).real
        terms = torch.stack((torch.stack(((work*g.inv).sum(), work.sum())),
                             torch.stack((visc_k, visc_z)),
                             -self.alpha * torch.stack(((power*g.inv).sum(), power.sum()))))
        if self.quadrature is not None:
            self.quadrature.append(terms)
        k, z = terms.sum(0)
        return rhs + self.f_hat - self.alpha * h, z, k

    def step_with_budget(self, h, dt):
        self.quadrature = []
        try:
            state, _, _ = super().step(h, dt)
            a, b, c, d = self.quadrature
            return state, dt * (a + 2*b + 2*c + d) / 6
        finally:
            self.quadrature = None

    def diagnostics(self, h, dt):
        g = self.grid
        p = (h.abs().square()*g.mask)
        diss = self.nu * g.k2 * p
        k, z = float(.5*(p*g.inv).sum()), float(.5*p.sum())
        phi = lambda a: float(a[self.tail_mask].sum() / a.sum().clamp_min(1e-30))
        cfl = dt * (float(self.max_speed)/(g.length/g.n) + self.nu*2*g.cut**2 + self.alpha)
        spec = torch.stack([
            torch.bincount(self.shell[g.mask], weights=a[g.mask], minlength=self.shell_count)
            for a in (.5*p*g.inv, .5*p, diss)])
        off_axis = float(p[:, 1:].sum() / p.sum().clamp_min(1e-30))
        return dict(K=k, Z=z, CFL=cfl, tail_Z=phi(p), tail_dissipation=phi(diss),
                    off_axis_Z=off_axis, **rms_reynolds(k,self.cfg)), closure.cpu(spec)


def embed(solver, modes, h_small):
    """Embed the same frozen low-mode coefficients on every DNS grid."""
    g = solver.grid
    ids = torch.as_tensor(modes % g.n, device=g.device)
    h = torch.zeros((g.n, g.n), device=g.device, dtype=torch.complex128)
    h[ids[:, None], ids[None, :]] = torch.as_tensor(h_small, device=g.device)
    return h


def fdns_labels(solver, low, h, cfg, bar_f):
    result = labels(solver, low, h, cfg["les"]["delta"], cfg["nu"], legacy=False)
    resolved_external = bar_f - cfg["alpha"] * result["reference_omega"]
    result["reference_source_rhs_gap"] -= resolved_external
    result["reference_residual"] -= resolved_external
    result["reference_forcing"] = bar_f
    result["reference_drag"] = -cfg["alpha"] * result["reference_omega"]
    a, b = result["reference_source"], result["reference_source_rhs_gap"]
    floor = cfg["gates"]["normalization_floor"]
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    tolerance = cfg["gates"]["max_source_identity_relative"]
    roundoff = 64*np.finfo(a.dtype).eps*rms(result["reference_dot"])
    # Tie the absolute fallback near zero to floating-point cancellation in the RHS.
    scale = max(rms(a), rms(b), floor, roundoff/tolerance)
    discrepancy = rms(a-b) / scale
    mean = abs(float(a.mean())) / scale
    if (not all(np.isfinite(x).all() for x in result.values())
            or max(discrepancy, mean) > tolerance):
        raise ValueError(f"Forced FDNS Source identity failed: {discrepancy=}, {mean=}")
    return result, discrepancy, mean


def configure_les(s, cfg, grid):
    """Attach resolved external RHS after constructing the frozen unforced model."""
    if s.nx != grid.n or s.ny != grid.n or s.config.domain.lx != grid.length:
        raise ValueError("Production grid/domain mismatch")
    mask = s.dealias.clone()
    mask[0, 0] = False
    if not torch.equal(mask, grid.mask):
        raise ValueError("Production projection mismatch")
    p = s.config.physics
    p.viscosity_kind, p.viscosity_value = "kinematic", cfg["nu"]
    p.forcing_enabled = True
    p.momentum_source_kind, p.momentum_source_amplitude = "acceleration", cfg["F0"]
    p.forcing_wave_number_rad_m = 2 * math.pi * cfg["forcing_mode"]/grid.length
    p.forcing_filter_gain = math.exp(-cfg["les"]["delta"]**2*p.forcing_wave_number_rad_m**2/24)
    bar_f = s.project(s.vorticity_forcing(0, torch.zeros_like(grid.k2)))
    torch.testing.assert_close(bar_f, forcing(grid, cfg, cfg["les"]["delta"]), atol=2e-14, rtol=2e-14)
    # The native RHS and the Engine reconstruction call the same external-term
    # hook. Output diagnostics retain the two individual terms separately.
    s.vorticity_forcing = lambda _t, w: bar_f - cfg["alpha"] * w
    return bar_f


def checkpoint_rows(cfg):
    path = ROOT / "research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json"
    manifest = common.read(path)
    if manifest["status"] != "FROZEN":
        raise ValueError("Checkpoint manifest is not frozen")
    for section in ("baseline_code_sha256", "training_code_sha256"):
        for name, digest in manifest[section].items():
            common.require(common.sha(ROOT/name) == digest, f"Frozen dependency changed: {name}")
    common.require(common.sha(common.q.study.common.CONFIG) == manifest["baseline_config_sha256"],
                   "Frozen baseline configuration changed")
    helpers = ROOT / "research/full_aug_orbit_rk4_one_step_v1_20261114/PHASE_ONE_MANIFEST.json"
    expected = common.read(helpers)["current_code_sha256"]["common_step.py"]
    common.require(common.sha(Path(common.decoder.__file__)) == expected, "Frozen decoder changed")
    rows = {}
    for row in manifest["checkpoints"]:
        seed = row["seed"]
        if row["mode"] != common.q.study.AUGMENTED or seed not in cfg["les"]["seeds"]:
            continue
        for name, key in (("file", "sha256"), ("run", "run_sha256"), ("done", "done_sha256")):
            common.require(common.sha(ROOT/row[name]) == row[key], f"Frozen {name} failed SHA check")
        training = common.read(ROOT/row["run"])
        done = common.read(ROOT/row["done"])
        common.require(seed not in rows and row["sha256"] == cfg["checkpoints"][str(seed)]
                       and done["best_sha256"] == row["sha256"] and done["status"] == "TRAINING_COMPLETE"
                       and training["cache_sha256"] == manifest["cache_sha256"], "Wrong augmented checkpoint")
        rows[seed] = row
    common.require(set(rows) == set(cfg["les"]["seeds"]), "Missing frozen seed")
    return rows
