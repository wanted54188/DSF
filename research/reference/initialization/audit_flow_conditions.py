"""Reconstruct the prescribed initial modes and compare the archived DNS/FDNS.

Run with the production repository root as the positional argument. No evolution
operator or model is imported. The script writes only to this paper's audit folder.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from plot_style import plt, panel


PAPER = Path(__file__).resolve().parents[1]
LONG = Path("research/ssc_long_horizon_v1_20261030/results/run01/reference")
MULTI = Path("research/full_c4_multi_ic_v1_20261110/results/paired01/reference")
VALID = Path("research/JCP_DSF_TemporalSym_20261117/VALIDATION_20261118")
FROZEN = Path("research/full_aug_hold_audit_v1_20261117/independent_ic105")
FLOWS = {110240101: ("101", "training"), 110240102: ("102", "development"),
         110240104: ("104", "development"), 110240105: ("IC105", "held-out")}
TIMES = 1 + np.arange(81) / 20
DT = 0.000125
FILTERED = PAPER / "data/reference/FILTERED_VALUES.csv"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(path, digest):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError(f"Archive fingerprint mismatch: {path}")


def initial_modes(seed, cfg):
    """Low-mode form of skew_full_validation_20261020/reference.py::initial."""
    cut = cfg["initial_max_mode"]
    rng = np.random.default_rng(seed)
    modes = np.arange(-cut, cut + 1)
    omega = np.zeros((len(modes), len(modes)), dtype=complex)
    variance = 0.0
    for ky in modes:
        for kx in range(cut + 1):
            r = math.hypot(kx, ky)
            if (kx == 0 and ky <= 0) or not 2 <= r <= cut:
                continue
            amp = float(rng.uniform(.5, 1.5) * r * np.exp(-(r / cfg["initial_peak_mode"])**4))
            z = amp * np.exp(1j * rng.uniform(0, 2 * math.pi)) / 2
            omega[cut + ky, cut + kx] = z
            omega[cut - ky, cut - kx] = z.conjugate()
            variance += amp * amp / 2
    return modes, omega * cfg["initial_RMS"] / math.sqrt(variance)


def native_series(rows):
    return {round(float(row["time_s"]) if row.get("time_s") else int(row["step"]) * DT, 8):
            (float(row["E"]), float(row["Z"])) for row in rows}


def archived_case(root, seed):
    if seed in list(FLOWS)[:2]:
        folder = root / LONG / str(seed) / "n1536_half"
        receipt = folder / "DONE.json"
        done = read_json(receipt)
        assert (done["seed"], done["n"], done["dt"]) == (seed, 1536, DT)
        original = Path(done["continuation_from"]).parent / "FREEZE.json"
        frozen = read_json(original)
        assert (frozen["seed"], frozen["n"], frozen["dt"]) == (seed, 1536, DT)
        config = frozen["config"]
        for name in ("time_s.npy", "reference_omega.npy", "RAW_DIAGNOSTICS.csv"):
            verify(folder / name, done["outputs"][name])
        index = np.arange(81) * 40
        np.testing.assert_allclose(np.load(folder / "time_s.npy")[index], TIMES, atol=1e-11, rtol=0)
        omega = np.load(folder / "reference_omega.npy", mmap_mode="r")[index]
        native = folder / "RAW_DIAGNOSTICS.csv"
        return config, omega, native_series(read_csv(native)), [receipt, original, native,
                                                               folder / "reference_omega.npy"]

    held = seed == 110240105
    folder = (root / VALID / "DNS" / "110240105_n1536_h0.00012500" if held
              else root / MULTI / str(seed))
    receipt = folder / "DONE.json"
    done = read_json(receipt)
    identity = folder / "RUN.json" if held else receipt
    run = read_json(identity)
    assert (run["case"], run.get("n", run.get("grid")), run["dt"]) == (seed, 1536, DT)
    if held:
        verify(identity, done["run_sha256"])
    states = []
    for i, t in enumerate(TIMES):
        path = folder / (f"frame_{10*i:04d}.npz" if held else f"frame_{8000+400*i:05d}.npz")
        verify(path, done["frames"][path.name])
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_allclose(float(saved["time_s"]), t, atol=1e-11, rtol=0)
            states.append(saved["reference_omega"])
    return run["config"], np.stack(states), native_series(done["observations"]), [receipt, identity]


def shell_spectra(modes, omega):
    k2 = modes[:, None]**2 + modes[None, :]**2
    shells = np.rint(np.sqrt(k2)).astype(int)
    z = .5 * np.abs(omega)**2
    e = np.divide(z, k2, out=np.zeros_like(z), where=k2 != 0)
    return tuple(np.bincount(shells.ravel(), weights=q.ravel()) for q in (e, z))


def validate_filtered_spectra(omega, modes, e, z, previous, seed):
    retained = (np.abs(modes[:, None]) <= 33) & (np.abs(modes[None, :]) <= 33)
    retained[0, 0] = False
    assert np.linalg.norm(omega[:, ~retained]) < 1e-12 * np.linalg.norm(omega)
    np.testing.assert_allclose(z.sum(axis=1), .5 * np.mean(np.fft.ifft2(omega, norm="forward").real**2,
                                                         axis=(-2, -1)), rtol=1e-13)
    scores = [row for row in previous if int(row["case"]) == seed and row["reference"] == "n1536_half"]
    np.testing.assert_allclose([float(row["time"]) for row in scores], TIMES, atol=1e-11, rtol=0)
    np.testing.assert_allclose([e.sum(axis=1), z.sum(axis=1)],
                               [[float(row[q]) for row in scores] for q in ("K", "Z")], rtol=1e-12)


def plot(out, spectra, series):
    colors = ["#00A6E6", "#FF851B", "#8A2BE2", "#00A36B"]
    styles = ["-", "--", "-.", ":"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.1), layout="constrained")
    axes = axes.ravel()
    for (flow, _), color, style in zip(FLOWS.values(), colors, styles):
        rows = [row for row in spectra if row["flow"] == flow]
        for ax, stage, tint in zip(axes[:2], ("DNS", "FDNS"), (0, 1)):
            subset = [row for row in rows if row["field"] == stage and row["time"] == tint and row["Z"] > 0]
            ax.plot([row["k"] for row in subset], [row["Z"] for row in subset],
                    color=color, linestyle=style, label=flow, linewidth=1.8)
        rows = [row for row in series if row["flow"] == flow]
        for ax, stage in zip(axes[2:], ("FDNS", "DNS")):
            z = np.array([row[f"Z_{stage}"] for row in rows])
            ax.plot(TIMES, 100*(1-z/z[0]), color=color, linestyle=style, linewidth=1.8)
    for ax, letter, title in zip(axes, "abcd", ("DNS initial spectrum", "FDNS at LES startup",
                                                "FDNS enstrophy loss", "DNS enstrophy loss")):
        panel(ax, letter, title)
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set(xlabel=r"Radial shell $k$", ylabel=r"$Z(k)$")
    axes[0].set_xlim(1, 12.6)
    axes[1].set_xlim(1, 47.8)
    for ax in axes[2:]:
        ax.set(xlabel=r"$t$", ylabel=r"$100[1-Z(t)/Z(1)]\;(\%)$", xlim=(1, 5))
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower left", ncol=2, fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"IC_PHYSICS.{ext}", dpi=400)
    plt.close(fig)


def main(root):
    previous = read_csv(FILTERED)
    summary, spectra, series, sources = [], [], [], {}
    for seed, (flow, role) in FLOWS.items():
        cfg, field, native, paths = archived_case(root, seed)
        settings = cfg["initial"]
        assert (settings["initial_max_mode"], settings["initial_peak_mode"],
                settings["initial_RMS"]) == (12, 8, 4)
        np.testing.assert_allclose([cfg["length"], cfg["nu"], cfg["delta"]],
                                   [2*np.pi, 1e-6, 4*(2*np.pi/100)], rtol=1e-14)
        modes, h0 = initial_modes(seed, settings)
        if flow == "IC105":
            freeze = root / FROZEN / "FROZEN.json"
            archive = root / FROZEN / "INITIAL_FOURIER_COEFFICIENTS.npz"
            attestation = root / FROZEN / "PROVENANCE_ATTESTATION.json"
            frozen = read_json(freeze)
            assert frozen["seed"] == seed
            assert read_json(attestation)["status"] == "USER_ATTESTED_UNUSED_BEFORE_FREEZE"
            verify(archive, frozen["coefficients_sha256"])
            with np.load(archive, allow_pickle=False) as saved:
                np.testing.assert_array_equal(modes, saved["modes"])
                np.testing.assert_allclose(h0, saved["omega_hat"], atol=1e-14, rtol=1e-13)
            paths += [freeze, archive, attestation]

        e0, z0 = shell_spectra(modes, h0)
        if 0 in native:
            np.testing.assert_allclose([e0.sum(), z0.sum()], native[0], atol=1e-13, rtol=1e-13)
        # Energy loss during the unforced 0–1 interval is 2 nu times integrated enstrophy.
        # Check the two archived prefixes with no t=0 scalar against the endpoint bounds.
        else:
            assert 2e-6*native[1][1] <= e0.sum() - native[1][0] <= 2e-6*z0.sum()
        modes = np.fft.fftfreq(100, d=.01).round().astype(int)
        omega = np.fft.fft2(field, norm="forward")
        e, z = (np.stack(values) for values in zip(*(shell_spectra(modes, w) for w in omega)))
        validate_filtered_spectra(omega, modes, e, z, previous, seed)
        k, zz = e.sum(axis=1), z.sum(axis=1)
        for time, kk, zed in zip(TIMES, k, zz):
            t = round(float(time), 8)
            energy, enstrophy = native[t]
            series.append(dict(flow=flow, flow_seed=seed, time=t, K_FDNS=kk, Z_FDNS=zed,
                               rms_FDNS=np.sqrt(2*zed), K_DNS=energy, Z_DNS=enstrophy))
        for name, time, es, zs, limit in (("DNS", 0, e0, z0, 12), ("FDNS", 1, e[0], z[0], 47),
                                          ("FDNS", 5, e[-1], z[-1], 47)):
            for j in range(1, limit+1):
                spectra.append(dict(flow=flow, flow_seed=seed, field=name, time=time,
                                    k=j, E=es[j], Z=zs[j]))
        summary.append(dict(
            flow=flow, role=role, flow_seed=seed, K_DNS_t0=e0.sum(), Z_DNS_t0=z0.sum(),
            rms_DNS_t0=np.sqrt(2*z0.sum()), E_peak_DNS_t0=int(e0.argmax()),
            Z_peak_DNS_t0=int(z0.argmax()), Z_peak_FDNS_t1=int(z[0].argmax()),
            rms_FDNS_t1=np.sqrt(2*zz[0]), K_FDNS_t1=k[0], K_FDNS_t5=k[-1],
            K_FDNS_change_pct=100*(k[-1]/k[0]-1), Z_FDNS_t1=zz[0], Z_FDNS_t5=zz[-1],
            Z_FDNS_loss_pct=100*(1-zz[-1]/zz[0]), K_DNS_t1=native[1][0], K_DNS_t5=native[5][0],
            K_DNS_loss_pct=100*(1-native[5][0]/native[1][0]), Z_DNS_t1=native[1][1],
            Z_DNS_t5=native[5][1], Z_DNS_loss_pct=100*(1-native[5][1]/native[1][1])))
        sources[flow] = [path.relative_to(root).as_posix() for path in dict.fromkeys(paths)]

    out = PAPER / "audit/initial_conditions"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("IC_PHYSICS_SUMMARY", summary), ("IC_PHYSICS_TIME_SERIES", series),
                       ("IC_PHYSICS_SPECTRA", spectra)):
        write_csv(out / f"{name}.csv", rows)
    (out / "IC_PHYSICS_SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    plot(out, spectra, series)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production_root", type=Path)
    main(parser.parse_args().production_root.resolve())
