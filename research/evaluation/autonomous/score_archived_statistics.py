"""Lightweight postprocessing of frozen fields; no solver, CNN or DNS imports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CASES = (110240101, 110240102, 110240104, 110240105)
SEEDS = (2026, 2028)
MODELS = ("NGM4", "DSF-Aug", "DSF-Sym", "DSF-Trend", "DSF-Hold")
PRIMARY = "n1536_half"
QUALITY = "REFERENCE_NOT_CONVERGED"
STEPS = np.arange(0, 801, 10)
TIMES = 1 + .005 * STEPS
EVAL = (40, 80)
OUT = Path(__file__).resolve().parents[1]


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def seed_key(value):
    return None if pd.isna(value) else int(value)


def model_key(value):
    return "NGM4" if value == "Clark4" else value


class Archives:
    def __init__(self, repo, paper):
        self.repo = repo
        self.paper = paper
        self.hashes = {}
        self.inventory = self.csv("data/reference/SOURCE_INVENTORY.csv")
        self.ic_receipt = json_read(self.verify(paper / "tables/ic105_receipt.json"))
        self.ic_hashes = {Path(p): h for p, h in self.ic_receipt["inputs_sha256"].items()}
        self.ic_accuracy = self.csv("tables/ic105_accuracy.csv")
        self.valid = repo / "research/JCP_DSF_TemporalSym_20261117/VALIDATION_20261118"
        self.op = self.valid / "LES/ic105/OPERATORS.npz"
        old = repo / "research/full_aug_parameter_pc_full_v1_20261116/results/run01/OPERATORS.npz"
        with np.load(self.verify(self.op, self.ic_hashes[self.op.relative_to(repo)])) as v, np.load(self.verify(old)) as p:
            for key in ("dealias", "inverse", "kx", "ky"):
                np.testing.assert_array_equal(v[key], p[key])
            self.mask = v["dealias"].copy()
            self.mask[0, 0] = False
            self.inverse = v["inverse"].copy()
        k = np.rint(np.fft.fftfreq(100, d=.01))
        self.shell = np.rint(np.hypot(k[:, None], k[None, :])).astype(int)
        self.exact_inverse = 1 / (k[:, None]**2 + k[None, :]**2)[self.mask]
        expected = (np.abs(k[:, None]) <= 33) & (np.abs(k[None, :]) <= 33) & (self.shell != 0)
        np.testing.assert_array_equal(self.mask, expected)
        assert self.shell[self.mask].max() == 47 and self.mask.sum() == 4488

    def verify(self, path, expected=None):
        path = path.resolve()
        key = str(path)
        if key not in self.hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 << 20), b""):
                    digest.update(block)
            self.hashes[key] = digest.hexdigest()
        if expected is not None:
            assert self.hashes[key] == expected, path
        return path

    def csv(self, relative):
        return pd.read_csv(self.verify(self.paper / relative))

    def reference(self, case):
        results = {}
        for row in self.inventory.query("case == @case and kind == 'DNS'").itertuples():
            if case == CASES[-1]:
                continue
            folder = self.repo / row.source
            done = json_read(self.verify(self.repo / row.receipt, row.receipt_sha256))
            if "outputs" in done:
                name = "reference_omega.npy"
                times = np.load(self.verify(folder / "time_s.npy", done["outputs"]["time_s.npy"]), mmap_mode="r")
                np.testing.assert_allclose(times[4*STEPS], TIMES, rtol=0, atol=1e-11)
                data = np.load(self.verify(folder / name, done["outputs"][name]), mmap_mode="r")[4*STEPS]
            else:
                data = self.reference_frames(folder, done, [f"frame_{8000+40*n:05d}.npz" for n in STEPS])
            results[row.reference] = data
        if case == CASES[-1]:
            folder = self.valid / f"DNS/{case}_n1536_h0.00012500"
            receipt = folder / "DONE.json"
            done = json_read(self.verify(receipt, self.ic_hashes[receipt.relative_to(self.repo)]))
            assert done["reference_quality"] == QUALITY and done["completed_t_end"] == 5
            results[PRIMARY] = self.reference_frames(folder, done, [f"frame_{n:04d}.npz" for n in STEPS])
        assert set(results) == ({"n1024_main", "n1536_main", PRIMARY} if case in CASES[:2] else {PRIMARY})
        return results

    def reference_frames(self, folder, done, names):
        states = []
        for time, name in zip(TIMES, names):
            with np.load(self.verify(folder / name, done["frames"][name]), allow_pickle=False) as z:
                assert abs(float(z["time_s"])-time) <= 1e-11, name
                states.append(z["reference_omega"])
        return np.stack(states)

    def les(self, case):
        results = {}
        if case == CASES[-1]:
            assert self.ic_receipt["scored"] == 9
            for row in self.ic_accuracy.itertuples():
                seed = seed_key(row.seed)
                folder = self.repo / row.archive
                if seed is None:
                    initial, key = folder / "INITIAL.npz", "omega"
                else:
                    startup = self.valid / "LES/ic105/startup" / f"{case}_{seed}"
                    receipt = json_read(self.verify(startup / "DONE.json"))
                    initial = self.verify(startup / "STARTUP.npz", receipt["outputs"]["STARTUP.npz"])
                    key = "omega0"
                status = folder / "STATUS.json"
                self.verify(status, self.ic_hashes[status.relative_to(self.repo)])
                results[row.model, seed] = self.step_fields(folder, initial, key, self.ic_hashes)
        else:
            for row in self.inventory.query("case == @case and kind == 'LES'").itertuples():
                model, seed = model_key(row.model), seed_key(row.seed)
                status = json_read(self.verify(self.repo / row.receipt, row.receipt_sha256))
                assert status["status"] == "VERIFIED" and status["last_accepted_step"] == 800
                source = self.repo / row.source
                if seed is None:
                    folder = source.parent
                    results[model, seed] = self.step_fields(folder, folder / "INITIAL.npz", "omega",
                                                           {source.relative_to(self.repo)/"0800.npz": row.source_sha256})
                else:
                    data = np.load(self.verify(source, row.source_sha256), mmap_mode="r", allow_pickle=False)
                    assert data.shape == (801, 100, 100)
                    results[model, seed] = data[STEPS]
        assert set(results) == {("NGM4", None), *((m, s) for m in MODELS[1:] for s in SEEDS)}
        return results

    def step_fields(self, folder, initial, key, expected):
        with np.load(self.verify(initial), allow_pickle=False) as z:
            field = [z[key]]
        for n in STEPS[1:]:
            step = folder / "steps" / f"{n:04d}.npz"
            with np.load(self.verify(step, expected.get(step.relative_to(self.repo))), allow_pickle=False) as z:
                assert json.loads(str(z["metadata"].item()))["accepted_step"] == n
                field.append(z["omega"])
        return np.stack(field)

    def field_statistics(self, fields):
        assert fields.shape == (81, 100, 100) and np.isfinite(fields).all()
        centered = fields - fields.mean(axis=(-2, -1), keepdims=True)
        sigma = np.sqrt(np.mean(centered**2, axis=(-2, -1)))
        assert (sigma > 0).all()
        hat = np.fft.fft2(fields, norm="forward")
        assert np.max(np.sum(np.abs(hat[:, ~self.mask])**2, axis=1)) < 1e-20
        power = .5 * np.abs(hat[:, self.mask])**2
        energy = power * self.inverse[self.mask]
        np.testing.assert_allclose(power.sum(axis=1), .5*sigma**2, rtol=2e-13, atol=1e-14)
        spectra = {i: tuple(np.bincount(self.shell[self.mask], weights=x[i], minlength=48)[1:]
                            for x in (energy, power)) for i in EVAL}
        for i, (e, z) in spectra.items():
            np.testing.assert_allclose([e.sum(), z.sum()], [energy[i].sum(), power[i].sum()],
                                       rtol=2e-13, atol=1e-14)
        return dict(fields=fields, centered=centered, sigma=sigma, K=energy.sum(axis=1),
                    K_exact_DNS=(power*self.exact_inverse).sum(axis=1),
                    Z=power.sum(axis=1), spectra=spectra)


def matching_archive(archives):
    rank = archives.csv("data/reference/REFERENCE_RANKING_ROBUSTNESS.csv")
    rank = {(r.case, model_key(r.model), seed_key(r.network_seed), r.reference): r.omega_MSE
            for r in rank.itertuples()}
    for row in archives.ic_accuracy.itertuples():
        assert row.experiment_status == "VERIFIED" and row.t_end == 5
        rank[row.case, row.model, seed_key(row.seed), PRIMARY] = row.full_omega_MSE
    pairs = pd.concat([archives.csv("tables/paired_les_accuracy.csv"), archives.ic_accuracy], ignore_index=True)
    pairs = {(r.case, model_key(r.model), None if r.model in ("NGM4", "Clark4") else seed_key(r.seed)): r
             for r in pairs.query("dt == .005").itertuples()}
    curves = pd.concat([archives.csv("data/inputs/les_state_metrics.csv"),
                        archives.csv("tables/ic105_state_metrics.csv")], ignore_index=True)
    curves = curves.query("dt == .005 and band == 'all' and step > 0")
    ref_curves = archives.csv("data/reference/FILTERED_VALUES.csv")
    return rank, pairs, curves, ref_curves


def rank_rows(summary):
    rows = []
    metrics = {"full_omega_MSE": True, "pdf_TV_t5": True, "pdf_center_TV_t5": True, "pdf_tail_TV_t5": True,
               "CC_t5": False, "CC_mean_positive_leads": False,
               "E_L1_all_t5": True, "E_L1_high_t5": True, "Z_L1_all_t5": True, "Z_L1_high_t5": True,
               "mean_abs_K_bias": True, "mean_abs_Z_bias": True}
    for (case, reference), group in summary.groupby(["case", "reference"]):
        for seed in SEEDS:
            paired = group[group.network_seed.eq(seed) | group.network_seed.isna()]
            assert len(paired) == 5
            for metric, ascending in metrics.items():
                ordered = paired.sort_values(metric, ascending=ascending)
                rows.append(dict(case=case, network_seed=seed, reference=reference, metric=metric,
                                 best=ordered.iloc[0].model, best_value=ordered.iloc[0][metric],
                                 ordering_best_to_worst=" > ".join(ordered.model)))
    return rows


def score(archives):
    official, pairs, curves, archive_ref = matching_archive(archives)
    summaries, densities, spectra, correlations, amplitude = [], [], [], [], []
    edges = np.linspace(-8., 8., 129)
    centers = .5*(edges[1:] + edges[:-1])
    center = np.abs(centers) < 2
    for case in CASES:
        ref = {k: archives.field_statistics(v) for k, v in archives.reference(case).items()}
        models = {k: archives.field_statistics(v) for k, v in archives.les(case).items()}
        for data in models.values():
            np.testing.assert_allclose(data["fields"][0], ref[PRIMARY]["fields"][0], atol=1e-12, rtol=0)
        for (model, seed), data in models.items():
            saved = curves[curves.case.eq(case) & curves.model.map(model_key).eq(model)]
            saved = saved[saved.seed.isna() | saved.seed.eq(0)] if seed is None else saved[saved.seed.eq(seed)]
            saved = saved.sort_values("step")
            np.testing.assert_array_equal(saved.step, STEPS[1:])
            for key in ("K", "Z"):
                np.testing.assert_allclose(data[key][1:], saved[key], rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(ref[PRIMARY][key][1:], saved[f"reference_{key}"],
                                           rtol=2e-10, atol=2e-12)
        for tag, r in ref.items():
            saved = archive_ref[archive_ref.case.eq(case) & archive_ref.reference.eq(tag)].sort_values("time")
            if not saved.empty:
                np.testing.assert_allclose(saved.time, TIMES, atol=1e-12, rtol=0)
                # This DNS-only report uses pure 1/k², while official LES scores
                # use the frozen inverse operator for both fields.
                np.testing.assert_allclose(r["K_exact_DNS"], saved.K, rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(r["Z"], saved.Z, rtol=2e-10, atol=2e-12)
            for (model, seed), data in {("FDNS", None): r, **models}.items():
                ident = dict(case=case, model=model, network_seed=seed, reference=tag, reference_quality=QUALITY)
                cc = np.mean(data["centered"]*r["centered"], axis=(-2,-1))/(data["sigma"]*r["sigma"])
                assert (abs(cc) <= 1+3e-15).all()
                cc = cc.clip(-1, 1)
                np.testing.assert_allclose(2*data["Z"] + 2*r["Z"] - 2*cc*data["sigma"]*r["sigma"],
                                           np.mean((data["fields"]-r["fields"])**2, axis=(-2,-1)),
                                           atol=3e-12, rtol=0)
                correlations.extend(dict(**ident, time=t, lead=t-1, CC=value) for t, value in zip(TIMES, cc))
                metrics = {}
                for i in EVAL:
                    t = int(TIMES[i])
                    xi = data["centered"][i]/data["sigma"][i]
                    xi_ref = r["centered"][i]/r["sigma"][i]
                    count, _ = np.histogram(xi, edges)
                    base_count, _ = np.histogram(xi_ref, edges)
                    assert count.sum() == base_count.sum() == 10000
                    density = count / (10000 * np.diff(edges))
                    tv = .5*abs(count-base_count)/10000
                    densities.extend(dict(**ident, time=t, bin_left=a, bin_right=b, count=int(c), density=y)
                                     for a,b,c,y in zip(edges[:-1], edges[1:],count,density))
                    metrics[f"pdf_TV_t{t}"] = tv.sum()
                    metrics[f"pdf_center_TV_t{t}"] = tv[center].sum()
                    metrics[f"pdf_tail_TV_t{t}"] = tv[~center].sum()
                    e,z = data["spectra"][i]
                    er,zr = r["spectra"][i]
                    spectra.extend(dict(**ident,time=t,shell=k,E=a,Z=b) for k,(a,b) in enumerate(zip(e,z),1))
                    amplitude.append(dict(**ident, time=t, omega_RMS=np.sqrt(np.mean(data["fields"][i]**2)),
                                          omega_sigma=data["sigma"][i], K=data["K"][i], Z=data["Z"][i],
                                          Z_signed_bias_pct=100*(data["Z"][i]/r["Z"][i]-1),
                                          Z_high_ratio=z[29:].sum()/zr[29:].sum()))
                    if t == 5:
                        for key, x, xref in (("E",e,er),("Z",z,zr)):
                            for band, sl in (("all",slice(None)),("high",slice(29,None))):
                                metrics[f"{key}_L1_{band}_t5"] = abs(x[sl]-xref[sl]).sum()/xref[sl].sum()
                if model == "FDNS":
                    continue
                mse = official[case,model,seed,tag]
                np.testing.assert_allclose(np.mean((data["fields"][1:]-r["fields"][1:])**2), mse,
                                           rtol=1e-11, atol=2e-12)
                bias = {key: np.mean(abs(data[key][1:]/r[key][1:]-1)) for key in ("K","Z")}
                if tag == PRIMARY:
                    expected = pairs[case,model,seed]
                    np.testing.assert_allclose([mse,bias["K"],bias["Z"]],
                                               [expected.full_omega_MSE,expected.mean_abs_K_bias,expected.mean_abs_Z_bias],
                                               rtol=1e-10, atol=2e-12)
                gains = {"improvement_vs_NGM4_pct": 100*(1-mse/official[case,"NGM4",None,tag])}
                for short, baseline in (("Aug","DSF-Aug"),("Sym","DSF-Sym")):
                    for s in ((seed,) if seed else SEEDS):
                        suffix = "" if seed else f"_seed_{s}"
                        gains[f"improvement_vs_{short}_pct{suffix}"] = 100*(1-mse/official[case,baseline,s,tag])
                summaries.append(dict(**ident, LES_dt=.005, MSE_samples=80, full_omega_MSE=mse,
                                      CC_start=1., CC_end=5., CC_samples=81, forecast_origins=1,
                                      CC_mean_positive_leads=cc[1:].mean(), CC_t5=cc[-1], **metrics,
                                      mean_abs_K_bias=bias["K"], mean_abs_Z_bias=bias["Z"], **gains))
    summary = pd.DataFrame(summaries)
    assert len(summary) == 72 and not summary.duplicated(["case","model","network_seed","reference"]).any()
    for name, data in (("METRIC_SUMMARY", summary), ("METRIC_RANKINGS", rank_rows(summary)),
                       ("vorticity_pdf_bins", densities), ("vorticity_cc", correlations),
                       ("spectra_E_Z", spectra), ("snapshot_amplitudes", amplitude)):
        pd.DataFrame(data).to_csv(OUT/"data"/f"{name}.csv", index=False, encoding="utf-8-sig")
    (OUT/"audit/STATISTICS_RECEIPT.json").write_text(json.dumps(dict(
        reference_quality=QUALITY, score_rows=len(summary), unique_LES_trajectories=36,
        pdf_bin_range=[-8,8], pdf_bin_width=.125, pdf_outside_total=0,
        short_forecast_origins=1, CC_times=TIMES.tolist(), snapshots=[3,5],
        max_shell=47, high_shells=[30,47], FFT_norm="forward", spectral_plot_floor=np.finfo(float).tiny,
        DNS=0, LES=0, training=0, CNN=0, sha256=archives.hashes,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    score(Archives(args.repo.resolve(), args.paper.resolve()))
