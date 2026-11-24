"""Score archived runs. Optional common-state inference is explicitly requested."""
import argparse
import re
from pathlib import Path

import numpy as np
import torch

from forced_run import (HERE, DEFAULT_RUN, MODELS, PROTOCOL, PILOT_SOURCE, PILOT_QUALITY, Counter, c, checkpoint_rows, clark_solver,
                        closure, configure_les, field_spectra, frozen, index, reference_archive, reference_receipt,
                        sample_path, verify_pilot_input)
from forced_physics import Grid, rms_reynolds


def transfer(w, source, qx, qy):
    n = w.shape[0]
    k = np.fft.fftfreq(n,d=1/n).round()
    k2 = k[:,None]**2+k[None,:]**2
    h = np.fft.fft2(w,norm="forward")
    psi = np.fft.ifft2(np.divide(h,k2,out=np.zeros_like(h),where=k2>0),norm="forward").real
    gx = np.fft.ifft2(1j*k[None,:]*h,norm="forward").real
    gy = np.fft.ifft2(1j*k[:,None]*h,norm="forward").real
    pi = -(qx*gx+qy*gy)
    return dict(T_K=float(np.mean(psi*source)), T_Z=float(np.mean(w*source)),
                forward_Z=float(np.maximum(pi,0).mean()), backscatter_Z=float(np.maximum(-pi,0).mean()),
                backscatter_fraction=float((pi<0).mean()))


def scores(w, ref):
    wc, rc = w-w.mean(), ref-ref.mean()
    denom = np.linalg.norm(wc)*np.linalg.norm(rc)
    mse = float(np.mean((w-ref)**2))
    return dict(omega_MSE=mse, relative_L2=np.sqrt(mse)/max(float(np.sqrt(np.mean(ref**2))),1e-12),
                CC=float(np.vdot(wc,rc)/denom) if denom else None)


def long_statistics(archive, start, bins, ref_scale):
    fields, spec, terms = [], [], []
    for f in archive:
        if float(f["time_s"]) < start-1e-10:
            continue
        if "reference_omega" in f:
            w,src,x,y = (f["reference_"+k] for k in ("omega","source","qx","qy"))
        else:
            w,src,x,y = (f[k] for k in ("omega","Source","qx","qy"))
        fields.append(w)
        spec.append(field_spectra(w))
        terms.append(transfer(w,src,x,y))
    spectra = np.stack(spec)
    spec = spectra.mean(axis=0)
    series = spectra.sum(axis=2)
    halves = np.stack([b.mean(axis=0) for b in np.array_split(series,2)])
    w = np.stack(fields)
    pdfs = {}
    for key,data in (("raw",w),("reference_scaled",(w-w.mean(axis=(1,2),keepdims=True))/ref_scale)):
        edges = bins[key]
        counts = np.histogram(data,bins=edges)[0]
        pdfs[key] = (counts/(data.size*np.diff(edges)), float((data<edges[0]).mean()),
                     float((data>edges[-1]).mean()))
    stats = {k:float(np.mean([v[k] for v in terms])) for k in terms[0]}
    drift = abs(halves[1]-halves[0])/np.maximum(halves.mean(axis=0),1e-12)
    return dict(K=float(spec[0].sum()), Z=float(spec[1].sum()), samples=len(fields),
                K_half_mean_drift=float(drift[0]), Z_half_mean_drift=float(drift[1]), **stats), spec, pdfs


def read_trajectories(out, cfg, receipt, attempts):
    les = cfg["les"]
    states, status = {}, []
    accepted = {}
    for name in ("REFERENCE_PRIMARY_ONLY.json",):
        path = PILOT_SOURCE/name
        if path.exists():
            ref = c.read(path)
            if ref["window"] == receipt["window"] and ref["primary_frames"] == receipt["primary_frames"]:
                accepted[name] = c.sha(path)
    for method,name in MODELS.items():
        for seed in ([None] if method=="Clark4" else les["seeds"]):
            slug = f"{name}_{seed or 'analytic'}"
            folder = out/"LES"/slug/attempts.get(slug,"primary")
            row = dict(model=name, seed=seed, attempt=folder.name)
            done = folder/"DONE.json"
            if not done.exists():
                failed = folder/"FAILED.json"
                loss = c.read(failed) if failed.exists() else dict(status=(
                    "INCOMPLETE" if (folder/"RUN.json").exists() else "NOT_RUN"))
                status.append(dict(row,status=loss["status"],**loss.get("costs",{}),
                                   last_attempted_step=loss.get("last_attempted_step"),
                                   solver_seconds=loss.get("solver_seconds")))
                continue
            d, r = c.read(done), c.read(folder/"RUN.json")
            count = index(les["duration"],les["sample_dt"])+1
            i0 = index(receipt["window"]["start"],les["sample_dt"])
            if (r["model"] != name or r["network_seed"] != seed or r["h"] != les["dt"]
                    or d["run_sha256"] != c.sha(folder/"RUN.json") or len(d["frames"]) != count
                    or r["checkpoint_sha256"] != (cfg["checkpoints"][str(seed)] if seed else None)
                    or r["reference_receipt_sha256"] != accepted.get(r["reference_receipt"])
                    or r["reference_quality"] != PILOT_QUALITY
                    or r["reference"]["initial"] != receipt["primary_frames"][str(i0)]
                    or d["completed_t_end"] != receipt["window"]["end"] or d["status"] != "COMPLETE"):
                raise ValueError(f"Invalid LES completion identity: {folder}")
            traj = []
            for i in range(count):
                path = sample_path(folder,i)
                if c.sha(path) != d["frames"][str(i)]:
                    raise ValueError(f"LES field hash mismatch: {path}")
                z = c.load(path)
                if abs(float(z["time_s"])-(receipt["window"]["start"]+i*les["sample_dt"])) >= 1e-10:
                    raise ValueError(f"LES timestamp mismatch: {path}")
                traj.append(z)
            states[name,seed] = traj
            startup = d["startup"]
            status.append(dict(row,status="COMPLETE",exploratory=r["exploratory"],reference_quality=PILOT_QUALITY,**d["costs"],
                               shared_startup_CNN=startup["costs"]["CNN"] if startup else 0,
                               shared_startup_RHS=startup["costs"]["RHS"] if startup else 0,
                               shared_startup_full_decode=startup["costs"]["full_decode"] if startup else 0,
                               **{"endpoint_"+k:v for k,v in d["endpoint_calls"].items()},
                               max_sampled_CFL=d["max_sampled_CFL"],
                               solver_seconds=d["solver_seconds"],endpoint_seconds=d["endpoint_seconds"],
                               complete_session_seconds=d["complete_session_seconds"]))
    return states,status


def figures(where, forecast, spec, stats, reference_field, native_field):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(MODELS.values(),("#FF1766","#00A6E6","#FF851B","#8A2BE2","#00C878")))
    fig, axes = plt.subplots(2,2,figsize=(9.6,6.6),layout="constrained")
    a,b,cax,d = axes.flat
    # Concatenate a native and filtered field after spectral interpolation of
    # neither. Each saved grid retains its own pixel sampling.
    vmax = max(abs(native_field).max(),abs(reference_field).max())
    a.set_axis_off()
    field_axes = [a.inset_axes(rect) for rect in ((0,.13,.44,.75),(.49,.13,.44,.75))]
    for ax,w,label in zip(field_axes,(native_field,reference_field),("DNS","FDNS")):
        im = ax.imshow(w,origin="lower",cmap="RdBu_r",vmin=-vmax,vmax=vmax,
                       extent=(0,2*np.pi,0,2*np.pi),interpolation="nearest")
        ax.set_title(label,fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    a.set_title("(a) Vorticity fields")
    bar = a.inset_axes([.25,.05,.50,.035])
    fig.colorbar(im,cax=bar,orientation="horizontal")
    d.loglog(np.arange(1,48),spec["FDNS",None][1],color="black",label="FDNS")
    for (m,seed), arr in spec.items():
        if m == "FDNS":
            continue
        group = [x for x in forecast if (x["model"],x["seed"]) == (m,seed)]
        label = f"{m} {seed}" if seed else m
        ls = "--" if seed == 2028 else "-"
        b.plot([x["lead_time"] for x in group],[x["CC"] for x in group],
               ls=ls,color=colors[m],label=label)
        d.loglog(np.arange(1,48),arr[1],ls=ls,color=colors[m])
    b.set(xlabel="Lead time",ylabel="Vorticity spatial CC",title="(b) Single-start forecast",ylim=(-1,1.05))
    b.legend(fontsize=7,ncol=2,frameon=False)
    labels = [f"{r['model']} {r['seed'] or ''}".strip() for r in stats]
    cax.barh(labels,[r["Z"] for r in stats],color=[colors.get(r["model"],"#303030") for r in stats])
    cax.invert_yaxis()
    cax.set(xlabel=r"Window-mean $Z$",title="(c) Resolved enstrophy")
    d.set(xlabel=r"Wavenumber $k$",ylabel=r"$Z(k)$",xlim=(1,47),
          title="(d) Window-mean enstrophy spectrum")
    d.legend(frameon=False)
    fig.suptitle("Exploratory · native DNS dissipation-tail check failed",fontsize=10)
    for ext in ("pdf","png"):
        fig.savefig(where/f"FORCED_FLOW_SUMMARY.{ext}",dpi=320)
    plt.close(fig)


def selected_references(out, cfg, receipt, tag):
    if tag not in receipt["variant_frames"]:
        raise ValueError(f"Reference {tag} has not been audited. Complete both refinement variants and run qualify.")
    window = receipt["window"]
    data = []
    for i,meta,frame in reference_archive(out,cfg,tag,window):
        if meta["frame_sha256"] != receipt["variant_frames"][tag][str(i)]:
            raise ValueError("Reference differs from the qualification receipt")
        data.append(frame)
    return data


def score_archive(out, cfg, dest, attempts, reference):
    ref_file, receipt = verify_pilot_input(cfg)
    window = receipt["window"]
    ref = selected_references(PILOT_SOURCE,cfg,receipt,reference)
    states,status = read_trajectories(out,cfg,receipt,attempts)
    dest.mkdir(parents=True,exist_ok=False)
    c.csvout(dest/"STATUS_AND_COST.csv",status)
    stat_start = window["start"]+cfg["les"]["sample_dt"]
    ref_window = np.stack([f["reference_omega"] for f in ref if float(f["time_s"]) >= stat_start-1e-10])
    sigma = float(np.sqrt(np.mean(ref_window**2)))
    if sigma <= 0:
        raise ValueError("Degenerate FDNS RMS")
    bins = {"raw":sigma*np.linspace(-12,12,241),"reference_scaled":np.linspace(-12,12,241)}
    forecast, frame_errors, paired, summary, spec_rows, pdf_rows, all_specs = [], [], [], [], [], [], {}
    for (name,seed), traj in {("FDNS",None):ref,**states}.items():
        ident = dict(model=name,seed=seed,reference_quality=PILOT_QUALITY)
        stats,spec,pdfs = long_statistics(traj,stat_start,bins,sigma)
        biases = {}
        if name == "FDNS":
            ref_stats, ref_spec = stats, spec
        else:
            for j,q in enumerate(("K","Z")):
                biases[q+"_relative_bias"] = (stats[q]-ref_stats[q])/ref_stats[q]
                a,b = (spec[j],ref_spec[j])
                biases[q+"_spectrum_L1"] = abs(a-b).sum()/b.sum()
        # This uses the RMS pooled over the configured time window, equivalently sqrt(2<K>).
        summary.append(dict(ident,**stats,**rms_reynolds(stats["K"],cfg),**biases))
        all_specs[name,seed] = spec
        spec_rows.extend(dict(ident,k=k+1,E=e,Z=z) for k,(e,z) in enumerate(spec.T))
        for mode,(density,below,above) in pdfs.items():
            edges = bins[mode]
            pdf_rows.extend(dict(ident,variable=mode,bin_left=edges[i],bin_right=edges[i+1],
                                 pdf=y,below_mass=below,above_mass=above) for i,y in enumerate(density))
        if name != "FDNS":
            errors = []
            for f,r in zip(traj,ref):
                t = float(f["lead_time"])
                row = dict(ident,time_s=float(f["time_s"]),lead_time=t,**scores(f["omega"],r["reference_omega"]))
                if t <= cfg["les"]["forecast_duration"]+1e-10:
                    forecast.append(row)
                if t > 0:
                    errors.append(row)
            count = index(cfg["les"]["duration"],cfg["les"]["sample_dt"])
            if len(errors) != count:
                raise ValueError(f"Expected {count} matched scoring frames for {name}/{seed}")
            frame_errors.extend(errors)
            paired.append(dict(ident,samples=count,window_omega_MSE=float(np.mean([r["omega_MSE"] for r in errors])),
                               final_omega_MSE=errors[-1]["omega_MSE"],final_relative_L2=errors[-1]["relative_L2"],
                               final_CC=errors[-1]["CC"]))
    if paired:
        by_model = {(r["model"],r["seed"]):r["window_omega_MSE"] for r in paired}
        for row in paired:
            candidate = row["window_omega_MSE"]
            for label, key in (("NGM4",("NGM4",None)),("Aug",("DSF-Aug",row["seed"]))):
                base = by_model.get(key)
                row["reduction_vs_"+label+"_pct"] = 100*(1-candidate/base) if base else None
        c.csvout(dest/"PAIRED_MSE_80.csv",paired)
        c.csvout(dest/"FRAME_ERRORS_80.csv",frame_errors)
    c.csvout(dest/"LONG_STATISTICS.csv",summary)
    c.csvout(dest/"MEAN_SPECTRA.csv",spec_rows)
    c.csvout(dest/"VORTICITY_PDF.csv",pdf_rows)
    c.dump(dest/"REPORT.json",dict(reference=reference,reference_sha256=c.sha(ref_file),
                                  window=window,statistics_start=stat_start,reference_RMS=sigma,
                                  forecast_origins=1,completed_trajectories=len(states),attempts=attempts,
                                  short_phase_checks_pass=receipt["short_phase_checks_pass"],
                                  statistical_reference_checks_pass=receipt["statistical_reference_checks_pass"],
                                  reference_quality=PILOT_QUALITY,numerical_reference_status=receipt["status"],
                                  physics_operator_pass=receipt["physics_operator_pass"],
                                  dissipation_tail=receipt["variants"]["primary"]["max_tail_dissipation"],
                                  dissipation_tail_limit=cfg["gates"]["max_native_dissipation_tail"]))
    if forecast:
        c.csvout(dest/"SHORT_FORECAST.csv",forecast)
        dt = cfg["dns"][reference]["dt"]
        mid = (window["start"]+window["end"])/2
        h = c.load(PILOT_SOURCE/"DNS"/reference/"blocks"/f"{index(mid,dt):08d}.npz")["omega_hat"]
        native = np.fft.ifft2(h,norm="forward").real
        f = ref[index(mid-window["start"],cfg["les"]["sample_dt"])]["reference_omega"]
        figures(dest,forecast,all_specs,summary,f,native)


@torch.inference_mode()
def replay(out, cfg, dest, device):
    """Sparse common-state a priori scores. History methods need finer archives."""
    ref_file = reference_receipt(out)
    receipt = c.read(ref_file)
    ref = selected_references(out,cfg,receipt,"primary")
    grid = Grid(cfg["les"]["n"],cfg["length"],device)
    rows = checkpoint_rows(cfg)
    results = []
    for method in ("Clark4","Aug","Fresh-C4"):
        for seed in ([None] if method=="Clark4" else cfg["les"]["seeds"]):
            e = None
            if seed is None:
                s,_ = clark_solver(cfg,grid)
            else:
                s = c.model(rows[seed],device)
                configure_les(s,cfg,grid)
                e = closure.Engine(s,method,Counter(),out,dest,{})
            try:
                for f in ref[::index(1,cfg["les"]["sample_dt"])]:
                    w = torch.as_tensor(f["reference_omega"],device=device)
                    if e is None:
                        field = s.closure_fields(s.state(w))
                        src = closure.D(s,torch.stack((field["qx"],field["qy"])))
                    else:
                        field = e.flux(s.state(w))
                        src = field["Source"]
                    y = closure.cpu(src)
                    r = f["reference_source"]
                    t = transfer(f["reference_omega"],y,closure.cpu(field["qx"]),closure.cpu(field["qy"]))
                    results.append(dict(model=MODELS[method],seed=seed,time_s=float(f["time_s"]),
                                        Source_MSE=float(np.mean((y-r)**2)),Source_CC=scores(y,r)["CC"],
                                        T_K=t["T_K"],T_Z=t["T_Z"]))
            finally:
                if e is not None:
                    e.close()
    dest.mkdir(parents=True,exist_ok=False)
    c.csvout(dest/"COMMON_STATE_APRIORI.csv",results)
    c.dump(dest/"REPORT.json",dict(methods=["NGM4","DSF-Aug","DSF-Sym"],stride=1.0,
                                  reference_quality=receipt["status"],
                                  reference_sha256=c.sha(ref_file)))


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=DEFAULT_RUN)
    parser.add_argument("--name",required=True)
    parser.add_argument("--reference",default="primary",choices=("primary",))
    parser.add_argument("--attempt-map",type=Path)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+",v) for v in (args.run,args.name)):
        parser.error("Use simple run and report names")
    c.HERE = HERE
    cfg = c.read(PROTOCOL)
    out = HERE/"runs"/args.run
    dest = out/"reports"/args.name
    if dest.exists():
        parser.error("Report already exists. Use a new name.")
    frozen(out,cfg)
    use = c.read(args.attempt_map) if args.attempt_map else {}
    if any(not isinstance(v,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",v) for v in use.values()):
        parser.error("Attempt map must contain simple attempt names")
    score_archive(out,cfg,dest,use,args.reference)


if __name__ == "__main__":
    cli()
