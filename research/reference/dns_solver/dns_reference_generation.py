"""Explicit-execution, resumable independent reference generation; no networks."""
import os
import time
from pathlib import Path
import numpy as np
import torch
from rp_common import read, write, sha
from rp_spectral import SpectralRK4, Grid, labels, raw_diagnostics


def atomic_array(path, value, archive=False):
    """Never replace a committed snapshot; checkpoint replacement is explicit elsewhere."""
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(value) or any(not np.array_equal(z[k], v) for k, v in value.items()):
                raise ValueError('Existing snapshot differs; refusing overwrite: ' + str(path))
        return
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez(f, **value)
    os.replace(temp, path)


def save_checkpoint(directory, h, state):
    # Alternating immutable files: a crash before the JSON commit leaves the old pointer usable.
    name = f'checkpoint_{state["step"]:07d}.npz'
    path = directory / name
    atomic_array(path, dict(h=h.detach().cpu().numpy()))
    old = read(directory/'CHECKPOINT.json') if (directory/'CHECKPOINT.json').exists() else None
    write(directory/'CHECKPOINT.json', dict(**state, file=name, sha256=sha(path)))
    if old and old['file'] != name:
        (directory/old['file']).unlink()


def task_verified(directory):
    m = read(directory/'DONE.json')
    for name, digest in m['outputs_sha256'].items():
        p = (directory/name).resolve()
        if not p.is_relative_to(directory.resolve()) or sha(p) != digest:
            raise ValueError('New reference output hash mismatch: '+name)
    return m


def generate(run, frozen, tag, device, resume=False):
    p = frozen['protocol']; c = frozen['config']; variant = p['new_variants'][tag]
    d = run/tag
    if (d/'DONE.json').exists():
        return task_verified(d)
    if d.exists() and not resume:
        raise FileExistsError('Partial task exists; use --resume after inspecting PROGRESS.json')
    d.mkdir(parents=True, exist_ok=True)
    lock = d/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        return _generate(d, run, frozen, tag, device, resume)
    except Exception as exc:
        write(d/'LAST_ERROR.json', dict(type=type(exc).__name__, message=str(exc),
              classification='BLOCKED_RESOURCE_LIMIT' if isinstance(exc, TimeoutError) else 'FAILED_EXECUTION',
              resume_supported=(d/'CHECKPOINT.json').exists()))
        raise
    finally:
        lock.unlink(missing_ok=True)


def _generate(d, run, frozen, tag, device, resume):
    c, p = frozen['config'], frozen['protocol']
    spec = p['new_variants'][tag]; dt = spec['dt']
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    solver = SpectralRK4(spec['n'], c['length'], c['nu'], device)
    low = Grid(c['les_n'], c['length'], device, torch.float64)
    start, end = c['window']
    times = (start/dt, end/dt, c['record_dt']/dt, p['diagnostics']['checkpoint_interval_s']/dt)
    if any(abs(x-round(x)) > 1e-8 for x in times):
        raise ValueError('Unaligned integration/output/checkpoint times')
    first, total, every, checkpoint_every = map(round, times)
    h = solver.initial(c['seed'], c['initial'])
    initial, _ = raw_diagnostics(solver, h)
    z0, e0 = initial['Z'], initial['E']
    initial_modes = h.detach().cpu().numpy()
    # Hash ordered coefficients in the common |kx|,|ky|<=12 box, independent of array size.
    import hashlib
    k = np.arange(-c['initial']['initial_max_mode'], c['initial']['initial_max_mode']+1)
    initial_mode_hash = hashlib.sha256(initial_modes[k[:, None] % spec['n'], k[None, :] % spec['n']].tobytes()).hexdigest()
    begin = time.perf_counter()
    frames, hist, spectra = [], [], []
    step0 = 0; elapsed_before = 0.; iz = torch.zeros((), dtype=torch.float64, device=device); ie = iz.clone()
    if resume and (d/'CHECKPOINT.json').exists():
        cp = read(d/'CHECKPOINT.json')
        if cp['freeze_sha256'] != sha(run/'FREEZE_V2.json') or sha(d/cp['file']) != cp['sha256']:
            raise ValueError('Checkpoint identity/hash mismatch')
        with np.load(d/cp['file'], allow_pickle=False) as z:
            h = torch.as_tensor(np.array(z['h']), device=device)
        step0 = cp['step']; frames = cp['frames']; hist = cp['history']; spectra = cp['spectra']
        for name, digest in cp['snapshot_hashes'].items():
            if sha(d/name) != digest:
                raise ValueError('Checkpoint snapshot hash mismatch')
        iz = torch.tensor(cp['integrated_Z'], device=device, dtype=torch.float64)
        ie = torch.tensor(cp['integrated_E'], device=device, dtype=torch.float64)
        solver.max_speed = torch.tensor(cp['max_speed'], device=device, dtype=torch.float64)
        elapsed_before = cp['elapsed_seconds']
    meta = dict(version=p['version'], tag=tag, initial_seed=c['seed'], role=c['role'],
                n=spec['n'], dt_s=dt, length=c['length'], nu=c['nu'], filter_width=c['delta'],
                low_grid=[low.n, low.n], cutoff=solver.grid.cut, dtype='float64',
                fft_normalization='forward', repeated_periodic_endpoint=False,
                source_definition=p['reference_source'], legacy_source_definition=p['legacy_source'],
                initial_common_coefficients_sha256=initial_mode_hash,
                initial_hash_scope='byte hash records this run; compare common coefficients with tolerance, not cross-resolution byte equality',
                device=str(device), torch_version=torch.__version__,
                device_name=torch.cuda.get_device_name() if torch.device(device).type == 'cuda' else 'CPU',
                DNS_claim=False, closure_inference=False, training=False,
                freeze_sha256=sha(run/'FREEZE_V2.json'))
    write(d/'TASK_MANIFEST.json', meta)

    def save(step):
        snapshot_hashes = {name: sha(d/name) for name in frames}
        for row in hist:
            name = row.get('high_file')
            if name:
                snapshot_hashes[name] = sha(d/name)
        save_checkpoint(d, h, dict(step=step, frames=frames, history=hist, spectra=spectra,
            integrated_Z=float(iz), integrated_E=float(ie), max_speed=float(solver.max_speed),
            elapsed_seconds=elapsed_before+time.perf_counter()-begin,
            snapshot_hashes=snapshot_hashes, freeze_sha256=sha(run/'FREEZE_V2.json')))

    with torch.inference_mode():
        for step in range(step0, total+1):
            if step >= first and (step-first) % every == 0:
                name = f'frame_{step:07d}.npz'
                if name not in frames:
                    lab = labels(solver, low, h, c['delta'], c['nu'])
                    atomic_array(d/name, dict(time_s=np.array(step*dt), **lab))
                    high_name = f'high_{step:07d}.npz'
                    atomic_array(d/high_name, dict(omega=solver.grid.ifft(h).cpu().numpy()))
                    obs, spectrum = raw_diagnostics(solver, h)
                    frames.append(name)
                    hist.append(dict(step=step, time_s=step*dt, high_file=high_name, **obs,
                        Z_balance_relative=abs(obs['Z']-z0-float(iz))/max(z0, 1e-30),
                        E_balance_relative=abs(obs['E']-e0-float(ie))/max(e0, 1e-30)))
                    spectra.append(spectrum.tolist())
            if step % p['diagnostics']['progress_every_steps'] == 0 or step == total:
                speed = float(solver.max_speed)
                cfl = dt*(speed/(c['length']/spec['n'])+4*c['nu']/(c['length']/spec['n'])**2)
                obs, _ = raw_diagnostics(solver, h)
                elapsed = elapsed_before+time.perf_counter()-begin
                progress = dict(status='RUNNING', tag=tag, step=step, steps=total,
                    time_s=step*dt, max_CFL=cfl, elapsed_seconds=elapsed,
                    estimated_seconds_remaining=(total-step)*((time.perf_counter()-begin)/max(step-step0, 1)),
                    records=len(frames), **obs)
                write(d/'PROGRESS.json', progress)
                print(f'{tag}: {step}/{total} t={step*dt:.4f} wall={elapsed:.1f}s CFL={cfl:.4f}', flush=True)
                if not np.isfinite(speed) or not np.isfinite(obs['Z']) or cfl > p['quality']['CFL']:
                    save(step)
                    raise RuntimeError('Nonfinite/CFL safety stop; not a closure-model verdict')
                if obs['Z'] > p['diagnostics']['enstrophy_ratio_stop']*z0:
                    save(step)
                    raise RuntimeError('Enstrophy safety stop')
                if elapsed > p['diagnostics']['wall_seconds_per_task']:
                    save(step)
                    raise TimeoutError('Per-task wall budget exhausted; checkpoint preserved')
            if step % checkpoint_every == 0 or step == total:
                save(step)
            if step == total:
                break
            h, dz, de = solver.step(h, dt)
            iz += dz; ie += de
    expected = round((end-start)/c['record_dt'])+1
    if len(frames) != expected:
        raise ValueError('Incomplete output time coverage')
    summary = dict(**meta, status='GENERATED_NOT_YET_QUALIFIED', frames=frames,
                   history=hist, raw_vorticity_spectra=spectra,
                   max_tail=max(x['tail_fraction'] for x in hist), max_CFL=cfl,
                   seconds=elapsed_before+time.perf_counter()-begin)
    write(d/'REFERENCE_METADATA.json', summary, exclusive=True)
    outputs = frames + [x['high_file'] for x in hist] + ['REFERENCE_METADATA.json', 'TASK_MANIFEST.json']
    result = dict(status='REFERENCE_COMPLETED', metadata=summary,
                  outputs_sha256={name: sha(d/name) for name in outputs})
    write(d/'DONE.json', result, exclusive=True)
    write(d/'PROGRESS.json', dict(progress, status='REFERENCE_COMPLETED'))
    return result
