"""Teacher-forced SGS measurements at saved states. Never advances a solver."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

FOLDER = Path(__file__).resolve().parent
CASES = (110240101, 110240102)
SEED = 2026
REFERENCE = 'research/ssc_long_horizon_v1_20261030/results/run01/reference'
PRODUCTION = 'research/full_aug_parameter_pc_full_v1_20261116'
FREEZE = 'research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json'
FIELDS = ('reference_omega', 'reference_qx', 'reference_qy', 'reference_source')
MODELS = ('FDNS', 'Clark4', 'DSF-Aug', 'DSF-Sym', 'DSF-Trend', 'DSF-Hold')
HISTORY = {'DSF-Trend': 'Parameter-PC', 'DSF-Hold': 'Parameter-Hold'}


def load_reference(c, root, case, provenance):
    folder = root / REFERENCE / str(case) / 'n1536_half'
    done = c.read(folder / 'DONE.json')
    assert done['seed'] == case and done['variant'] == 'n1536_half'
    assert (done['n'], done['dt'], done['record_dt']) == (1536, .000125, .00125)
    provenance[str((folder / 'DONE.json').relative_to(root))] = c.sha(folder / 'DONE.json')
    arrays = {}
    for name in (*FIELDS, 'time_s'):
        path = folder / (name + '.npy')
        sha = c.sha(path)
        assert sha == done['outputs'][path.name], f'Archive hash mismatch: {path}'
        provenance[str(path.relative_to(root))] = sha
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    np.testing.assert_allclose(arrays['time_s'], 1 + .00125 * np.arange(3201), rtol=0, atol=1e-10)
    assert all(arrays[name].shape == (3201, 100, 100) and arrays[name].dtype == np.float64 for name in FIELDS)
    return arrays


class ForwardCounter:
    def __init__(self):
        self.calls = 0

    def tick(self, kind):
        assert kind == 'CNN', 'A solver RHS was invoked during offline scoring'
        self.calls += 1

    def note(self, kind):
        pass


def pearson(predicted, reference, mask):
    left, right = predicted[:, mask], reference[mask]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean()
    den = np.linalg.norm(left, axis=1) * np.linalg.norm(right)
    assert mask.sum() >= 2 and np.all(den > 0), 'Degenerate spatial CC'
    return np.sum(left * right, axis=1) / den


def score(source, pi, omega, psi, masks):
    positive, negative = pi[0] > 0, pi[0] < 0
    full = np.ones(pi.shape[1:], dtype=bool)
    diff = source - source[0]
    power = abs(np.fft.fft2(diff, axes=(-2, -1), norm='forward')) ** 2
    band_mse = {f'Source_MSE_{key}': power[:, mask].sum(axis=1) for key, mask in masks.items()}
    total = band_mse['Source_MSE_all']
    np.testing.assert_allclose(total, sum(band_mse[f'Source_MSE_{key}'] for key in ('low', 'mid', 'high')),
                               rtol=0, atol=2e-14)
    np.testing.assert_allclose(total, np.mean(diff**2, axis=(1, 2)), rtol=0, atol=2e-14)
    tk = np.mean(psi * source, axis=(1, 2))
    tz = np.mean(omega * source, axis=(1, 2))
    pi_mean = np.mean(pi, axis=(1, 2))
    return dict(**band_mse, Pi_Z_MSE=np.mean((pi-pi[0])**2, axis=(1, 2)),
                Source_CC=pearson(source, source[0], full), Pi_Z_CC=pearson(pi, pi[0], full),
                Pi_Z_positive_ref_CC=pearson(pi, pi[0], positive),
                Pi_Z_negative_ref_CC=pearson(pi, pi[0], negative),
                net_TK=tk, net_TZ=tz, mean_Pi_Z=pi_mean,
                budget_residual=np.abs(tz + pi_mean),
                n_positive_ref=positive.sum(), n_negative_ref=negative.sum())


def summarize(data, columns):
    metrics = ['Source_MSE_' + band for band in ('all', 'low', 'mid', 'high', 'corner')]
    metrics += ['Pi_Z_MSE', 'Source_CC', 'Pi_Z_CC', 'Pi_Z_positive_ref_CC', 'Pi_Z_negative_ref_CC',
                'net_TK', 'net_TZ', 'mean_Pi_Z']
    groups = data.groupby(columns, sort=False)
    summary = groups[metrics].mean()
    summary['samples'] = groups.size()
    summary['guard_failures'] = groups.guard_fail.sum()
    summary = summary.reset_index()
    conditioning = [key for key in columns if key != 'model']
    for key in ('net_TK', 'net_TZ'):
        ref = summary.loc[summary.model.eq('FDNS'), conditioning + [key]]
        if conditioning:
            denom = summary[conditioning].merge(ref, on=conditioning, validate='many_to_one')[key]
        else:
            denom = ref[key].item()
        assert np.all(np.abs(denom) > 1e-10), f'Small mean reference flux for {key}'
        summary[key+'_ratio'] = summary[key] / denom
    return summary


@torch.inference_mode()
def replay(c, e, solver, arrays, case, device, max_step):
    counter = ForwardCounter()
    engine = e.Engine(solver, 'Fresh-C4', counter, None, None, None, diagnostics=False)
    masks = {k: e.cpu(v) for k, v in c.q.masks(solver).items()}
    clark = c.q.study.common.prior.c4.flux4
    bank = e.Bank(None)
    output, failures = [], []
    reference_divergence_error = 0.0
    rows_by_step = []
    try:
        for n in range(max_step+1):
            sample = 4*n
            w = torch.as_tensor(np.array(arrays['reference_omega'][sample]), device=device)
            st = solver.state(w)
            states = [st, *(solver.state(c.q.spatial(w, turn)) for turn in range(1, 4))]
            members, raw = [], []
            for state in states:
                native = engine.original(state)
                c.q.physical(solver, state, native)
                members.append(native)
                raw.append(engine.captured_raw)
            heads = torch.stack(raw)
            if n % 5 in (0, 1):
                bank.observe(n, {'step': n, 'heads': heads})
            if n == 0:
                continue
            fluxes = [torch.stack(clark(solver, st, solver.closure.filter_width)),
                      e.vec(members[0], 'qd')+e.vec(members[0], 'qb')]
            mean = c.q.mean_fields(solver, st, members)
            branches = {'DSF-Sym': (mean, members[0])}
            taus = {}
            for model, key in HISTORY.items():
                predicted, tau = bank.predict(n, 0, key)
                assert 0 <= tau <= 4
                if tau == 0:
                    torch.testing.assert_close(predicted, heads, rtol=0, atol=0)
                    branches[model] = (mean, members[0])
                else:
                    decoded = [engine.decode(state, pred) for state, pred in zip(states, predicted)]
                    for state, member in zip(states, decoded):
                        c.q.physical(solver, state, member)
                    branches[model] = (c.q.mean_fields(solver, st, decoded), decoded[0])
                taus[model] = tau
            guards = {}
            for model, (mean_flux, member) in branches.items():
                d, b = e.vec(mean_flux, 'qd'), e.vec(mean_flux, 'qb')
                try:
                    e.verify_physical(solver, st, d, b, member)
                except e.TrajectoryFailure as error:
                    if error.status != 'FAILED_PHYSICAL':
                        raise
                    failures.append(dict(case=case, step=n, model=model, status=error.status,
                                         reason=error.details))
                    guards[model] = True
                fluxes.append(d+b)
            qref = torch.as_tensor(np.stack([arrays[key][sample] for key in ('reference_qx', 'reference_qy')]),
                                   device=device)
            sref = np.asarray(arrays['reference_source'][sample])
            src_check = e.cpu(e.D(solver, qref))
            delta = np.linalg.norm(src_check-sref) / np.linalg.norm(sref)
            reference_divergence_error = max(reference_divergence_error, float(delta))
            assert delta < 5e-6, 'FDNS source/flux archive or operator mismatch'
            model_source = torch.stack([e.D(solver, flux) for flux in fluxes])
            source = np.concatenate((sref[None], e.cpu(model_source)), axis=0)
            grad = torch.stack((st['omega_x'], st['omega_y']))
            pi = e.cpu(-(torch.stack([qref, *fluxes])*grad).sum(dim=1))
            values = score(source, pi, e.cpu(w), e.cpu(st['psi']), masks)
            assert np.all(values['budget_residual'][1:] < 1e-11), 'Model SGS budget identity failed'
            if n%5 == 1:
                # Both history predictions equal the current second anchor here.
                np.testing.assert_allclose(source[3:6], np.broadcast_to(source[3], source[3:6].shape), atol=0, rtol=0)
            for j, model in enumerate(MODELS):
                output.append(dict(case=case, model=model, network_seed=SEED if model.startswith('DSF') else None,
                                   step=n, time_s=1+n*c.H, phase=n%5,
                                   active_first=bank.active[0]['step'] if model in HISTORY else None,
                                   active_last=bank.active[1]['step'] if model in HISTORY else None,
                                   tau=taus.get(model), guard_fail=int(guards.get(model, False)),
                                   **{key: val[j] if isinstance(val, np.ndarray) else val for key, val in values.items()}))
            if n in (10, 400, 800):
                rows_by_step.append((n, source.copy(), pi.copy()))
            if n % 100 == 0 or n == max_step:
                print(f'Case {case % 1000}: {n}/{max_step} states scored, {counter.calls} CNN forwards', flush=True)
    finally:
        engine.close()
    assert counter.calls == 4*(max_step+1)
    return pd.DataFrame(output), failures, counter.calls, reference_divergence_error, rows_by_step


def crosscheck(data):
    # Old Figure 2 used samples at the period-5 pending-refresh phase only.
    # Compare on those exact times, after matching the original model and case.
    old = pd.read_csv(FOLDER/'DATA/apriori_source.csv')
    old = old.loc[old.band.eq('all') & old.time_s.gt(1), ['case', 'model', 'time_s', 'source_mse']].copy()
    old['step'] = np.rint((old.time_s-1)/.005).astype(int)
    checked = data.merge(old, on=['case', 'step', 'model'], validate='one_to_one')
    assert len(checked) == 3 * len(CASES) * int(data.step.max()//10)
    score_difference = abs(checked.Source_MSE_all-checked.source_mse)
    np.testing.assert_allclose(checked.Source_MSE_all, checked.source_mse, rtol=2e-5, atol=1e-9)
    return dict(rows=len(checked), max_absolute_Source_MSE_difference=float(score_difference.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--probe', action='store_true', help='Validate the first two scored legacy timestamps')
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / PRODUCTION))
    import pc_common as c
    import pc_engine as e

    assert c.ROOT == root and c.H == .005
    assert c.P['refresh']['pair_phases'] == [0, 1] and c.P['refresh']['period'] == 5
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    out = FOLDER / 'DATA' / ('FDNS_REPLAY_PROBE' if args.probe else 'FDNS_REPLAY')
    out.mkdir(exist_ok=True, parents=True)
    begin = time.perf_counter()
    freeze = c.read(root / FREEZE)
    assert freeze['status'] == 'FROZEN' and freeze['quality'] == c.QUALITY
    row, = [r for r in freeze['checkpoints'] if r['mode'] == c.q.study.AUGMENTED and r['seed'] == SEED]
    files = dict(freeze['training_code_sha256']) | dict(freeze['baseline_code_sha256'])
    files.update({row[key]:row[sha] for key, sha in [('file','sha256'), ('run','run_sha256'), ('done','done_sha256')]})
    files[str(c.q.study.common.CONFIG.relative_to(root))] = freeze['baseline_config_sha256']
    for path, expected in files.items():
        assert c.sha(root/path) == expected, f'Frozen input changed: {path}'
    expected_decoder = c.read(root/'research/full_aug_orbit_rk4_one_step_v1_20261114/PHASE_ONE_MANIFEST.json')
    assert c.sha(Path(c.decoder.__file__)) == expected_decoder['current_code_sha256']['common_step.py']
    files[FREEZE] = c.sha(root / FREEZE)
    for module in (c, e, c.q, c.feasible, c.decoder, c.q.study.common.prior.c4):
        p = Path(module.__file__)
        files[str(p.relative_to(root))] = c.sha(p)
    solver = c.model(row, device)
    steps = 20 if args.probe else 800
    all_rows, all_fails, all_fields, calls, max_identity_error = [], [], {}, 0, 0.
    print(f'Frozen seed {SEED}: {row["sha256"]}; device {device}; {steps} states per case', flush=True)
    for case in CASES:
        arrays = load_reference(c, root, case, files)
        data, fails, count, identity_error, fields = replay(c, e, solver, arrays, case, device, steps)
        all_rows.append(data)
        all_fails.extend(fails)
        all_fields.update({f'case{case}_step{n}_{name}': val for n, s, p in fields for name, val in [('Source', s), ('Pi_Z', p)]})
        calls += count
        max_identity_error = max(max_identity_error, identity_error)
    data = pd.concat(all_rows, ignore_index=True)
    validation = crosscheck(data)
    assert len(data) == len(CASES)*len(MODELS)*steps
    assert data.groupby(['case','model','phase']).size().eq(steps//5).all()
    data.to_csv(out/'STATE_METRICS.csv', index=False)
    summarize(data, ['model']).to_csv(out/'SUMMARY.csv', index=False)
    summarize(data, ['case', 'model']).to_csv(out/'BY_FLOW.csv', index=False)
    summarize(data, ['phase', 'model']).to_csv(out/'BY_REFRESH_PHASE.csv', index=False)
    np.savez_compressed(out/'FIELD_EXAMPLES.npz', models=np.array(MODELS), **all_fields)
    (out/'GUARD_EVENTS.json').write_text(json.dumps(all_fails, default=c.clean, indent=2), encoding='utf-8')
    receipt = dict(status='COMPLETE_WITH_GUARD_EVENTS' if all_fails else 'COMPLETE',
                   reference_tag='n1536_half', reference_quality=c.QUALITY, cases=CASES, seed=SEED,
                   checkpoint_sha256=row['sha256'], score_times=f'1+n*0.005, n=1..{steps}',
                   states_per_case=steps, measured_CNN_forwards=calls, new_DNS_steps=0, new_LES_steps=0,
                   new_RHS_calls=0, network_training_steps=0, model_source_replay=validation,
                   max_reference_source_flux_relative_error=max_identity_error, guard_events=len(all_fails),
                   fdns='archived filtered states and flux, no solver advance, no interpolation',
                   elapsed_s=time.perf_counter()-begin, device=device, input_sha256=files,
                   outputs_sha256={p.name:c.sha(p) for p in out.iterdir() if p.is_file() and p.name!='RECEIPT.json'})
    (out/'RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(f'COMPLETE. Offline forwards={calls}; independent archive replay={validation}; guard events={len(all_fails)}', flush=True)


if __name__ == '__main__':
    main()
