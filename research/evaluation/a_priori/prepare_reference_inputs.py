"""Assemble matched archived diagnostics and rescore saved LES states with NumPy."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = Path('research/JCP_DSF_TemporalSym_20261117')
VALIDATION = STUDY / 'VALIDATION_20261118'
CASES = (110240101, 110240102)
SEEDS = (2026, 2028)
OFFLINE_MODELS = {'Clark4': 'Clark4', 'FULL-C4-Augmented': 'DSF-Aug',
                  'FULL-C4-Augmented-rotation-mean': 'DSF-Sym'}
LES_MODELS = {'Aug': 'DSF-Aug', 'Fresh-C4': 'DSF-Sym',
              'Parameter-PC': 'DSF-Trend', 'Parameter-Hold': 'DSF-Hold'}
VARIANTS = {'n1024_main': (1024, .00025), 'n1536_main': (1536, .00025),
            'n1536_half': (1536, .000125)}
KEYS = ['case', 'model', 'seed', 'time_s']
TIMES = 1 + np.arange(81) * .05
PROVENANCE = {}


def tracked(path, expected=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    sha = digest.hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f'Manifest hash mismatch: {path}')
    PROVENANCE[str(path)] = sha
    return path


def read(path):
    return json.loads(tracked(path).read_text(encoding='utf-8-sig'))


def common_offline(frame):
    data = frame.loc[frame.model.isin(OFFLINE_MODELS) & frame.case.isin(CASES)
                     & frame.seed.fillna(2026).eq(2026)].copy()
    data.model = data.model.map(OFFLINE_MODELS)
    return data


def check_times(frame):
    assert len(frame) == 486 and not frame.duplicated(KEYS).any()
    for _, rows in frame.groupby(['case', 'model']):
        np.testing.assert_allclose(np.sort(rows.time_s), TIMES, atol=1e-10, rtol=0)


def apriori(root, out):
    current = root / 'research/full_c4_augmented_orbit_v1_20261110/results/run01'
    archive = root / 'reports/20261116_retired_research/full_c4_augmented_orbit_v1_20261110/results/run01'
    receipt = read(current / 'MANIFEST.json')
    assert receipt['status'] == 'PASS' and receipt['quality'] == 'REFERENCE_NOT_CONVERGED'
    paths = [tracked(folder / name, receipt['outputs'][name]) for folder, name in (
        (current, 'SOURCE.csv'), (archive, 'LOCAL_TRANSFER.csv'), (archive, 'REFERENCE_SPECTRA.csv'))]
    source, transfer, spectra = [common_offline(pd.read_csv(path)) for path in paths]
    source.band = source.band.replace({'middle': 'mid', 'tail': 'high'})
    for _, rows in source.groupby('band'):
        check_times(rows)
    np.testing.assert_allclose(source[source.band.isin(['low', 'mid', 'high'])].groupby(
        KEYS, dropna=False).source_mse.sum().sort_index(), source[source.band == 'all'].set_index(
        KEYS).source_mse.sort_index(), atol=2e-14, rtol=0)
    check_times(transfer)
    assert len(transfer.merge(source[source.band == 'all'], on=KEYS, validate='one_to_one')) == 486
    metrics = ['TK', 'TZ', 'TK_ref', 'TZ_ref']
    assert (transfer[['true_positive', 'false_positive', 'false_negative']] >= 0).all().all()
    f1 = 2 * transfer.true_positive / (2 * transfer.true_positive + transfer.false_positive + transfer.false_negative)
    np.testing.assert_allclose(f1, transfer.backscatter_f1, atol=2e-14, rtol=0)
    spectra = spectra.sort_values(KEYS + ['k'])
    groups = spectra.groupby(KEYS, dropna=False, sort=False)
    sum_spectra = groups[metrics].sum().reset_index()
    merged = transfer.merge(sum_spectra, on=KEYS, validate='one_to_one', suffixes=('', '_sum'))
    assert len(merged) == 486
    for col in metrics:
        np.testing.assert_allclose(merged[col], merged[col + '_sum'], atol=3e-14, rtol=0)
    ref = spectra.groupby(['case', 'time_s', 'k'])[['TK_ref', 'TZ_ref']].agg(['min', 'max'])
    for col in ['TK_ref', 'TZ_ref']:
        np.testing.assert_allclose(ref[(col, 'min')], ref[(col, 'max')], atol=3e-14, rtol=0)
    skill = []
    for key, group in groups:
        np.testing.assert_array_equal(group.k, np.arange(1, 48))
        skill.append(dict(zip(KEYS, key)) | {
            field + '_spectral_cc': np.corrcoef(group[field], group[field + '_ref'])[0, 1]
            for field in ('TK', 'TZ')})
    transfer = transfer.merge(pd.DataFrame(skill), on=KEYS, validate='one_to_one')
    assert np.isfinite(transfer[['TK_spectral_cc', 'TZ_spectral_cc']]).all().all()
    source.to_csv(out / 'apriori_source.csv', index=False)
    transfer.to_csv(out / 'apriori_transfer.csv', index=False)
    spectra.to_csv(out / 'apriori_transfer_spectra.csv', index=False)
    summary = transfer.groupby('model')[metrics + ['pi_mse', 'backscatter_f1', 'TK_spectral_cc', 'TZ_spectral_cc']].mean()
    for field in ('TK', 'TZ'):
        assert not np.isclose(summary[field + '_ref'], 0).any()
        summary[field + '_net_ratio'] = summary[field] / summary[field + '_ref']
    summary['source_mse'] = source[source.band == 'all'].groupby('model').source_mse.mean()
    summary.to_csv(out / 'apriori_summary.csv')


def rescore(root, out):
    old = root / 'research/full_aug_parameter_pc_full_v1_20261116/results/run01'
    audit = root / 'research/full_aug_hold_audit_v1_20261117/results/reference01'
    reference = root / 'research/ssc_long_horizon_v1_20261030/results/run01/reference'
    validation = root / VALIDATION
    independent = validation / read(validation / 'LATEST_REPORT.json')['path']
    report = old / read(old / 'LATEST_REPORT.json')['path']
    prior_scores = pd.read_csv(tracked(independent / 'PAIRED_ACCURACY.csv'))
    legacy = pd.read_csv(tracked(audit / 'EXISTING_REFERENCE_RANKING.csv'))
    legacy = legacy.loc[legacy.window.eq('full') & legacy.band.eq('all')]
    inputs = read(audit / 'INPUT_SHA256.json')
    state_hashes = read(report / 'MANIFEST.json')['outputs']
    op = old / 'OPERATORS.npz'
    with np.load(tracked(op, inputs[str(op.relative_to(root))]), allow_pickle=False) as operators:
        mask = operators['dealias'].copy()
    mask[0, 0] = False
    scores = []
    steps = np.arange(10, 801, 10)

    for case in CASES:
        refs = {}
        for variant, (size, dt) in VARIANTS.items():
            folder = reference / str(case) / variant
            done = read(folder / 'DONE.json')
            assert done['n'] == size and done['dt'] == dt
            paths = [tracked(folder / file, inputs[str((folder / file).relative_to(root))])
                     for file in ('time_s.npy', 'reference_omega.npy')]
            t = np.load(paths[0], allow_pickle=False)
            indices = np.searchsorted(t, TIMES[1:] - 1e-10)
            np.testing.assert_allclose(t[indices], TIMES[1:], atol=1e-10, rtol=0)
            omega = np.load(paths[1], mmap_mode='r', allow_pickle=False)[indices]
            refs[variant] = np.fft.ifft2(np.fft.fft2(omega, axes=(-2, -1)) * mask, axes=(-2, -1)).real
        clark = validation / 'CLARK' / str(case) / 'limited_completion/Clark4_0.005'
        status = read(clark / 'STATUS.json')
        assert status['status'] == 'VERIFIED' and status['last_accepted_step'] == 800
        states = []
        for step in steps:
            with np.load(tracked(clark / 'steps' / f'{step:04d}.npz'), allow_pickle=False) as record:
                states.append(record['omega'])
        models = [('Clark4', None, np.asarray(states))]
        for seed in SEEDS:
            for key, model in LES_MODELS.items():
                name = f'STATES_{case}_{seed}_{key}.npy'
                array = np.load(tracked(report / name, state_hashes[name]), mmap_mode='r', allow_pickle=False)
                assert array.shape == (801, 100, 100)
                models.append((model, seed, array[steps]))
        for model, seed, field in models:
            for variant, omega in refs.items():
                delta = np.fft.fft2(field - omega, axes=(-2, -1), norm='forward')
                mse = float(np.mean(np.sum(abs(delta[:, mask])**2, axis=1)))
                scores.append(dict(case=case, model=model, seed=seed, reference_variant=variant,
                                   n=VARIANTS[variant][0], reference_dt=VARIANTS[variant][1],
                                   samples=80, full_omega_MSE=mse))
                if model in ('DSF-Sym', 'DSF-Hold'):
                    expected = legacy.loc[legacy.case.eq(case) & legacy.seed.eq(seed)
                                          & legacy.reference_variant.eq(variant)]
                    col = 'Fresh_MSE' if model == 'DSF-Sym' else 'Hold_MSE'
                    np.testing.assert_allclose(mse, expected[col].item(), atol=1e-12, rtol=0)
                if model == 'Clark4' and variant == 'n1536_half':
                    expected = prior_scores.loc[prior_scores.case.eq(case) & prior_scores.model.eq('Clark4')]
                    np.testing.assert_allclose(mse, expected.full_omega_MSE.item(), atol=1e-12, rtol=0)
    data = pd.DataFrame(scores)
    data.to_csv(out / 'reference_model_scores.csv', index=False)
    ranking = []
    for (case, variant), subset in data.groupby(['case', 'reference_variant']):
        for seed in SEEDS:
            pair = subset.loc[subset.seed.eq(seed) | subset.model.eq('Clark4')].sort_values('full_omega_MSE')
            assert len(pair) == 5
            hold = pair.loc[pair.model.eq('DSF-Hold'), 'full_omega_MSE'].item()
            other = pair.loc[~pair.model.eq('DSF-Hold')].iloc[0]
            ranking.append(dict(case=case, seed=seed, reference_variant=variant,
                                winner=pair.model.iloc[0], best_alternative=other.model,
                                hold_MSE=hold, alternative_MSE=other.full_omega_MSE,
                                hold_advantage_pct=100*(1-hold/other.full_omega_MSE),
                                ranking=' > '.join(pair.model)))
    pd.DataFrame(ranking).to_csv(out / 'reference_rankings.csv', index=False)


def reference_diagnostics(root, out):
    quality = root / 'research/ssc_long_horizon_v1_20261030/results/run01/quality'
    rows = []
    for case in CASES:
        folder = quality / str(case)
        dataset = pd.read_csv(tracked(folder / 'CONVERGENCE.csv'))
        assert len(dataset) == 18 and dataset.threshold.eq(.01).all()
        rows.append(dataset.assign(case=case))
    pd.concat(rows, ignore_index=True).to_csv(out / 'reference_filtered_checks.csv', index=False)
    validation = root / VALIDATION
    folder = validation / read(validation / 'LATEST_QUALIFICATION.json')['path']
    quality_summary = read(folder / 'QUALIFICATION.json')
    tail = pd.read_csv(tracked(folder / 'TAIL_CFL.csv'))
    assert tail.n.eq(1536).all() and np.isclose(tail.dt, .000125).all()
    status = pd.DataFrame(quality_summary['cases'])[['case', 'status', 'missing_variants']]
    tail.merge(status, on='case', validate='one_to_one').to_csv(out / 'reference_tail_cfl.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=HERE.parents[2], type=Path)
    args = parser.parse_args()
    out = HERE / 'DATA'
    out.mkdir(exist_ok=True)
    apriori(args.project_root, out)
    reference_diagnostics(args.project_root, out)
    rescore(args.project_root, out)
    meta = dict(sources_sha256=PROVENANCE, CNN_calls=0, new_RHS_calls=0, temporal_interpolation=False,
                reference_quality='REFERENCE_NOT_CONVERGED')
    (out / 'PROVENANCE.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
