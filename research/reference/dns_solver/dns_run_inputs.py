"""Metadata-only scope, input identity, freezing and atomic output helpers."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECISIVE = HERE.parent
STUDY = DECISIVE.parent
ROOT = STUDY.parents[1]
CAMPAIGN = DECISIVE / 'results/campaign01'
for path in (ROOT, STUDY.parent, STUDY, DECISIVE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            f.write(content)
    else:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(content, encoding='utf-8')
        os.replace(temp, path)


def protocol():
    return read(HERE / 'protocol.json')


def scope(seed):
    old = read(DECISIVE / 'DECISION_RULES.json')
    if seed not in old['roles']['development']:
        raise PermissionError('Only registered development seeds are allowed; locked/reserved seeds forbidden')
    return 'development'


def config(seed):
    scope(seed)
    from dv_common import freeze_verify
    old = freeze_verify(CAMPAIGN)
    audit = old['audit']
    p = read(audit['frozen']['runs'][0]['config'])
    d = p['domain']
    lx, ly = d['x_max_m'] - d['x_min_m'], d['y_max_m'] - d['y_min_m']
    if lx != ly or len(set(audit['frozen']['grid'])) != 1:
        raise ValueError('Square domain/grid required; no inferred substitute')
    if p['data']['filter_kind'] != 'gaussian' or p['physics']['forcing_enabled']:
        raise ValueError('This platform is restricted to the audited Gaussian decaying case')
    if abs(p['paper']['dealias_fraction'] - 2/3) > 1e-14:
        raise ValueError('Strict 2/3 Fourier operator required')
    nu = p['physics']['viscosity_value']
    if p['physics']['viscosity_kind'] == 'dynamic':
        nu /= p['physics']['density_kg_m3']
    elif p['physics']['viscosity_kind'] != 'kinematic':
        raise ValueError('Unknown viscosity units')
    if abs(nu - audit['original_nu']) > 1e-18:
        raise ValueError('Frozen viscosity mismatch')
    source = read(STUDY / 'protocol.json')['independent']
    return dict(seed=seed, role='development', audit=audit, length=lx,
                les_n=audit['frozen']['grid'][0], nu=nu,
                delta=p['data']['filter_width_m'], initial=source,
                window=[source['window_start_s'], source['window_start_s'] + source['window_duration_s']],
                record_dt=source['reference_record_dt_s'])


def run_path(value, seed):
    p = (Path(value) if value else HERE / 'results' / f'development_{seed}_v2_01').resolve()
    if not p.is_relative_to((HERE / 'results').resolve()):
        raise ValueError('All outputs must stay in the new platform results directory')
    return p


def reused(seed):
    result = {}
    for tag in protocol()['reused_variants']:
        d = CAMPAIGN / 'tasks' / f'ref_{seed}_{tag}'
        m = read(d / 'DONE.json')
        if m['role'] != scope(seed) or m['initial_seed'] != seed or m['status'] != 'COMPLETED':
            raise ValueError('Reused reference identity mismatch')
        for name, digest in m['outputs_sha256'].items():
            p = (d / name).resolve()
            if not p.is_relative_to(d.resolve()) or sha(p) != digest:
                raise ValueError('Reused artifact hash mismatch')
        result[tag] = dict(path=str(d / m['archive_name']), metadata=m['metadata'],
                           sha256=m['outputs_sha256'][m['archive_name']])
    return result


def prepare(run, seed):
    c = config(seed)
    inputs = reused(seed)
    if shutil.disk_usage(HERE).free < protocol()['diagnostics']['minimum_free_disk_GB'] * 1e9:
        raise RuntimeError('Insufficient free disk for high-field snapshots/checkpoints')
    run.mkdir(parents=True, exist_ok=False)
    identities = {str(p): sha(p) for p in HERE.glob('*.py')}
    identities[str(HERE / 'protocol.json')] = sha(HERE / 'protocol.json')
    write(run / 'FREEZE_V2.json', dict(version=protocol()['version'], seed=seed, config=c,
          protocol=protocol(), reused=inputs, identities=identities,
          old_freeze_sha256=sha(CAMPAIGN / 'FREEZE.json')), exclusive=True)
    return verify(run, seed)


def verify(run, seed):
    scope(seed)
    f = read(run / 'FREEZE_V2.json')
    if f['seed'] != seed:
        raise ValueError('Run/seed mismatch')
    for path, digest in f['identities'].items():
        if sha(path) != digest:
            raise ValueError('Frozen v2 code/protocol changed: ' + path)
    config(seed)
    if sha(CAMPAIGN / 'FREEZE.json') != f['old_freeze_sha256']:
        raise ValueError('Historical freeze changed')
    for item in f['reused'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Reused array archive changed')
    return f
