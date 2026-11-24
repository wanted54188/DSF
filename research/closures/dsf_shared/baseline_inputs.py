"""Read-only inputs from the newly trained NS baseline."""

import hashlib
import json
from pathlib import Path
import sys
from types import MethodType


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
BENCH = ROOT / 'research/full_c4_ns_retrain_v1_20261103'
RUN = BENCH / 'results/run01'
REVIEW = ROOT / 'research/full_c4_identifiability_review_20261104'
CONFIG = ROOT / 'results/revision_v2/runs/full/seed_2026_recovery01/effective_config.json'
QUALITY = 'REFERENCE_NOT_CONVERGED'
CASES = (110240101, 110240102)
FULL = (('FULL-C4', 2026), ('FULL-C4', 2028))
MODELS = (*FULL, ('FULL-Prior4', 2026), ('FULL-Prior4', 2028), ('Clark4', None))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and not any(
        str(i) in str(path) for i in (110240201, 110240202, 110240301, 110240302)
    ), f'Input outside authorized paths: {path}')
    return path


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def audit():
    files = {}

    def record(path, expected=None):
        actual = sha(path)
        require(expected is None or actual == expected, f'SHA mismatch: {path}')
        files[str(safe(path).relative_to(ROOT))] = actual

    def load(path):
        record(path)
        return read(path)

    benchmark = load(RUN / 'AUDIT.json')
    registry = load(RUN / 'BASELINE_REGISTRY.json')
    code = benchmark['code_sha256']
    protocol = benchmark['protocol']
    require(benchmark['quality'] == QUALITY and registry['reference_quality'] == QUALITY
            and registry['code_sha256'] == code and len(registry['models']) == 8
            and (protocol['train_seed'], protocol['validation_seed']) == CASES
            and protocol['primary_variant'] == 'n1536_half', 'Baseline identity mismatch')
    for path, digest in code.items():
        record(ROOT / path, digest)
    record(CONFIG, benchmark['files'][str(CONFIG)])
    for case in CASES:
        directory = RUN / 'data' / str(case)
        cache = load(directory / 'DONE.json')
        digest = registry['cache_sha256'][str(case)]
        require(cache['seed'] == case and cache['frames'] == 81 and cache['quality'] == QUALITY
                and cache['code_sha256'] == code and cache['cache_sha256'] == digest,
                'Cache receipt mismatch')
        record(directory / 'CACHE.pt', digest)

    selected = {}
    for mode, seed in MODELS[:-1]:
        directory = RUN / 'training' / f'{mode}_{seed}'
        identity = load(directory / 'RUN.json')
        done = load(directory / 'DONE.json')
        rows = [r for r in registry['models'] if (r['mode'], r['seed']) == (mode, seed)]
        require(len(rows) == 1, 'Unique selected checkpoint required')
        row = rows[0]
        checkpoint = directory / 'best.pt'
        require(safe(row['path']) == safe(checkpoint) and identity['mode'] == mode
                and identity['seed'] == seed and identity['code_sha256'] == code
                and identity['protocol'] == protocol and identity['quality'] == QUALITY
                and identity['cache_sha256'] == registry['cache_sha256']
                and done['status'] == 'TRAINING_COMPLETE' and done['best_epoch'] == row['best_epoch']
                and done['best_sha256'] == row['sha256'], 'Checkpoint receipt mismatch')
        record(checkpoint, row['sha256'])
        selected[(mode, seed)] = identity

    saved = load(REVIEW / 'SAVED_EVIDENCE.json')
    normalization = load(REVIEW / 'NORMALIZATION_CHECK.json')
    require(saved['reference_quality'] == QUALITY, 'Saved evidence quality mismatch')
    for identity in saved['training_identities']:
        require(identity['loss_weights'] == protocol['loss_weights'], 'Loss weights changed')
        for path, digest in identity['verified_code'].items():
            require(code[path] == digest, 'Saved review code identity mismatch')
    for path in HERE.glob('*.py'):
        record(path)
    return files, selected, saved, normalization


def benchmark_modules():
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import common
    import data
    import objective
    require(Path(common.__file__).resolve().parent == BENCH
            and Path(data.__file__).resolve().parent == BENCH
            and Path(objective.__file__).resolve().parent == BENCH, 'Module origin mismatch')
    return common, data, objective


def cache(case):
    import torch
    require(case in CASES, 'Only the training and development caches are permitted')
    data = torch.load(safe(RUN / 'data' / str(case) / 'CACHE.pt'),
                      map_location='cpu', weights_only=True)
    require(data['omega'].shape == (81, 100, 100)
            and torch.allclose(data['time_s'], 1 + .05 * torch.arange(81, dtype=torch.float64),
                               rtol=0, atol=1e-10), 'Unexpected cache geometry or times')
    return data


def solver(model, budget, identity=None, *, n=100, dtype=None):
    import torch
    common, _, _ = benchmark_modules()
    mode, seed = model
    s = common.make_solver(mode if seed is not None else 'FULL-C2', seed or 2026, n=n)
    if identity is not None:
        require(model in MODELS[:-1], 'Only registered checkpoints are permitted')
        payload = torch.load(safe(RUN / 'training' / f'{mode}_{seed}' / 'best.pt'),
                             map_location='cpu', weights_only=True)
        require(payload['identity'] == identity and payload['backscatter_factor'] == 1.,
                'Checkpoint payload identity mismatch')
        s.load_state_dict(payload['model'], strict=True)
    if seed is None:
        def analytic(self, state):
            qx, qy = common.prior.c4.flux4(self, state, self.config.data.filter_width_m)
            return dict(qx=qx, qy=qy)
        s.closure_fields = MethodType(analytic, s)
    if dtype is not None:
        s.to(dtype=dtype)
    s.eval().requires_grad_(False)
    require(s.config.model.sgs_enabled and s.closure_output_scale == 1.
            and not s.config.physics.forcing_enabled, 'Unexpected solver configuration')
    if n == 100:
        k = torch.fft.fftfreq(n, d=1/n).abs() <= 33
        require(torch.equal(s.dealias, k[:, None] & k[None, :]), 'Expected rectangular K33')
    s.closure.register_forward_pre_hook(lambda *_: budget.tick('cnn'))
    original = s.rhs

    def counted_rhs(w, t):
        budget.tick('rhs')
        return original(w, t)

    s.rhs = counted_rhs
    return s
