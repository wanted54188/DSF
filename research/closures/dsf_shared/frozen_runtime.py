"""Frozen inputs, production factory, lossless atomic IO and explicit budgets."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid
spec_io = None

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('original_aug_readonly', ROOT/'research/full_aug_rotation_pc_pilot_v1_20261114/experiment.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
spec = importlib.util.spec_from_file_location('original_flux_certificate', ROOT/'research/full_aug_history_flux_feasibility_v1_20261114/feasibility.py')
feasible = importlib.util.module_from_spec(spec); spec.loader.exec_module(feasible)
spec = importlib.util.spec_from_file_location('parameter_original_decoder',ROOT/'research/full_aug_orbit_rk4_one_step_v1_20261114/common_step.py')
decoder = importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
P = json.loads((HERE/'PROTOCOL.json').read_text(encoding='utf-8'))
QUALITY = P['reference_quality']; METHODS = tuple(P['methods']); H = .005; END = 800
require = q.require

# Same bounded Windows atomic replace retry as the independently tested repair.
spec_io=importlib.util.spec_from_file_location('parameter_atomic_io',ROOT/'research/full_aug_history_pc_windows_io_fix_v1_20261115/atomic_io.py')
atomic_io=importlib.util.module_from_spec(spec_io);spec_io.loader.exec_module(atomic_io)
atomic_replace=atomic_io.RetryReplace()


def sha(path): return q.sha(path)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.integer,np.bool_)): return value.item()
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    return value


def safe(path):
    path = Path(path).resolve(); require(path.is_relative_to(HERE), 'Output escaped new experiment')
    return path


def dump(path, value):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    with temp.open('w', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    atomic_replace(temp, path)


def npz(path, **arrays):
    path = safe(path); require(not path.exists(), 'Immutable file exists: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    atomic_replace(temp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def meta(z): return json.loads(str(z['metadata'].item()))
def packed(value): return np.array(json.dumps(clean(value), ensure_ascii=False, allow_nan=False))
def csvout(path, rows, columns=None):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, columns or list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def local_identity():
    return {p.name:sha(p) for p in sorted([*HERE.glob('*.py'), HERE/'PROTOCOL.json'])}
def key(case, seed, method): return f'{case}_{seed}_{method}'


def model(row, device='cpu'):
    payload = torch.load(ROOT/row['file'], map_location='cpu', weights_only=True)
    require(payload['identity'] == read(ROOT/row['run']) and payload['epoch'] == row['epoch']
            and payload['backscatter_factor'] == 1., 'Checkpoint payload identity')
    s = q.study.make_solver(q.study.AUGMENTED, row['seed']).double().to(device).eval().requires_grad_(False)
    s.load_state_dict(payload['model'], strict=True)
    return s


def inventory():
    """Only metadata/hashes/available saved low-grid frames; no CNN or DNS."""
    files = {}; missing = {}; refs = {}; rows = []
    def verify(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, 'Input SHA changed: '+str(path))
        files[str(path.relative_to(ROOT))] = digest
        return path
    frozen_path = ROOT/'research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json'
    freeze = read(verify(frozen_path))
    require(freeze['status'] == 'FROZEN' and freeze['quality'] == QUALITY, 'Frozen status')
    for section in ('training_code_sha256','baseline_code_sha256'):
        for name,digest in freeze[section].items(): verify(ROOT/name,digest)
    verify(q.study.common.CONFIG,freeze['baseline_config_sha256'])
    for helper in (Path(q.__file__), Path(feasible.__file__),Path(decoder.__file__),Path(atomic_io.__file__)): verify(helper)
    audit_decoder=read(ROOT/'research/full_aug_orbit_rk4_one_step_v1_20261114/PHASE_ONE_MANIFEST.json')
    require(sha(Path(decoder.__file__))==audit_decoder['current_code_sha256']['common_step.py'],'Original decoder helper changed')
    for row in freeze['checkpoints']:
        if row['mode'] != q.study.AUGMENTED: continue
        for k,h in (('file','sha256'),('run','run_sha256'),('done','done_sha256')): verify(ROOT/row[k],row[h])
        r,d = read(ROOT/row['run']),read(ROOT/row['done'])
        require(r['cache_sha256'] == freeze['cache_sha256'] and d['status'] == 'TRAINING_COMPLETE'
                and d['best_sha256'] == row['sha256'], 'Not original 101-only Aug')
        rows.append(row)
    require({r['seed'] for r in rows} == {2026,2028}, 'Frozen seeds differ')
    audit_path = q.study.inputs.RUN/'AUDIT.json'; audit = read(verify(audit_path))
    for case in P['cases']:
        try:
            if case in (110240101,110240102):
                r, = [r for r in audit['references'] if r['seed'] == case and r['variant'] == 'n1536_half']
                tp = verify(r['time_path'],r['time_sha256']); ts = np.load(tp,allow_pickle=False)
                fields = {}
                for name in ('reference_omega','reference_dot','reference_source'):
                    item = r['fields'][name]; path = verify(item['path'],item['expected_sha256'])
                    a = np.load(path,mmap_mode='r',allow_pickle=False)
                    require(a.shape == (3201,100,100) and a.dtype == np.float64,'Reference field shape')
                    fields[name] = str(path.relative_to(ROOT))
                require(np.max(abs(ts-(1+.00125*np.arange(3201))))<1e-10,'Reference timestamps')
                refs[str(case)] = dict(kind='dense',fields=fields,time_path=str(tp.relative_to(ROOT)),
                    steps=list(range(801)),convergence=QUALITY)
            else:
                require(case == 110240104,'Unauthorized case')
                folder = ROOT/'research/full_c4_multi_ic_v1_20261110/results/paired01/reference/110240104'
                rec = read(verify(folder/'DONE.json'))
                require(rec['case'] == case and rec['quality'] == QUALITY and rec['status'] == 'OPERATOR_CHECKS_PASS'
                        and rec['grid'] == 1536 and rec['dt'] == .000125,'104 reference identity')
                frames = {}
                for n in range(801):
                    step = round((1+H*n)/rec['dt']); name = f'frame_{step:05d}.npz'
                    if name not in rec['frames']: continue
                    path = verify(folder/name,rec['frames'][name])
                    with np.load(path,allow_pickle=False) as z:
                        require(abs(float(z['time_s'])-(1+H*n))<1e-10,'104 timestamp')
                        require(all(z[k].shape == (100,100) and z[k].dtype == np.float64
                                    for k in ('reference_omega','reference_dot','reference_source')),'104 field shape')
                    frames[str(n)] = str(path.relative_to(ROOT))
                require(all(str(n) in frames for n in range(0,801,10)), '104 missing primary comparison frame')
                refs[str(case)] = dict(kind='frames',frames=frames,steps=sorted(map(int,frames)),
                    convergence=QUALITY, sensitivity='NOT_RUN_SINGLE_DISCRETIZATION',
                    max_native_tail_fraction=rec['max_native_tail_fraction'])
        except FileNotFoundError as error:
            missing[str(case)] = str(error)
    source_reuse=[]
    for case,phase in ((110240102,'development'),(110240104,'test')):
        folder=ROOT/'research/full_c4_multi_ic_v1_20261110/results/paired01'
        if not (folder/phase/'DONE.json').exists(): continue
        rec=read(verify(folder/phase/'DONE.json')); multi=read(verify(folder/'FROZEN.json',rec['freeze_sha256']))
        old_aug={r['seed']:r['sha256'] for r in multi['checkpoints'] if r['mode']==q.study.AUGMENTED}
        require(old_aug=={r['seed']:r['sha256'] for r in rows},'Reference Source reuse has different Aug weights')
        path=verify(folder/phase/'SOURCE.csv',rec['outputs']['SOURCE.csv'])
        source_reuse.append(dict(case=case,path=str(path.relative_to(ROOT)),kind='multi_reference_scores'))
    folder=ROOT/'research/full_c4_augmented_orbit_v1_20261110/results/run01'
    if (folder/'MANIFEST.json').exists():
        rec=read(verify(folder/'MANIFEST.json'))
        freeze_key=str(frozen_path.relative_to(ROOT))
        require(rec['input_sha256'][freeze_key]==sha(frozen_path),'101 Source reuse freeze changed')
        path=verify(folder/'SOURCE.csv',rec['outputs']['SOURCE.csv'])
        source_reuse.append(dict(case=110240101,path=str(path.relative_to(ROOT)),kind='orbit_reference_scores'))
    return dict(checkpoints=rows, references=refs, missing=missing, input_sha256=files,source_reuse=source_reuse,
                effective_config_sha256=freeze['baseline_config_sha256'], reference_quality=QUALITY)


class References:
    def __init__(self, description):
        self.desc = description; self.arrays = {}
        if description['kind'] == 'dense':
            self.arrays = {k:np.load(ROOT/v,mmap_mode='r',allow_pickle=False) for k,v in description['fields'].items()}
    def at(self, n):
        if n not in self.desc['steps']: return None
        if self.desc['kind'] == 'dense': return {k:np.array(v[4*n]) for k,v in self.arrays.items()}
        z = load(ROOT/self.desc['frames'][str(n)])
        return {k:z[k] for k in ('reference_omega','reference_dot','reference_source')}


class Journal:
    """Count EVERY invocation before it runs, including discarded interrupted work."""
    def __init__(self, folder):
        self.folder = safe(folder); self.folder.mkdir(parents=True,exist_ok=True)
        self.used = dict(CNN=0,RHS=0); self.per_key = {}
        self.incomplete_lines = 0
        for p in sorted(self.folder.glob('*.jsonl')):
            for line in p.read_text(encoding='utf-8').splitlines():
                try: row = json.loads(line)
                except json.JSONDecodeError:
                    self.incomplete_lines += 1; continue
                self._account(row)
        require(self.incomplete_lines == 0,'Invocation log has a torn line; exact cost cannot be claimed')
        self.path = self.folder/(str(time.time_ns())+'_'+uuid.uuid4().hex+'.jsonl')
        self.f = self.path.open('x',encoding='utf-8',buffering=1)
        self.scope = 'none'; self.phase = 'none'
    def _account(self,row):
        if row['kind'] not in self.used: return
        self.used[row['kind']] += 1
        self.per_key.setdefault(row['scope'],dict(CNN=0,RHS=0))[row['kind']] += 1
    def tick(self,kind):
        require(self.used[kind] < P['hard_budget'][kind], 'HARD_BUDGET_EXHAUSTED '+kind)
        row = dict(kind=kind,scope=self.scope,phase=self.phase,time_ns=time.time_ns())
        self.f.write(json.dumps(row)+'\n'); self.f.flush()
        os.fsync(self.f.fileno())
        self._account(row)
    def note(self,kind,**data):
        self.f.write(json.dumps(clean(dict(kind=kind,scope=self.scope,time_ns=time.time_ns(),**data)))+'\n')
        self.f.flush()
    def close(self):
        self.f.flush(); os.fsync(self.f.fileno()); self.f.close()


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()
def resource_memory(device):
    rss=None
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counter=Counters(); counter.cb=ctypes.sizeof(counter)
        handle_fn=ctypes.windll.kernel32.GetCurrentProcess; handle_fn.restype=wintypes.HANDLE
        fn=ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        require(bool(fn(handle_fn(),ctypes.byref(counter),counter.cb)),'Windows RSS query failed')
        rss=int(counter.WorkingSetSize)
    return dict(process_rss_bytes=rss,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0)


def operators(s):
    return dict(kx=s.kx.cpu().numpy(),ky=s.ky.cpu().numpy(),inverse=s.inverse_k_squared.cpu().numpy(),
                k_squared=s.k_squared.cpu().numpy(),dealias=s.dealias.cpu().numpy(),
                nu=np.array(s.config.physics.kinematic_viscosity_m2_s),
                filter_width=np.array(s.closure.filter_width),coefficient_max=np.array(s.closure.dissipation_coefficient_soft_limit),
                beta=np.array(s.closure.backscatter_factor*s.closure.maximum_backscatter_ratio),
                transfer_max=np.array(s.closure.maximum_local_backscatter_transfer_ratio))
