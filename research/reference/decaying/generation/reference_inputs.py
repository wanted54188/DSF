"""Frozen paths, metadata-only audit, access control and atomic persistence."""
from pathlib import Path
import csv, hashlib, importlib.util, json, os, re, sys, time, zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUNDLE=ROOT/'A0B1 full/jcp_dissipation_20261024/decisive/reference_bundle_20261025'
PLATFORM=BUNDLE.parent/'reference_platform_v2_20261025'
MANIFEST=BUNDLE/'results/all_four_01/COMPLETE_REFERENCE_MANIFEST.json'
OLD=ROOT/'research/ssc_comprehensive_v1_20261029'
SEEDS=(110240101,110240102)
VARIANTS=('n1024_main','n1536_main','n1536_half')
METHODS=('FULL','RemoveD','Clark','SSC_0.5_24','SSC_1_16','MixAll','P1')
FIELDS=('reference_omega','reference_dot','reference_source','reference_qx','reference_qy','flux_source','reference_residual')
for p in (ROOT,PLATFORM):
    if str(p) not in sys.path:sys.path.insert(0,str(p))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError('Unsupported JSON value '+type(v).__name__)
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=convert),encoding='utf-8');os.replace(tmp,p)
def atomic_npz(p,**a):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:np.savez(f,**a)
    os.replace(tmp,p)
def csvout(p,rows):
    rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r));tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def seed_guard(seed):
    if seed not in SEEDS:raise PermissionError('Only development 110240101/110240102 are authorized')
def output_guard(p):
    p=Path(p).resolve()
    if not p.is_relative_to(HERE/'results'):raise PermissionError('Output must be under the new research results/')
    return p
def array_guard(p):
    p=Path(p).resolve()
    star_marker=re.search(r'(^|[^A-Z0-9])STAR(?:[-_ ]?CCM\+?)?([^A-Z0-9]|$)',str(p).upper())
    if any(str(s) in str(p) for s in (110240201,110240202,110240301,110240302)) or star_marker:
        raise PermissionError('Locked/reserved/STAR array access is forbidden')
    return p
def archive(p):return np.load(array_guard(p),allow_pickle=False)
def npz_schema(p):
    """Read only ZIP member headers, never decode a field array."""
    result={}
    with zipfile.ZipFile(array_guard(p)) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):continue
            with z.open(name) as f:
                version=np.lib.format.read_magic(f)
                shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
            result[name[:-4]]=dict(shape=list(shape),dtype=str(dtype),fortran_order=fortran)
    return result
def engine():
    name='ssc_long_private_frozen_engine'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,OLD/'engine.py');m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]
def op(n,length):return engine().diag.Fourier(n,n,length,length,2/3)
def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))
def relative(a,b):return rms(np.asarray(a)-b)/max(rms(b),1e-30)
def index(times,t):
    i=int(np.argmin(abs(np.asarray(times)-t)))
    if abs(times[i]-t)>1e-9:raise ValueError(f'No exact reference time {t}; interpolation forbidden')
    return i
def policy():
    old=read(BUNDLE/'protocol.json')
    return dict(version='SSC_long_horizon_v1_20261030',seeds=list(SEEDS),roles='development_only',
        window=[1.,5.],variants=list(VARIANTS),production='n1536_half',quality=old['quality'],
        native_raw_relative_limit=.01,overlap_relative_limit=old['replay_endpoint_relative_limit'],
        native_record_dt=.05,checkpoint_dt=.05,methods=list(METHODS),weights=[2026,2028],les_dt=[.005,.0025],
        prediction_error_thresholds=[.05,.10,.20],source_record_dt=.05,field_plot_times=[1.2,2.,2.5,5.],
        safety_E_Z_ratio=100.,reference_resource_seconds_per_invocation=86400.,
        verdict_rule='SSC optimized must complete all conditions and beat BOTH Clark and FULL in full-window and terminal omega for every paired condition at both dt; require margin beyond observed 2x reference differences plus dt score drift; E/Z tradeoffs separate; never claim independent validation/JCP novelty',
        parameter_selection='NONE: SSC(.5,24) and SSC(1,16) frozen before long-reference access',
        quality_scope='all dense low frames + sparse native full-mode differences, candidate tail/CFL, separate flux/RHS identities',
        fallback='No automatic mesh/viscosity/threshold change. Failed quality blocks LES verdict; preserve failures.',
        forcing='none; original decaying NS',DNS_claim=False)
def project_module_sources(modules=None):
    """Real imported project files only; torch.ops/classes expose virtual filenames."""
    paths=set()
    for module in list(sys.modules.values()) if modules is None else modules:
        # Read the module dictionary without invoking a lazy module __getattr__.
        raw=vars(module).get('__file__') if hasattr(module,'__dict__') else None
        if not isinstance(raw,(str,os.PathLike)):continue
        path=Path(raw)
        # Python source imports have absolute filenames. Relative names such as
        # torch.ops.__file__ == '_ops.py' are virtual, not relative to our cwd.
        if not path.is_absolute():continue
        path=path.resolve()
        if path.suffix=='.py' and path.is_relative_to(ROOT):
            if not path.is_file():raise FileNotFoundError('Imported project source is missing: '+str(path))
            paths.add(path)
    return paths

def dependencies():
    paths=[*HERE.glob('*.py'),*OLD.glob('*.py'),PLATFORM/'rp_spectral.py',PLATFORM/'rp_common.py',
        ROOT/'research/skew_full_validation_20261020/core.py',ROOT/'research/skew_full_validation_20261020/reference.py',
        ROOT/'spnsgs/solver.py',ROOT/'spnsgs/config.py',ROOT/'spnsgs/filters.py',ROOT/'spnsgs/paper.py',
        ROOT/'A0B1 full/operators.py',BUNDLE/'protocol.json',MANIFEST]
    # Also protect the imported frozen harness implementation, not its restricted data.
    paths+=list((ROOT/'research/full_fourier_source_v1_20261028').glob('*.py'))
    paths+=list((ROOT/'research/full_low_high_control_v1_20261029').glob('*.py'))
    for r in read(OLD/'results/campaign03/RUN.json')['models']:paths.extend([Path(r['checkpoint']),Path(r['config'])])
    engine()  # imports code only; no arrays or neural checkpoint deserialization
    paths.extend(project_module_sources())
    return {str(p):sha(p) for p in paths}
def verify_frozen(run):
    f=read(run/'FREEZE.json')
    if f['policy']!=policy():raise ValueError('Frozen protocol changed; use a new output revision')
    for p,h in f['dependencies'].items():
        if sha(p)!=h:raise ValueError('Frozen dependency changed: '+p)
    return f
def audit():
    manifest=read(MANIFEST);result=dict(status='PASS_METADATA_ONLY',manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),
        cases=[],locked_arrays_accessed=False,array_decoding=False)
    for seed in SEEDS:
        case=next(c for c in manifest['cases'] if c['seed']==seed);seed_guard(seed)
        if case['role']!='development' or case['status']!='PASS_REFERENCE_QUALITY':raise ValueError('Unqualified case')
        ready_schema=npz_schema(case['ready_path']);rows=[]
        if ready_schema['reference_omega']['shape']!=[161,100,100] or ready_schema['time_s']['shape']!=[161]:raise ValueError('Official READY schema mismatch')
        for v in VARIANTS:
            key={'n1024_main':'spatial_control','n1536_main':'candidate_main','n1536_half':'candidate_half'}[v]
            d=array_guard(case['candidate_paths'][key]);f=read(d/'FREEZE.json');c=f['config'];md=read(d/'REFERENCE_METADATA.json')
            if c['seed']!=seed or c['role']!='development' or c['initial']['case']!='decaying':raise ValueError('Wrong physical problem')
            for k in ('nu','delta','length','record_dt','les_n','initial'):
                if rows and c[k]!=rows[0]['config'][k]:raise ValueError('Variant physical mismatch: '+k)
            cp=read(d/'CHECKPOINT.json') if (d/'CHECKPOINT.json').exists() else None
            if cp and (cp['freeze_sha256']!=sha(d/'FREEZE.json') or sha(d/cp['file'])!=cp['sha256'] or abs(cp['step']*f['dt']-1.2)>1e-10):
                raise ValueError('Original checkpoint hash/time mismatch')
            schema=npz_schema(d/cp['file']) if cp else None
            if schema and schema['h']!={'shape':[f['n'],f['n']],'dtype':'complex128','fortran_order':False}:raise ValueError('Original checkpoint schema mismatch')
            rows.append(dict(variant=v,path=str(d),n=f['n'],dt=f['dt'],checkpoint_schema=schema,config={k:c[k] for k in ('nu','delta','length','les_n','record_dt','initial','window')},
                checkpoint=cp and {k:cp[k] for k in ('step','file','sha256')},checkpoint_path=str(d/cp['file']) if cp else None,
                freeze_sha256=sha(d/'FREEZE.json'),done_sha256=sha(d/'DONE.json'),fields_sha256=read(d/'DONE.json')['outputs_sha256']['FIELDS.npz'],
                historical_seconds=md['seconds'],historical_duration=1.2-md['balance_start_time'],
                original_origin=md['origin'],old_quality=read(Path(case['ready_path']).parent/'QUALITY.json')['gates']))
        if result['cases']:
            for k in ('nu','delta','length','les_n','initial','record_dt'):
                if rows[0]['config'][k]!=result['cases'][0]['variants'][0]['config'][k]:raise ValueError('Cross-case physics mismatch')
        result['cases'].append(dict(seed=seed,role='development',ready_path=case['ready_path'],ready_sha256=case['ready_sha256'],ready_schema=ready_schema,variants=rows))
    physical=result['cases'][0]['variants'][0]['config']
    for model in read(OLD/'results/campaign03/RUN.json')['models']:
        m=read(model['config']);nu=m['physics']['viscosity_value'];dom=m['domain']
        if m['physics']['viscosity_kind']=='dynamic':nu/=m['physics']['density_kg_m3']
        elif m['physics']['viscosity_kind']!='kinematic':raise ValueError('Unknown original viscosity units')
        if m['physics']['forcing_enabled'] or m['data']['filter_kind']!='gaussian' or not np.isclose(nu,physical['nu'],atol=1e-18,rtol=0):raise ValueError('Original FULL/reference physics mismatch')
        if m['data']['filter_width_m']!=physical['delta'] or dom['x_max_m']-dom['x_min_m']!=physical['length'] or dom['y_max_m']-dom['y_min_m']!=physical['length']:raise ValueError('Original FULL/reference domain/filter mismatch')
    return result
def estimate(a):
    nf=round(4/a['cases'][0]['variants'][0]['config']['record_dt'])+1;rows=[]
    for case in a['cases']:
        for v in case['variants']:
            n=v['n'];cold=not bool(v['checkpoint']);duration=5. if cold else 3.8
            seconds=v['historical_seconds']/v['historical_duration']*duration
            low=len(FIELDS)*nf*100**2*8;native=(round(4/.05)+1)*n*n*8;cp=2*n*n*16
            rows.append(dict(seed=case['seed'],variant=v['variant'],steps=round(duration/v['dt']),
                estimated_GB=(low+native+cp)/1e9,historical_device_extrapolation_seconds=seconds,
                working_array_budget_GB=n*n*(32*16+32*8)/1e9,
                memory_note='Rough buffer budget, not a certified peak; allow 2x for FFT workspace/context; tasks sequential',
                interpretation='Historical device/load extrapolation, NOT a CPU or new GPU benchmark; allow 3x margin'))
    return dict(tasks=rows,total_estimated_GB=sum(x['estimated_GB'] for x in rows),
        minimum_free_GB=1.3*sum(x['estimated_GB'] for x in rows)+2,
        note='Sequential reference tasks; mmap/chunk storage. CPU/GPU wall cost must be measured on the chosen device. No full-field decoding in estimate.')
def log(run,message):
    line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+message;print(line,flush=True)
    with (run/'EXECUTION.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
