"""Paper campaign. No expensive work without an explicit execution flag.

Original modules and artifacts are read only. New runs use the original
transactional driver, guards, raw-logit decoder and physical history clock.
"""
import sys,json,csv,hashlib,argparse,copy,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'research/full_aug_parameter_pc_full_v1_20261116'
sys.path[:0]=[str(ROOT),str(OLD)]
NAMES={'Aug':'DSF-Aug','Fresh-C4':'DSF-Sym','Parameter-PC':'DSF-Trend','Parameter-Hold':'DSF-Hold','Clark4':'Clark4','Parameter-Hold-Single':'DSF-SingleHold'}
QUALITY='REFERENCE_NOT_CONVERGED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def table(p,rows,columns=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,columns or list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def audit():
    """Independent saved-state FFT scoring, zero CNN and zero integration."""
    (HERE/'RESULTS/FIGURES').mkdir(parents=True,exist_ok=True)
    import pc_common as c
    import pc_metrics as m
    run=OLD/'results/run01';rec=read(run/'RUN.json');report=run/read(run/'LATEST_REPORT.json')['path'];manifest=read(report/'MANIFEST.json')
    assert sha(report/'MANIFEST.json')==read(run/'LATEST_REPORT.json')['manifest_sha256']
    inputs={};evidence=[]
    def check(p,expected=None):
        p=Path(p);h=sha(p);assert expected is None or h==expected,(str(p),'SHA mismatch');inputs[str(p.relative_to(ROOT))]=h;return p
    for k,h in rec['code_identity'].items():check(OLD/k,h)
    freeze=ROOT/'research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json'
    f=read(check(freeze));models=[r for r in f['checkpoints'] if r['mode']=='FULL-C4-Augmented']
    for r in models:check(ROOT/r['file'],r['sha256'])
    write(HERE/'AUDIT/FROZEN_MODELS.json',models)
    op=c.load(check(run/'OPERATORS.npz'));masks,_=m.geometry(op)
    records=[]
    for case in rec['protocol']['cases']:
        ref=c.References(rec['inventory']['references'][str(case)])
        desc=ref.desc
        for p in (desc['fields'].values() if desc['kind']=='dense' else desc['frames'].values()):check(ROOT/p,rec['inventory']['input_sha256'][p])
        truth={n:m.project(ref.at(n)['reference_omega'],op) for n in range(0,801,10)}
        for seed in (2026,2028):
            for method in tuple(NAMES)[:4]:
                name=f'STATES_{case}_{seed}_{method}.npy';a=np.load(check(report/name,manifest['outputs'][name]),mmap_mode='r');assert a.shape==(801,100,100)
                for n,r in truth.items():
                    st=m.fields(a[n],op);rt=m.fields(r,op);power=abs(m.F(a[n]))**2;rp=abs(m.F(r))**2;err=abs(m.F(a[n]-r))**2
                    for band,mask in masks.items():
                        mse=float(err[mask].sum());records.append(dict(case=case,seed=seed,method=NAMES[method],step=n,time=1+n*.005,band=band,omega_MSE=mse,omega_relative_L2=float(np.sqrt(mse/max(rp[mask].sum(),1e-300))),u_MSE=float(np.mean((st['u']-rt['u'])**2)),v_MSE=float(np.mean((st['v']-rt['v'])**2)),K=float(.5*(power*op['inverse'])[mask].sum()),Z=float(.5*power[mask].sum()),reference_Z=float(.5*rp[mask].sum())))
    table(HERE/'RESULTS/STATE_METRICS.csv',records)
    grouped={}
    for r in records:grouped.setdefault((r['case'],r['seed'],r['method'],r['band']),[]).append(r)
    sums=[]
    for key,rr in grouped.items():
        for window in ('full','early','late','t5'):
            chosen=[r for r in rr if {'full':r['step']>0,'early':40<=r['step']<=240,'late':r['step']>240,'t5':r['step']==800}[window]]
            sums.append(dict(zip(('case','seed','method','band'),key),window=window,samples=len(chosen),**{k:float(np.mean([r[k] for r in chosen])) for k in ('omega_MSE','u_MSE','v_MSE','omega_relative_L2','K','Z','reference_Z')}))
    table(HERE/'RESULTS/WINDOW_SUMMARY.csv',sums)
    idx={(r['case'],r['seed'],r['method'],r['band'],r['window']):r for r in sums};paired=[]
    for key,r in idx.items():
        for base in ('DSF-Aug','DSF-Sym','DSF-Trend','DSF-Hold'):
            if base==r['method']:continue
            b=idx[key[0],key[1],base,key[3],key[4]]
            paired.append(dict(case=key[0],seed=key[1],candidate=key[2],baseline=base,band=key[3],window=key[4],candidate_MSE=r['omega_MSE'],baseline_MSE=b['omega_MSE'],improvement_pct=100*(1-r['omega_MSE']/b['omega_MSE']),reference_quality=QUALITY))
    table(HERE/'RESULTS/PAIRED_COMPARISONS.csv',paired)
    for src,dst in [('COSTS.csv','COST_ANALYSIS.csv'),('SPECTRA.csv','SPECTRAL_RESULTS.csv'),('FLUX_BUDGETS.csv','PHYSICAL_BUDGETS.csv'),('REFERENCE_SOURCE_REUSED.csv','APRIORI_EXISTING.csv')]:
        p=report/src
        if not p.exists():continue
        check(p,manifest['outputs'][src]);rr=rows(p)
        for r in rr:
            if 'method' in r:r['method']=NAMES.get(r['method'],r['method'])
        table(HERE/'RESULTS'/dst,rr)
    source=ROOT/'research/full_aug_hold_audit_v1_20261117/results/audit01'
    for p in source.glob('*.csv'):
        check(p);evidence.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),status='EXISTING_VERIFIED',scope='saved-state analysis; finite samples only'))
    for parent in ('research','docs','spnsgs','configs/jcp'):
        for p in (ROOT/parent).glob('*'):
            if p==HERE:continue
            evidence.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p) if p.is_file() else '',status='INDEXED',scope='directory identity only' if p.is_dir() else 'file'))
    table(HERE/'AUDIT/EVIDENCE_MATRIX.csv',evidence);write(HERE/'AUDIT/INPUT_SHA256.json',inputs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));labels=[]
    for i,(case,seed) in enumerate(( (c,s) for c in (110240101,110240102,110240104) for s in (2026,2028))):
        vals=[idx[case,seed,name,'all','full']['omega_MSE'] for name in ('DSF-Aug','DSF-Sym','DSF-Trend','DSF-Hold')]
        ax.bar(np.arange(4)+i*5,vals);labels.extend([f'{case%1000}/{seed}\n'+n for n in ('Aug','Sym','Trend','Hold')])
    ax.set_xticks([i*5+j for i in range(6) for j in range(4)],labels,rotation=90,fontsize=7);ax.set_ylabel('Full-window absolute vorticity MSE');ax.set_title(QUALITY);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(HERE/f'RESULTS/FIGURES/development_MSE.{ext}')
    plt.close(fig)
    write(HERE/'AUDIT/AUDIT_RECEIPT.json',dict(status='PASS',trajectories=24,uniform_frames=1944,CNN=0,new_RHS=0,reference_quality=QUALITY))
    print('PASS: 24 saved trajectories independently rescored; 0 CNN, 0 integration')

def bind(period=5,fine=False,single=False):
    import torch,pc_common as c,pc_engine as e,driver as d
    import pc_storage as storage
    c.HERE=HERE;c.P=copy.deepcopy(c.P);c.P['hard_budget']={'CNN':1000000,'RHS':1000000}
    class Bank(e.Bank):
        def observe(self,n,anchor):
            assert n%period in (0,1)
            if n%period==0:self.pending=anchor
            else:
                assert self.pending and self.pending['step']==n-1
                self.active=[self.pending,anchor];self.pending=None
        def predict(self,n,fraction,method):return super().predict(n,fraction,'Parameter-Hold' if 'Hold' in method else method)
    class Engine(e.Engine):
        def orbit(self,w):
            if not single:return super().orbit(w)
            f=self.original(self.s.state(w));z=e.vec(f,'qd');b=e.vec(f,'qb')
            return dict(original=f,heads=self.captured_raw[None].repeat(4,1,1,1),d=z,b=b)
        def prepare(self,n,w,bank):
            # Base prepare has hard-coded five-step schedule; all other logic stays native.
            self.n=n;self.stage=0;self.bank=bank;self.records=[];self.values=[];self.snapshots=[];self.cache=None;self.fresh=None
            if self.method.startswith('Parameter') and n%period in (0,1) and not(n==1 and bank.active[-1]['step']==1):
                self.j.phase='refresh';fresh=self.orbit(w);anchor=e.save_anchor(self.root,self.folder/'anchors',n,fresh['heads']);bank.observe(n,anchor);self.cache=fresh['original'];self.fresh=fresh
        def flux(self,st):
            if not(single and self.method.startswith('Parameter')):return super().flux(st)
            heads,tau=self.bank.predict(self.n,self.fractions[self.stage],'Parameter-Hold')
            original=self.original;method=self.method
            def held(ss):
                self.captured_raw=heads[0];return self.decode(ss,heads[0])
            try:
                self.original=held;self.method='Aug';out=super().flux(st);self.last_check['tau']=tau;return out
            finally:self.original=original;self.method=method
        def advance(self,n,w,bank):
            if not fine:return super().advance(n,w,bank)
            c.sync(w.device);begin=time.perf_counter();self.prepare(n,w,bank)
            frozen=bank.descriptor() if bank else None
            for half,fracs in enumerate(((0.,.25,.25,.5),(.5,.75,.75,1.))):
                self.stage=0;self.fractions=fracs;w=self.s.step(w,1+c.H*(n+half*.5),c.H/2)
                assert bool(torch.isfinite(w).all())
            assert not bank or bank.descriptor()==frozen
            self.fractions=(0.,.5,.5,1.)
            return w,dict(solver_wall_seconds=time.perf_counter()-begin)
    d.Engine=Engine;d.Bank=Bank
    def load_bank(root,desc,device):return Bank([e.load_anchor(root,a,device) for a in desc['active']],e.load_anchor(root,desc['pending'],device) if desc['pending'] else None)
    d.load_bank=load_bank
    if fine:
        def scan(folder):
            rr=[]
            for expected,p in enumerate(sorted((folder/'steps').glob('*.npz')),2):
                z=c.load(p);v=c.meta(z);assert v['accepted_step']==expected and z['rhs'].shape==(8,100,100);rr.append((p,v))
            return rr
        storage.scan=scan
    return c,e,d,storage,Engine,Bank

def les(args):
    if not args.execute:print('NOT_RUN: add --execute to manually run the registered campaign');return
    import torch
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c,e,d,storage,Engine,Bank=bind(args.period,args.fine,args.single)
    assert args.period in (4,5,8),'Frozen sensitivity candidates only'
    root=HERE/'EXPERIMENTS/RUNS'/args.run;assert Path(args.run).name==args.run
    inventory=c.inventory();cases=[110240101,110240102,110240104]
    if args.ic105:
        cases=[110240105];folder=HERE/'EXPERIMENTS/RUNS/dns_110240105_1536_0.000125'
        done=read(folder/'DONE.json');assert done['status']=='OPERATOR_CHECKS_PASS'
        inventory['references']['110240105']=dict(kind='frames',frames={str(n):str((folder/f'frame_{n:04d}.npz').relative_to(ROOT)) for n in range(0,801,10)},steps=list(range(0,801,10)))
    identity=dict(config=vars(args),checkpoints=inventory['checkpoints'],code={p.name:sha(p) for p in HERE.glob('*.py')},reference_quality=QUALITY)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'RUN.json').exists():assert read(root/'RUN.json')==identity,'Resume identity changed'
    else:c.dump(root/'RUN.json',identity)
    with storage.RunLock(root),torch.inference_mode():
        j=c.Journal(root/'invocations')
        try:
            for case in cases:
                ref=c.References(inventory['references'][str(case)])
                for seed in (2026,2028):
                    row=next(r for r in inventory['checkpoints'] if r['seed']==seed)
                    start,_=d.startup(root,case,seed,row,ref,j,'cuda')
                    methods=('Parameter-Hold-Single',) if args.single else tuple(NAMES)[:4]
                    for method in methods:d.run_trajectory(root,case,seed,method,row,start,j,'cuda')
            c.dump(root/'EXECUTION.json',dict(actual_calls=j.used,status='TERMINAL',reference_quality=QUALITY))
        except BaseException as ex:
            c.dump(root/f'INTERRUPTION_{time.time_ns()}.json',dict(error=str(ex),traceback=traceback.format_exc()));raise
        finally:j.close()

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True);sub.add_parser('audit')
    p=sub.add_parser('les');p.add_argument('--execute',action='store_true');p.add_argument('--run',default='main01');p.add_argument('--single',action='store_true');p.add_argument('--fine',action='store_true');p.add_argument('--ic105',action='store_true');p.add_argument('--period',type=int,default=5)
    a=ap.parse_args()
    if a.cmd=='audit':audit()
    else:les(a)
if __name__=='__main__':main()
