"""Same-state Clark4 flux vs PRL NGM4 stress. No NN, RHS, integration or edits.

The production scalar/vector FFT routines and original Clark4 code are called
directly through a stateless adapter. All controls use separate buffer objects.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from spnsgs.solver import SpectralVorticitySolver,SPNSGSClosure
C4PATH=ROOT/'research/ssc_clark4_dissipation_v1_20261102/clark4.py'
spec=importlib.util.spec_from_file_location('_readonly_equivalence_clark4',C4PATH)
c4=importlib.util.module_from_spec(spec);spec.loader.exec_module(c4)
RUN=ROOT/'research/full_aug_parameter_pc_full_v1_20261116/results/run01'
CLARK=ROOT/'research/full_aug_hold_audit_v1_20261117/paper_runs/dt_half01/limited_completion/Clark4_0.005'
EPSILON=1e-12
# Acceptance limits are inherited, not selected after inspecting these states.
NATIVE_LIMIT=2e-6
IDEAL_LIMIT=2e-11
PAD_LIMIT=2e-11
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def norm(x):return float(torch.linalg.vector_norm(x))
def rms(x):return float(torch.sqrt(torch.mean(x*x)))
def rel(a,b):return norm(a-b)/max(norm(b),EPSILON)
def array(x):return x.detach().cpu().numpy()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

class Operators:
 """Use identical production state/project/derivative functions, without a model."""
 project=SpectralVorticitySolver.project
 derivative=SpectralVorticitySolver.derivative
 state=SpectralVorticitySolver.state
 def __init__(self,buffers,kind='native',length=2*np.pi):
  for k in ('kx','ky','k_squared','dealias'):
   setattr(self,k,torch.from_numpy(np.array(buffers[k],copy=True)))
  self.inverse_k_squared=torch.from_numpy(np.array(buffers['inverse'],copy=True))
  self.ny=self.nx=100
  if kind in ('exact_k_native_inverse','ideal'):
   modes=torch.fft.fftfreq(100,d=1/100,dtype=torch.float64)
   self.ky,self.kx=torch.meshgrid(modes*2*np.pi/length,modes*2*np.pi/length,indexing='ij')
   self.k_squared=self.kx.square()+self.ky.square()
  if kind in ('native_k_exact_inverse','ideal'):
   self.k_squared=self.kx.square()+self.ky.square()
   self.inverse_k_squared=torch.zeros_like(self.k_squared)
   positive=self.k_squared>0;self.inverse_k_squared[positive]=1/self.k_squared[positive]

def derivatives(s,x):
 d=s.derivative
 return (d(x,'x'),d(x,'y'),d(d(x,'x'),'x'),d(d(x,'x'),'y'),d(d(x,'y'),'y'))

def padded_product(s,a,b):
 """150^2 independent exact product of retained P33 modes, cropped to P33."""
 n=100;nn=150;modes=np.rint(np.fft.fftfreq(n)*n).astype(int);index=np.ix_(modes%nn,modes%nn)
 aa=np.zeros((nn,nn),dtype=np.complex128);bb=np.zeros_like(aa)
 aa[index]=np.fft.fft2(array(a),norm='forward')*array(s.dealias)
 bb[index]=np.fft.fft2(array(b),norm='forward')*array(s.dealias)
 prod=np.fft.ifft2(aa,norm='forward').real*np.fft.ifft2(bb,norm='forward').real
 hh=np.fft.fft2(prod,norm='forward')[index]*array(s.dealias);hh[0,0]=0
 return torch.from_numpy(np.fft.ifft2(hh,norm='forward').real)

def stress_source(s,stress):
 xx,xy,yy=stress;d=s.derivative
 fx=-(d(xx,'x')+d(xy,'y'))
 fy=-(d(xy,'x')+d(yy,'y'))
 return s.project(d(fy,'x')-d(fx,'y'))

def evaluate(s,w,delta,pad=False):
 st=s.state(w);d=s.derivative
 weights=(delta**2/12,delta**2/12,delta**4/288,2*delta**4/288,delta**4/288)
 du=derivatives(s,st['u']);dv=derivatives(s,st['v'])
 product=(lambda a,b:padded_product(s,a,b)) if pad else (lambda a,b:a*b)
 # B: independently construct the symmetric velocity stress; do not use q.
 tau=[];tau2=[]
 for a,b in ((du,du),(du,dv),(dv,dv)):
  terms=[weight*product(x,y) for weight,x,y in zip(weights,a,b)]
  tau.append(sum(terms));tau2.append(sum(terms[:2]))
 sb=stress_source(s,tau);sb2=stress_source(s,tau2)
 if not pad:
  # A: actual frozen Clark4 implementation, not a rewritten approximation.
  sa=c4.source(s,c4.flux4(s,st,delta))
  sa2=c4.source(s,c4.flux2(st,delta))
  sa4=c4.source(s,c4.fourth_increment(s,st,delta))
 else:
  dw=derivatives(s,st['omega']);q=[];q2=[]
  for v in (du,dv):
   terms=[weight*product(a,b) for weight,a,b in zip(weights,v,dw)]
   q.append(sum(terms));q2.append(sum(terms[:2]))
  sa=c4.source(s,q);sa2=c4.source(s,q2);sa4=sa-sa2
 curl=d(st['v'],'x')-d(st['u'],'y')
 curlstate=dict(st,omega=curl,omega_x=d(curl,'x'),omega_y=d(curl,'y'))
 scurl=c4.source(s,c4.flux4(s,curlstate,delta)) if not pad else None
 deviatoric=[tau[0]-.5*(tau[0]+tau[2]),tau[1],tau[2]-.5*(tau[0]+tau[2])]
 diagnostics=dict(epsilon_eq=rel(sb,sa),SA_L2=norm(sa),SB_L2=norm(sb),difference_L2=norm(sa-sb),
  SA_RMS=rms(sa),SB_RMS=rms(sb),difference_Linf=float((sa-sb).abs().max()),
  order2_relative=rel(sb2,sa2),order4_relative=rel(sb-sb2,sa4),
  order_sum_reconstruction_relative=rel(sa2+sa4,sa),
  velocity_curl_to_omega_relative=rel(curl,w),
  divergence_velocity_relative=norm(d(st['u'],'x')+d(st['v'],'y'))/max(norm(d(st['u'],'x'))+norm(d(st['v'],'y')),EPSILON),
  stress_trace_removal_relative=rel(stress_source(s,deviatoric),sb),
  state_mean=float(w.mean()),source_mean_A=float(sa.mean()),source_mean_B=float(sb.mean()))
 if scurl is not None:
  diagnostics.update(stress_vs_curlomega_flux_relative=rel(sb,scurl),
   storedomega_vs_curlomega_flux_relative=rel(sa,scurl),
   error_vector_decomposition_relative=norm((sa-sb)-((sa-scurl)+(scurl-sb)))/max(norm(sa-sb),EPSILON))
 return sa,sb,diagnostics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='check01');a=ap.parse_args()
 assert Path(a.run).name==a.run and a.run not in ('','.','..')
 out=HERE/'results'/a.run;out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 def forbidden(*args,**kwargs):raise AssertionError('Forbidden: CNN, RHS or time integration')
 SPNSGSClosure.forward=forbidden
 SpectralVorticitySolver.rhs=forbidden;SpectralVorticitySolver.step=forbidden
 hashes={}
 def verify(p,expected=None):
  p=Path(p);digest=sha(p);assert expected is None or digest==expected,('Input SHA mismatch',str(p))
  hashes[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=digest;return p
 run=read(verify(RUN/'RUN.json'));last=read(verify(RUN/'LATEST_REPORT.json'))
 report=RUN/last['path'];manifest=read(verify(report/'MANIFEST.json',last['manifest_sha256']))
 expected={str(Path(k)):v for k,v in run['inventory']['input_sha256'].items()}
 for p in (C4PATH,ROOT/'spnsgs/solver.py'):
  verify(p,expected.get(str(p.relative_to(ROOT))))
 verify(HERE/'check_equivalence.py');verify(ROOT/'research/full_c4_prior_integration_v1_20261103/tests/test_integration.py')
 b=load(verify(RUN/'OPERATORS.npz'));assert b['kx'].dtype==np.float64
 cfgpath=ROOT/'results/revision_v2/runs/full/seed_2026_recovery01/effective_config.json'
 cfg=read(verify(cfgpath,run['inventory']['effective_config_sha256']))
 domain=cfg['domain'];length=domain['x_max_m']-domain['x_min_m']
 assert domain['y_max_m']-domain['y_min_m']==length
 delta=float(b['filter_width']);assert abs(delta-cfg['data']['filter_width_m'])<1e-14
 native=Operators(b);ideal=Operators(b,'ideal',length)
 description=dict(case_policy='Prespecified 12 saved states; no result-dependent selection',
  definitions={'A':'original Clark4 flux Source with production project/derivative/state',
   'B':'P33 curl(-div tau_NGM4), stress11/12/22 constructed independently',
   'stress':'tau_ij=Delta^2/12 sum_k u_i,k u_j,k + Delta^4/288 sum_kl u_i,kl u_j,kl',
   'omega_sign':'omega=D_x v-D_y u','mixed_Hessian_weight':2,
   'epsilon':'norm(A-B)/max(norm(A),1e-12)'},
  paper_equation='https://arxiv.org/html/2509.20365v3#S0.E4',paper_DOI='10.1103/v28b-5qmp',
  filter_width=delta,domain_length=length,LES_grid=100,mask='unchanged rectangular P33',
  inherited_limits={'native':NATIVE_LIMIT,'ideal':IDEAL_LIMIT,'padding':PAD_LIMIT},
  inherited_limit_source='research/full_c4_prior_integration_v1_20261103/tests/test_integration.py:test_c4_stress_curl_and_flux_source',
  control_policy='New stateless buffer instances only; original operators/checkpoints/results never modified',
  reference_quality='REFERENCE_NOT_CONVERGED; irrelevant to algebraic identity but still applies to physical accuracy')
 dump(out/'PROTOCOL.json',description)
 states=[]
 for n in (200,400,800):
  file=verify(CLARK/'steps'/f'{n:04d}.npz');z=load(file)
  info=json.loads(str(z['metadata'].item()));assert info['accepted_step']==n
  states.append(dict(id=f'Clark4_101_t{1+n*.005:g}',case=110240101,origin='Clark4 autonomous',seed='',time=1+n*.005,w=z['omega'],saved_source=z['observation__Source'],file=str(file.relative_to(ROOT))))
 clarkstatus=read(verify(CLARK/'STATUS.json'));cp=read(verify(CLARK/'CHECKPOINT.json'));verify(CLARK/cp['file'],cp['sha256']);assert clarkstatus['status']=='VERIFIED'
 for case,ns in ((110240101,(0,400,800)),(110240102,(400,800)),(110240104,(400,800))):
  ref=run['inventory']['references'][str(case)]
  if ref['kind']=='dense':
   name=ref['fields']['reference_omega'];file=verify(ROOT/name,expected[str(Path(name))]);arr=np.load(file,mmap_mode='r')
  else:arr=None
  for n in ns:
   if arr is None:
    name=ref['frames'][str(n)];file=verify(ROOT/name,expected[str(Path(name))]);w=load(file)['reference_omega']
   else:w=np.array(arr[4*n])
   states.append(dict(id=f'Reference_{case%1000}_t{1+n*.005:g}',case=case,origin='filtered reference',seed='',time=1+n*.005,w=w,file=str(file.relative_to(ROOT))))
 name='STATES_110240101_2026_Parameter-Hold.npy';file=verify(report/name,manifest['outputs'][name]);arr=np.load(file,mmap_mode='r')
 for n in (400,800):states.append(dict(id=f'Hold_101_2026_t{1+n*.005:g}',case=110240101,origin='DSF-Hold autonomous',seed=2026,time=1+n*.005,w=np.array(arr[n]),file=str(file.relative_to(ROOT))))
 assert len(states)==12
 rows=[];snapshots={};begin=time.perf_counter()
 kinds=('native','exact_k_native_inverse','native_k_exact_inverse','ideal')
 for state in states:
  w=torch.from_numpy(np.array(state['w'],copy=True));assert w.shape==(100,100) and torch.isfinite(w).all()
  context={k:state[k] for k in ('id','case','origin','seed','time','file')}
  for kind in kinds:
   s=Operators(b,kind,length);sa,sb,diag=evaluate(s,w,delta)
   if kind=='native':
    pA,pB,pdiag=evaluate(s,w,delta,pad=True)
    diag.update(padded150_A_relative=rel(pA,sa),padded150_B_relative=rel(pB,sb),padded150_epsilon_eq=pdiag['epsilon_eq'])
    assert max(diag['padded150_A_relative'],diag['padded150_B_relative'])<PAD_LIMIT
    if 'saved_source' in state:diag['A_to_archived_Clark4_Source_relative']=rel(sa,torch.from_numpy(state['saved_source']))
    snapshots.update({state['id']+'__'+k:v for k,v in dict(omega=array(w),S_A=array(sa),S_B=array(sb),difference=array(sa-sb)).items()})
   if kind=='ideal':snapshots[state['id']+'__ideal_difference']=array(sa-sb)
   limit=NATIVE_LIMIT if kind=='native' else IDEAL_LIMIT if kind=='ideal' else None
   rows.append(dict(**context,operators=kind,**diag,acceptance_limit=limit,status='PASS' if limit is not None and diag['epsilon_eq']<limit else 'FAIL' if limit is not None else 'DIAGNOSTIC_CONTROL'))
  print(state['id'],f"native={rows[-4]['epsilon_eq']:.4e}",f"ideal={rows[-1]['epsilon_eq']:.4e}",flush=True)
 table(out/'EQUIVALENCE_RESULTS.csv',rows);np.savez_compressed(out/'SAME_STATE_SOURCES.npz',**snapshots)
 # Independently quantify construction inconsistencies, without modifying them.
 k2=native.kx.square()+native.ky.square();active=native.dealias&(k2>0)
 ops=dict(native_k_vs_ideal_max_abs=float(torch.maximum((native.kx-ideal.kx).abs(),(native.ky-ideal.ky).abs()).max()),
  native_inverse_consistency_max_abs=float((k2*native.inverse_k_squared-1)[active].abs().max()),
  native_saved_k_squared_consistency_max_abs=float((native.k_squared-k2).abs().max()),
  mode_count=int(active.sum()),padded_product_grid=150)
 dump(out/'OPERATOR_DIAGNOSTICS.json',ops)
 for name,digest in hashes.items():assert sha(ROOT/name)==digest,('Original input mutated',name)
 dump(out/'INPUT_SHA256.json',hashes)
 native_rows=[r for r in rows if r['operators']=='native'];ideal_rows=[r for r in rows if r['operators']=='ideal']
 result=dict(status='PASS_WITH_NATIVE_OPERATOR_PRECISION_FLOOR' if all(r['status']=='PASS' for r in native_rows+ideal_rows) else 'FAIL',
  states=len(states),native_min=min(r['epsilon_eq'] for r in native_rows),native_max=max(r['epsilon_eq'] for r in native_rows),
  ideal_min=min(r['epsilon_eq'] for r in ideal_rows),ideal_max=max(r['epsilon_eq'] for r in ideal_rows),
  padded_max_A=max(r['padded150_A_relative'] for r in native_rows),padded_max_B=max(r['padded150_B_relative'] for r in native_rows),
  archived_Source_replay_max=max(r.get('A_to_archived_Clark4_Source_relative',0) for r in native_rows),
  source_means_max_abs=max(max(abs(r['source_mean_A']),abs(r['source_mean_B'])) for r in native_rows),
  new_CNN=0,new_RHS=0,new_time_steps=0,device='CPU',dtype='float64',seconds=time.perf_counter()-begin,
  interpretation='Same fourth-order Gaussian closure under kinematic and Galerkin compatibility assumptions; native implementation is not bitwise equal to stress curl. Native observed discrepancy must be reported, not replaced by ideal control.',
  originals_unchanged=True)
 dump(out/'SUMMARY.json',result)
 write_report(out,result,rows,ops)
 dump(out/'OUTPUT_SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
 print(json.dumps(result,indent=2))

def write_report(out,result,rows,ops):
 text='''# Clark4 与 PRL NGM4 的同状态 Source 等价性核验

应力公式独立使用作者原文 Eq.(4) 的二阶及四阶项：

tau_ij = Delta²/12 Σ_k (∂k u_i)(∂k u_j) + Delta⁴/288 Σ_kl (∂kl u_i)(∂kl u_j)。

二维混合 Hessian 项必须计两次。采用 omega=Dx v-Dy u，先作 f=-div(tau)，再作 S_B=P33(Dx f_y-Dy f_x)。S_A直接调用原clark4.py的flux4/source，未手工替代原实现。

来源：[作者全文 Eq.(4)](https://arxiv.org/html/2509.20365v3)，[PRL DOI](https://doi.org/10.1103/v28b-5qmp)。本次沿用我们自己的实际Delta与P33，不改成论文算例的过滤宽度或分辨率。

## 结果

'''
 text+=f"状态：{result['status']}。12个既有状态，CPU float64；0 CNN、0 RHS、0新时间步。\n\n"
 text+='|状态|生产算子 epsilon_eq|理想float64算子 epsilon_eq|\n|---|---:|---:|\n'
 for r in [r for r in rows if r['operators']=='native']:
  ideal=next(x for x in rows if x['id']==r['id'] and x['operators']=='ideal')
  text+=f"|{r['id']}|{r['epsilon_eq']:.8e}|{ideal['epsilon_eq']:.8e}|\n"
 text+=f"\n原生误差范围 {result['native_min']:.8e} 至 {result['native_max']:.8e}；纯float64算子对照 {result['ideal_min']:.8e} 至 {result['ideal_max']:.8e}。\n"
 text+=f"\n150²补零乘积相对原生collocation的最大Source差异：A为{result['padded_max_A']:.8e}，B为{result['padded_max_B']:.8e}。A相对已保存Clark4 Source最大重放误差为{result['archived_Source_replay_max']:.8e}。\n"
 text+='''
## 该如何解释

数学上，在二维不可压缩、omega=curl(u)、微分交换且投影后满足产品法则的条件下，对每个多重导数a=∂^alpha u，有curl[-div(a⊗a)]=-div[a curl(a)]。将二阶/四阶各项及相同系数相加，即得到两种Source表示的等价关系。这个推导针对最终涡量Source，不要求完整通量向量在某种规范下相等。

当前生产算子的kx、ky、k²、逆k²先按float32构造再转double，因此double计算并不能恢复其最初丢失的位数。原生omega与Dx v-Dy u存在重构误差，波数符号也并非理想整数符号。本次通过“只换精确波数、保留旧inverse”和“保留原波数、按其平方重建inverse”分别诊断，再使用两者都一致的纯float64控制。所有控制均是隔离对象，不覆盖生产缓冲区，也不进入任何旧LES轨迹。

生产模式的容差2e-6与理想控制2e-11继承此前制造解测试，而非看本次结果后放宽。PASS是给定算子精度与所选有限状态的代数验收，不是模型预测精度、稳定性或参考资格验收。不能把理想控制的误差当作当前生产算子的误差，也不能从同一闭合推出不同论文算例下相同的长期性能。

结论措辞应为：当前Clark4是同一四阶Gaussian NGM4闭合的涡量通量形式；在现有生产离散与所查状态上，两种Source在已识别的算子构造/运动学重构精度内一致，未达到逐位相等。原始五模型结果不需因此重算，仍保留REFERENCE_NOT_CONVERGED；本检查不改变参考资格。

## 文件

EQUIVALENCE_RESULTS.csv给出各原生/隔离控制、分阶残差、curl重构、去迹与padding检查。SAME_STATE_SOURCES.npz保留每个原始涡量、S_A、S_B、差值及理想差值。INPUT_SHA256.json记录源身份并在结束后复核未改动；OPERATOR_DIAGNOSTICS.json记录实际符号不一致程度。

复算：python -B check_equivalence.py --run check03（新目录，拒绝覆盖已有检查）。本次不修改论文主稿或生产代码。
'''
 (out/'REPORT_CN.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
