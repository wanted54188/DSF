"""Eight fresh matched runs; epoch-atomic resume, EMA and fixed selection."""
import copy
import math
import time
import numpy as np
import torch
import common as c
import data
import objective


def clone_state(s):return {k:v.detach().cpu().clone() for k,v in s.state_dict().items()}
def schedule(p,epoch):
    return max(0.,min(1.,(epoch-p['signed_warmup_epochs'])/max(p['signed_ramp_epochs'],1)))


def validation(s,cache,weights):
    sums={};l2=[]
    with torch.no_grad():
        for i in range(len(cache['time_s'])):
            st,f,source=data.forward_frame(s,cache,i)
            q=objective.quantities(s,cache,i,st,f,source)
            for k,v in q.items():sums[k]=sums.get(k,0.)+v.detach()/len(cache['time_s'])
            l2.append(torch.sqrt(q['source']))
    values={k:float(v) for k,v in sums.items()}
    if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite development validation objective')
    return dict(**values,objective=sum(weights[k]*values[k] for k in weights),source_relative_L2=float(torch.stack(l2).mean()))


def train_one(a,out,mode,seed,cache,val,device,resume=False,stop_after_epoch=None):
    p=a['protocol'];dest=out/'training'/f'{mode}_{seed}';meta=dest/'RUN.json';last=dest/'last.pt'
    if (dest/'DONE.json').exists():
        if not resume:raise FileExistsError('Completed run exists; use --resume to skip verified work')
        rec=c.read(dest/'DONE.json')
        if rec['code_sha256']!=a['code_sha256'] or c.sha(dest/'best.pt')!=rec['best_sha256']:raise ValueError('Completed checkpoint changed')
        print('REUSE COMPLETED '+dest.name,flush=True);return
    s=c.make_solver(mode,seed,device);initial=c.model_hash(s);data.enable_cached_prior(s)
    n=sum(x.numel() for x in s.parameters())
    ident=dict(code_sha256=a['code_sha256'],protocol=p,mode=mode,seed=seed,
        device=device,torch_version=str(torch.__version__),
        initial_parameter_sha256=initial,parameter_count=n,
        cache_sha256={str(x):c.read(out/'data'/str(x)/'DONE.json')['cache_sha256'] for x in c.CASES},
        initialization='fresh; no PoF checkpoint loaded',quality='REFERENCE_NOT_CONVERGED')
    opt=torch.optim.AdamW(s.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    lr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=p['epochs'],eta_min=p['minimum_learning_rate'])
    rng=np.random.default_rng(seed);history=[];ema=None;best=math.inf;best_epoch=None;start=1;elapsed0=0.;updates=0
    if dest.exists():
        if not resume or not meta.exists() or c.read(meta)!=ident:raise ValueError('Resume mode/config/data identity mismatch')
        if not last.is_file():raise FileNotFoundError('No committed epoch checkpoint; preserve this run and choose new output')
        v=torch.load(last,map_location=device,weights_only=False)
        if v['identity']!=ident:raise ValueError('Checkpoint identity mismatch')
        s.load_state_dict(v['model']);opt.load_state_dict(v['optimizer']);lr.load_state_dict(v['scheduler'])
        ema=v['ema'];best=v['best'];best_epoch=v['best_epoch'];history=v['history'];start=v['epoch']+1
        rng.bit_generator.state=v['shuffle_rng'];torch.set_rng_state(v['torch_rng'].cpu())
        if device.startswith('cuda') and v['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in v['cuda_rng']])
        elapsed0=v['wall_seconds'];updates=v['updates']
    else:dest.mkdir(parents=True);c.dump(meta,ident)
    for other in p['modes']:
        other_meta=out/'training'/f'{other}_{seed}'/'RUN.json'
        if other_meta.exists() and c.read(other_meta)['initial_parameter_sha256']!=initial:
            raise AssertionError('Paired modes did not start at identical parameters')
    tic=time.perf_counter();limit=p['epochs'] if stop_after_epoch is None else min(stop_after_epoch,p['epochs'])
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start,limit+1):
            epoch_start=time.perf_counter();s.train();s.closure.set_backscatter_factor(schedule(p,epoch));order=rng.permutation(len(cache['time_s']))
            means={};max_grad=0.
            for offset in range(0,len(order),p['batch_frames']):
                batch=order[offset:offset+p['batch_frames']];opt.zero_grad(set_to_none=True)
                for i in batch:
                    st,f,source=data.forward_frame(s,cache,int(i))
                    q=objective.quantities(s,cache,int(i),st,f,source);loss=objective.total(q,p['loss_weights'])
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite objective')
                    (loss/len(batch)).backward()
                    for k,v in q.items():means[k]=means.get(k,0.)+v.detach()/len(order)
                norm=torch.nn.utils.clip_grad_norm_(s.parameters(),p['gradient_clip'],error_if_nonfinite=True)
                max_grad=max(max_grad,float(norm));opt.step();updates+=1
                if epoch>=p['ema_start_epoch']:
                    if ema is None:ema={k:v.detach().clone() for k,v in s.state_dict().items()}
                    else:
                        with torch.no_grad():
                            for k,v in s.state_dict().items():ema[k].mul_(p['ema_decay']).add_(v,alpha=1-p['ema_decay'])
            means={k:float(v) for k,v in means.items()}
            lr.step();vrec=None
            if epoch==1 or epoch%p['validation_every_epochs']==0 or epoch==p['epochs']:
                raw=clone_state(s)
                if ema is not None:s.load_state_dict(ema)
                s.eval();s.closure.set_backscatter_factor(1.)
                vrec=validation(s,val,p['loss_weights'])
                if vrec['source']<best:
                    best=vrec['source'];best_epoch=epoch
                    c.save_pt(dest/'best.pt',dict(identity=ident,model=clone_state(s),epoch=epoch,
                        validation=vrec,selection=p['selection'],backscatter_factor=1.,EMA=ema is not None))
                s.load_state_dict(raw)
            elapsed=elapsed0+time.perf_counter()-tic
            row=dict(epoch=epoch,updates=updates,training=means,validation=vrec,best_epoch=best_epoch,
                best_validation_source_MSE=best,gradient_norm_max=max_grad,signed_factor=schedule(p,epoch),
                next_learning_rate=lr.get_last_lr()[0],epoch_seconds=time.perf_counter()-epoch_start,wall_seconds=elapsed)
            history.append(row)
            c.dump(dest/'HISTORY.json',history)
            c.save_pt(last,dict(identity=ident,model=clone_state(s),optimizer=opt.state_dict(),scheduler=lr.state_dict(),
                ema=ema,best=best,best_epoch=best_epoch,history=history,epoch=epoch,updates=updates,
                shuffle_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,wall_seconds=elapsed))
            c.dump(dest/'STATUS.json',dict(status='TRAINING' if epoch<p['epochs'] else 'TRAINING_COMPLETE',
                epoch=epoch,total_epochs=p['epochs'],best_epoch=best_epoch,wall_seconds=elapsed,
                mean_epoch_seconds=elapsed/epoch,estimated_remaining_seconds=elapsed/epoch*(p['epochs']-epoch)))
            if epoch==1 or epoch%5==0 or epoch==limit:
                metric=vrec['source_relative_L2'] if vrec else None
                print(f'{mode}/{seed} epoch {epoch}/{p["epochs"]} trainMSE={means["source"]:.5g} '
                      f'valL2={metric} best={best_epoch} wall={elapsed:.1f}s ETA={elapsed/epoch*(p["epochs"]-epoch):.1f}s',flush=True)
        if limit==p['epochs']:
            c.dump(dest/'DONE.json',dict(status='TRAINING_COMPLETE',code_sha256=a['code_sha256'],mode=mode,seed=seed,
                epochs=p['epochs'],optimizer_updates=updates,initial_parameter_sha256=initial,best_epoch=best_epoch,
                best_sha256=c.sha(dest/'best.pt'),last_sha256=c.sha(last),wall_seconds=elapsed0+time.perf_counter()-tic,
                training_forwards=p['epochs']*len(cache['time_s']),device=device,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                quality='REFERENCE_NOT_CONVERGED'))
    except BaseException as exc:
        c.dump(dest/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc),resume='Resume from last fully committed epoch; partial epoch is replayed'))
        raise


def run(a,out,device,resume=False,stop_after_epoch=None):
    c.verify(a);p=a['protocol'];torch.set_num_threads(p['cpu_threads'])
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable; choose --device cpu explicitly')
        torch.backends.cuda.matmul.allow_tf32=p['TF32'];torch.backends.cudnn.allow_tf32=p['TF32']
    cache=data.load_cache(a,out,p['train_seed'],device);val=data.load_cache(a,out,p['validation_seed'],device)
    # Manual train preflight compares cached reference-state and current
    # device's live operators. No training update or validation label is used.
    adapter=[]
    for mode in p['modes']:
        live=c.make_solver(mode,2026,device);cached=c.make_solver(mode,2026,device);data.enable_cached_prior(cached)
        with torch.no_grad():
            native=-live.rhs(cache['omega'][0],1.)['div_q']
            _,_,fast=data.forward_frame(cached,cache,0)
            relative=float((native-fast).square().mean().sqrt()/native.square().mean().sqrt().clamp_min(1e-12))
        if relative>p['reference_adapter_relative_tolerance']:raise ValueError('Cached/live device adapter mismatch')
        adapter.append(dict(mode=mode,device=device,cached_live_relative=relative,tolerance=p['reference_adapter_relative_tolerance']))
    c.dump(out/'TRAINING_ADAPTER_CHECK.json',adapter)
    for seed in p['training_seeds']:
        for mode in p['modes']:train_one(a,out,mode,seed,cache,val,device,resume,stop_after_epoch)
    records=[]
    for seed in p['training_seeds']:
        for mode in p['modes']:
            d=out/'training'/f'{mode}_{seed}'
            if not (d/'DONE.json').exists():continue
            r=c.read(d/'DONE.json');records.append(dict(mode=mode,seed=seed,path=str(d/'best.pt'),sha256=r['best_sha256'],
                best_epoch=r['best_epoch'],initial_parameter_sha256=r['initial_parameter_sha256']))
    if len(records)==8:
        c.dump(out/'BASELINE_REGISTRY.json',dict(version=p['version'],code_sha256=a['code_sha256'],models=records,
            cache_sha256={str(x):c.read(out/'data'/str(x)/'DONE.json')['cache_sha256'] for x in c.CASES},
            reference_quality='REFERENCE_NOT_CONVERGED',train_initial=p['train_seed'],development_validation_initial=p['validation_seed'],
            primary_full_baseline='NEW_TRAINED_FULL-C2',old_PoF_models='HISTORICAL_CONTEXT_ONLY'))
        c.dump(c.HERE/'LATEST_BASELINE.json',dict(run=str(out),registry=str(out/'BASELINE_REGISTRY.json'),
            registry_sha256=c.sha(out/'BASELINE_REGISTRY.json'),status='EIGHT_MATCHED_RUNS_COMPLETE',
            quality='REFERENCE_NOT_CONVERGED'))
