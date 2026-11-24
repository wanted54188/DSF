"""Automatic final paired report/rebuild. Reads committed artifacts; never a CNN."""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
import pc_common as c
import pc_metrics as m
import pc_storage as storage


WINDOWS={'full':[n for n in range(10,801,10)],'early':[n for n in range(40,241,10)],
         'late':[n for n in range(250,801,10)],'t5':[800]}


def observed(z): return storage.observation(z)
def classify(statuses,pairs,speeds,endpoint_distance):
    pc=[r for r in statuses if r['method']=='Parameter-PC']
    if any(r['status'].startswith('FAILED') for r in pc):
        return 'STOP','闭环反馈失败的当前候选'
    if len(pc)!=6 or not all(r['status']=='VERIFIED' for r in statuses):
        return 'CONDITIONAL','NOT_RUN_INCOMPLETE_CONTROLS_OR_INPUTS'
    target=[r for r in pairs if r['candidate']=='Parameter-PC' and r['baseline'] in ('Aug','Parameter-Hold') and r['window'] in ('full','t5') and r['band']=='all']
    accurate=len(target)==24 and all(r['omega_MSE_improvement_pct'] is not None and r['omega_MSE_improvement_pct']>0 for r in target)
    physical=[r for r in pairs if r['candidate']=='Parameter-PC' and r['baseline']=='Aug' and r['window']=='full' and r['band'] in ('all','high')]
    no_degrade=all(r['mean_abs_Z_bias_change'] is not None and r['mean_abs_Z_bias_change']<=0 and
                  (r['band']!='all' or r['mean_abs_K_bias_change']<=0) for r in physical)
    speed=[r for r in speeds if r['candidate']=='Parameter-PC' and r['baseline']=='Fresh-C4']
    faster=len(speed)==6 and all(r['measured_loop_speedup'] is not None and r['measured_loop_speedup']>1 for r in speed)
    if accurate and no_degrade and faster: return 'CONDITIONAL','本次固定实验内有效的低成本时间预测SGS候选'
    near=len(endpoint_distance)==6 and all(r['PC_to_Fresh_endpoint_MSE']<r['Hold_to_Fresh_endpoint_MSE'] for r in endpoint_distance)
    if near and faster: return 'CONDITIONAL','仅近似Fresh-C4的计算加速候选，未证真实预测优势'
    return 'CONDITIONAL','当前候选未建立一致的精度、物理与成本优势'


def journals(root):
    calls=defaultdict(lambda:dict(CNN=0,RHS=0)); begin=defaultdict(int); end=defaultdict(int); wall=defaultdict(float)
    for path in sorted((root/'invocations').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line); scope=row['scope']; kind=row['kind']
            if kind in ('CNN','RHS'): calls[scope][kind]+=1
            elif kind=='SCOPE_BEGIN': begin[scope]+=1
            elif kind=='SCOPE_END': end[scope]+=1; wall[scope]+=row['seconds']
    return calls,{scope:dict(seconds=wall[scope] if begin[scope]==end[scope] else None,
                           complete_wall_record=begin[scope]==end[scope],sessions=begin[scope]) for scope in set(begin)|set(end)}


def summarize(group,steps):
    bystep={r['step']:r for r in group}; complete=all(n in bystep and bystep[n]['reference_available'] for n in steps)
    if not complete: return dict(status='NOT_RUN_INCOMPLETE_WINDOW',samples=sum(n in bystep and bystep[n]['reference_available'] for n in steps))
    values=[bystep[n] for n in steps]
    mean=lambda key:float(np.mean([r[key] for r in values]))
    return dict(status='VERIFIED',samples=len(values),mean_omega_MSE=mean('omega_MSE'),
        mean_omega_relative_L2=mean('omega_relative_L2'),mean_u_relative_L2=mean('u_relative_L2'),
        mean_v_relative_L2=mean('v_relative_L2'),mean_velocity_relative_L2=mean('velocity_relative_L2'),
        mean_abs_K_bias=float(np.mean([abs(r['K_relative_bias']) for r in values])),
        mean_abs_Z_bias=float(np.mean([abs(r['Z_relative_bias']) for r in values])),
        mean_K=mean('K'),mean_Z=mean('Z'))


def build_report(root,template=False):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with storage.RunLock(root): return _build_report_unlocked(root,template)


def _build_report_unlocked(root,template=False):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    out=root/'reports'/f'report_{len(list((root/"reports").glob("report_*")))+1:04d}'
    out.mkdir(parents=True,exist_ok=False); begin=time.perf_counter()
    if template:
        statuses=[dict(case=case,seed=seed,method=method,status='NOT_RUN',reference_quality=c.QUALITY)
                  for case in c.P['cases'] for seed in c.P['seeds'] for method in c.METHODS]
        c.csvout(out/'TRAJECTORY_STATUS.csv',statuses)
        for name,cols in [('LONG_ROLLOUT_METRICS.csv',['case','seed','method','step','time_s','band','omega_MSE']),
                          ('PAIRED_LONG_SUMMARY.csv',['case','seed','window','band','candidate','baseline','status','omega_MSE_improvement_pct']),
                          ('COSTS.csv',['case','seed','method','actual_CNN','actual_RHS','measured_loop_seconds'])]: c.csvout(out/name,[],cols)
        c.dump(out/'DECISION.json',dict(status='NOT_RUN',decision='NOT_RUN',reference_quality=c.QUALITY))
        (out/'REPORT_CN.md').write_text('完整LES尚未运行，四方法×三流动×两seed均NOT_RUN。\n\n手动执行driver.py --execute-full-rollout后自动产生A—F答案、完整配对数据及图。\nREFERENCE_NOT_CONVERGED；104不是盲测。\n',encoding='utf-8')
        return out
    run=c.read(root/'RUN.json'); c.require(run['code_identity']==c.local_identity(),'Report code/protocol changed from run')
    for name,digest in run['inventory']['input_sha256'].items(): c.require(c.sha(c.ROOT/name)==digest,'Input changed before report')
    existing=sorted(root.glob('DATA_SHA256_*.json'))
    if existing:
        for name,digest in c.read(existing[-1]).items(): c.require(c.sha(root/name)==digest,'Committed artifact changed: '+name)
    # Append a new immutable checksum generation, including newly resumed work.
    manifest=root/f'DATA_SHA256_{len(existing)+1:04d}.json'
    paths=[p for p in root.rglob('*') if p.is_file() and ('reports' not in p.relative_to(root).parts)
           and p.suffix in ('.npz','.jsonl')]
    c.dump(manifest,{str(p.relative_to(root)):c.sha(p) for p in paths})
    op=c.load(root/'OPERATORS.npz') if (root/'OPERATORS.npz').exists() else None
    calls,walls=journals(root); statuses=[]; metrics=[]; spectra=[]; fluxes=[]; consistency=[]; checks=[]; rkchecks=[]; stagechecks=[]; costs=[]
    shared=[]
    for case in c.P['cases']:
        for seed in c.P['seeds']:
            scope=f'{case}_{seed}_STARTUP'
            shared.append(dict(case=case,seed=seed,actual_CNN=calls[scope]['CNN'],actual_RHS=calls[scope]['RHS'],
                measured_loop_seconds=walls.get(scope,{}).get('seconds'),shared_by_all_four_methods=True))
    c.csvout(out/'SHARED_STARTUP_COST.csv',shared)
    endpoint={}; groups=defaultdict(list);head_consistency=[]
    for case in c.P['cases']:
        ref=c.References(run['inventory']['references'][str(case)]) if str(case) in run['inventory']['references'] else None
        if ref and op is not None:
            for n in range(0,801,10):
                sample=ref.at(n)
                if sample is None: continue
                spectra.extend(m.reference_spectra(sample,op,
                    dict(case=case,seed=0,method='REFERENCE',step=n,time_s=1+n*c.H)))
        for seed in c.P['seeds']:
            startup=root/'startup'/f'{case}_{seed}'/'STARTUP.npz'
            start=c.load(startup) if startup.exists() else None
            for method in c.METHODS:
                scope=c.key(case,seed,method); folder=root/'trajectories'/scope
                status=c.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else dict(status='NOT_RUN')
                ctx=dict(case=case,seed=seed,method=method)
                statuses.append(dict(**ctx,**status)); per=defaultdict(list)
                wall=walls.get(scope,dict(seconds=None,complete_wall_record=False,sessions=0))
                costs.append(dict(**ctx,status=status['status'],actual_CNN=calls[scope]['CNN'],actual_RHS=calls[scope]['RHS'],
                    measured_loop_seconds=wall['seconds'],wall_record_complete=wall['complete_wall_record'],sessions=wall['sessions'],
                    solver_seconds=status.get('solver_wall_seconds'),endpoint_seconds=status.get('endpoint_wall_seconds'),
                    IO_seconds=status.get('IO_wall_seconds'),storage_bytes=storage.tree_bytes(folder),
                    process_rss_bytes=status.get('resource',{}).get('process_rss_bytes'),
                    CUDA_peak_allocated_bytes=status.get('resource',{}).get('CUDA_peak_allocated_bytes')))
                if start is None or op is None: continue
                states=[start['omega0'],start['omega1']]; previous=start['omega1']
                # Reconstruct shared startup once per logical method; its cost remains shared.
                warm=m.verify_step(start['omega0'],start['omega1'],start['rhs'],op)
                rkchecks.append(dict(**ctx,step=0,common_startup=True,**warm))
                observed_snapshots=[(0,observed(start))]
                accepted=sorted((folder/'steps').glob('*.npz'))
                for path in accepted:
                    z=c.load(path); info=c.meta(z); n=info['accepted_step']
                    c.require(n==len(states),'Missing accepted state in report')
                    reconstruction=m.verify_step(previous,z['omega'],z['rhs'],op)
                    rkchecks.append(dict(**ctx,step=n-1,common_startup=False,**reconstruction))
                    previous=z['omega']; states.append(previous)
                    if observed(z): observed_snapshots.append((n-1,observed(z)))
                    for record in info['stages']:
                        stagechecks.append(dict(**ctx,**{k:v for k,v in record.items() if k!='budget'}))
                if (folder/'ENDPOINT.npz').exists():
                    z=c.load(folder/'ENDPOINT.npz'); observed_snapshots.append((800,{k:v for k,v in z.items() if k!='metadata'}))
                # Full accepted time series only; never pad a failed run with zero states.
                np.save(out/f'STATES_{scope}.npy',np.stack(states),allow_pickle=False)
                if len(states)==801: endpoint[case,seed,method]=states[-1]
                for n,w in enumerate(states):
                    reference=ref.at(n) if ref else None
                    rows=m.state_rows(w,reference['reference_omega'] if reference else None,op,
                                      dict(**ctx,step=n,time_s=1+n*c.H))
                    metrics.extend(rows)
                    for row in rows: groups[case,seed,method,row['band']].append(row)
                for n,snap in observed_snapshots:
                    ss,fl,ch,co=m.snapshot(snap,op,float(op['nu']),dict(**ctx,step=n,time_s=1+n*c.H))
                    spectra.extend(ss); fluxes.append(fl); checks.append(ch); consistency.extend(co)
                    if 'fresh_heads' in snap and snap['predicted_heads'].shape==snap['fresh_heads'].shape:
                        for channel in range(7):
                            error=float(np.mean((snap['predicted_heads'][:,channel]-snap['fresh_heads'][:,channel])**2))
                            power=float(np.mean(snap['fresh_heads'][:,channel]**2))
                            head_consistency.append(dict(**ctx,step=n,time_s=1+n*c.H,channel=channel,MSE=error,
                                fresh_head_power=power,relative_L2=(error/power)**.5 if power else None))
                print(f'Report rebuilt {case%1000}/{seed}/{method}: {len(states)} accepted states; {status["status"]}',flush=True)
    summary=[]; smap={}
    for (case,seed,method,band),rows in groups.items():
        for window,steps in WINDOWS.items():
            result=summarize(rows,steps); entry=dict(case=case,seed=seed,method=method,band=band,window=window,**result)
            summary.append(entry); smap[case,seed,method,band,window]=entry
    paired=[]
    for case in c.P['cases']:
        for seed in c.P['seeds']:
            for candidate in c.METHODS:
                for baseline in c.METHODS:
                    if candidate==baseline: continue
                    for band in ('all','low','mid','high','corner'):
                        for window in WINDOWS:
                            a=smap.get((case,seed,candidate,band,window),{}); b=smap.get((case,seed,baseline,band,window),{})
                            ok=a.get('status')==b.get('status')=='VERIFIED'
                            row=dict(case=case,seed=seed,candidate=candidate,baseline=baseline,band=band,window=window,
                                status='VERIFIED' if ok else 'NOT_RUN_INCOMPLETE_WINDOW',
                                omega_MSE_improvement_pct=100*(1-a['mean_omega_MSE']/b['mean_omega_MSE']) if ok and b['mean_omega_MSE'] else None,
                                candidate_omega_MSE=a.get('mean_omega_MSE'),baseline_omega_MSE=b.get('mean_omega_MSE'),
                                mean_abs_K_bias_change=a['mean_abs_K_bias']-b['mean_abs_K_bias'] if ok else None,
                                mean_abs_Z_bias_change=a['mean_abs_Z_bias']-b['mean_abs_Z_bias'] if ok else None)
                            paired.append(row)
    costidx={(r['case'],r['seed'],r['method']):r for r in costs}; speeds=[]; distances=[]
    for case in c.P['cases']:
        for seed in c.P['seeds']:
            for candidate,baseline in (('Parameter-PC','Fresh-C4'),('Parameter-Hold','Fresh-C4'),('Parameter-PC','Aug'),('Parameter-PC','Parameter-Hold')):
                a=costidx[case,seed,candidate]; b=costidx[case,seed,baseline]
                complete=a['status']==b['status']=='VERIFIED'; timing=complete and a['measured_loop_seconds'] and b['measured_loop_seconds']
                speeds.append(dict(case=case,seed=seed,candidate=candidate,baseline=baseline,
                    equal_horizon_complete=complete,CNN_saved_fraction=1-a['actual_CNN']/b['actual_CNN'] if complete and b['actual_CNN'] else None,
                    measured_loop_speedup=b['measured_loop_seconds']/a['measured_loop_seconds'] if timing else None,
                    measured_solver_speedup=(b['solver_seconds']+b['endpoint_seconds'])/(a['solver_seconds']+a['endpoint_seconds']) if complete else None))
            if all((case,seed,k) in endpoint for k in ('Parameter-PC','Parameter-Hold','Fresh-C4')):
                distances.append(dict(case=case,seed=seed,PC_to_Fresh_endpoint_MSE=float(np.mean((endpoint[case,seed,'Parameter-PC']-endpoint[case,seed,'Fresh-C4'])**2)),
                    Hold_to_Fresh_endpoint_MSE=float(np.mean((endpoint[case,seed,'Parameter-Hold']-endpoint[case,seed,'Fresh-C4'])**2))))
    decision,category=classify(statuses,paired,speeds,distances)
    timechecks=[]
    for path in sorted((root/'timecheck').glob('*/DONE.json')):
        row=c.read(path);folder=path.parent
        if row['status']=='VERIFIED_LOCAL_DT_CHECK':
            z=c.load(folder/'STATES.npz');h=c.H/2;ks=z['rhs']
            mid=m.project(z['initial']+(h/6)*(ks[0]+2*ks[1]+2*ks[2]+ks[3]),op)
            m.verify_step(z['initial'],mid,ks[:4],op,h=h)
            m.verify_step(mid,z['two_half'],ks[4:],op,h=h)
            row['independent_half_RK_reconstruction']='PASS'
        timechecks.append(row)
    if decision!='STOP' and any(r['method']=='Parameter-PC' and r['status'].startswith('FAILED') for r in timechecks):
        decision='CONDITIONAL_TIME_DISCRETIZATION_NOT_ROBUST';category='主步实验与半步物理检查不一致，不能判定方法可靠'
    elif decision!='STOP' and (len(timechecks)!=len(c.P['cases'])*len(c.P['seeds'])*len(c.METHODS)*len(c.P['timestep_diagnostic']['steps'])
                              or any(r['status']!='VERIFIED_LOCAL_DT_CHECK' for r in timechecks)):
        decision='CONDITIONAL';category='NOT_RUN_INCOMPLETE_LOCAL_TIME_CHECKS; '+category
    if run.get('synthetic_software_test'):
        decision='SOFTWARE_TEST_ONLY'; category='SYNTHETIC_NOT_REAL_LES_EVIDENCE'
    for name,rows,cols in [('TRAJECTORY_STATUS.csv',statuses,None),('LONG_ROLLOUT_METRICS.csv',metrics,['case','seed','method','step','time_s','band','omega_MSE']),
        ('WINDOW_SUMMARY.csv',summary,['case','seed','method','band','window','status']),('PAIRED_LONG_SUMMARY.csv',paired,None),
        ('SPECTRA.csv',spectra,['case','seed','method','step','time_s','shell']),('FLUX_BUDGETS.csv',fluxes,['case','seed','method','step']),
        ('FRESH_CONSISTENCY.csv',consistency,['case','seed','method','step','band']),('SNAPSHOT_IDENTITIES.csv',checks,['case','seed','method','step']),
        ('RK4_IDENTITIES.csv',rkchecks,['case','seed','method','step']),('STAGE_PHYSICAL.csv',stagechecks,['case','seed','method','step','stage']),
        ('COSTS.csv',costs,None),('MEASURED_SPEEDUPS.csv',speeds,None),('FRESH_ENDPOINT_DISTANCES.csv',distances,['case','seed','PC_to_Fresh_endpoint_MSE','Hold_to_Fresh_endpoint_MSE'])]:
        c.csvout(out/name,rows,cols if not rows else None)
    c.csvout(out/'HEAD_CONSISTENCY.csv',head_consistency,None if head_consistency else ['case','seed','method','step','channel','MSE'])
    c.csvout(out/'TIME_STEP_SENSITIVITY.csv',timechecks,None if timechecks else ['case','seed','method','step','status'])
    decode_calls=defaultdict(int)
    for path in sorted((root/'invocations').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            if event['kind']=='ALGEBRAIC_DECODE':decode_calls[event['scope']]+=1
    c.csvout(out/'ALGEBRAIC_DECODER_CALLS.csv',[dict(scope=k,calls=v) for k,v in decode_calls.items()],['scope','calls'])
    reuse=[]
    for entry in run['inventory'].get('source_reuse',[]):
        for row in c.csvread(c.ROOT/entry['path']):
            if row['model']!='FULL-C4-Augmented': continue
            if entry['kind']=='orbit_reference_scores' and int(row['case'])!=entry['case']: continue
            reuse.append(dict(case=entry['case'],seed=row['seed'],time_s=row['time_s'],band=row['band'],
                              reference_Source_MSE=row.get('source_mse',row.get('MSE')),identity='FROZEN_REFERENCE_STATE_ONLY'))
    c.csvout(out/'REFERENCE_SOURCE_REUSED.csv',reuse,['case','seed','time_s','band','reference_Source_MSE','identity'])
    plot(metrics,spectra,out,synthetic=run.get('synthetic_software_test',False))
    write_cn(out,decision,category,statuses,paired,speeds,distances)
    final_report=out/'REPORT_CN.md'
    details=['','## 当前候选的准确含义与局部步长复核','',
        '这是完整七通道head的历史预测，不再是旧方法的历史向量通量相加。四朝向保持各自raw坐标，当前状态重新解码；未更新网络权重或发明新的limiter。',
        '单朝向原解码结构与联合历史/状态的C4关系可证明；最终signed均值没有全状态保域保证，所有PASS仅表示实际stage。不能把实验成功写成普遍稳定性定理。',
        f'固定局部步长复核共{len(timechecks)}项；通过{sum(r["status"]=="VERIFIED_LOCAL_DT_CHECK" for r in timechecks)}项；失败{sum(r["status"].startswith("FAILED") for r in timechecks)}项。详情见TIME_STEP_SENSITIVITY.csv。',
        '复核从同一自主状态及该步已经刷新的相同历史锚点出发，比较原h一步与两个h/2；两半步之间不刷新，历史时间单位仍为h=.005。它是局部敏感性，不是完整时窗的时间收敛证明，正式dt未改变。',
        'HEAD_CONSISTENCY.csv记录同状态预测head与fresh head误差；同状态Source一致性见FRESH_CONSISTENCY.csv。所有额外代数解码调用单列，不冒称CNN调用或免费成本。']
    final_report.write_text(final_report.read_text(encoding='utf-8')+'\n'.join(details)+'\n',encoding='utf-8')
    if run.get('synthetic_software_test'):
        report=out/'REPORT_CN.md'
        report.write_text('**SYNTHETIC SOFTWARE TEST ONLY：以下为程序验收数据，不是真实LES实验或研究结论。**\n\n'+report.read_text(encoding='utf-8'),encoding='utf-8')
    c.dump(out/'DECISION.json',dict(decision=decision,category=category,reference_quality=c.QUALITY,
        physical_bound_scope='CLOSED_ENVELOPE_ONLY; bounded head reachability NOT_RUN',
        full_experiment_status='VERIFIED' if all(r['status']=='VERIFIED' for r in statuses) else 'INCLUDES_FAILURE_OR_NOT_RUN',
        previous_joint_gate='FAIL_UNCHANGED'))
    c.dump(out/'MANIFEST.json',dict(report_seconds=time.perf_counter()-begin,execution_identity=c.sha(root/'RUN.json'),
        data_manifest=manifest.name,data_manifest_sha256=c.sha(manifest),
        reference_quality=c.QUALITY,outputs={str(p.relative_to(out)):c.sha(p) for p in out.iterdir() if p.is_file()}))
    c.dump(root/'LATEST_REPORT.json',dict(path=str(out.relative_to(root)),manifest_sha256=c.sha(out/'MANIFEST.json')))
    print('FINAL:',decision,category,'report:',out,flush=True)
    return out


def write_cn(out,decision,category,statuses,paired,speeds,distances):
    text=['# 历史旋转预测PC：完整自主LES固定对照','',f'**判断：{decision}；分类：{category}。**',
        'REFERENCE_NOT_CONVERGED。101为训练流动；102、104均已揭盲，104不是独立盲测。','',
        '共同启动为t=1→1.005的一步Aug；其后四方法分别自主运行。History第一帧刷新先pending，两帧齐备才更新；步内锚点冻结，tau从真实最后锚点累计，最大5。未改参数、周期、损失、界或limiter。','',
        '## A. PC能否比原Aug更准确，以及C. 是否优于Hold','',
        '下表为共同81时刻、排除t=1的全窗平均omega绝对MSE改善百分比；终点也单独列出。不是相对L2改善。未完成窗口留NOT_RUN，不用部分窗口代替完整结果。','',
        '|流动/seed|PC对Aug全窗MSE改善|PC对Aug终点改善|PC对Hold全窗改善|PC状态|','|---|---:|---:|---:|---|']
    idx={(r['case'],r['seed'],r['candidate'],r['baseline'],r['band'],r['window']):r for r in paired}
    fmt=lambda v:'NOT_RUN' if v is None else f'{v:+.4f}%'
    for case in c.P['cases']:
        for seed in c.P['seeds']:
            get=lambda b,w:idx[case,seed,'Parameter-PC',b,'all',w]['omega_MSE_improvement_pct']
            status=next(r['status'] for r in statuses if (r['case'],r['seed'],r['method'])==(case,seed,'Parameter-PC'))
            text.append(f'|{case%1000}/{seed}|{fmt(get("Aug","full"))}|{fmt(get("Aug","t5"))}|{fmt(get("Parameter-Hold","full"))}|{status}|')
    text+=['','## B. 与Fresh-C4的差距','',
        '|流动/seed|PC对Fresh全窗MSE改善|终点MSE改善|全窗绝对K偏差变化|全窗绝对Z偏差变化|谱尾绝对Z偏差变化|',
        '|---|---:|---:|---:|---:|---:|']
    for case in c.P['cases']:
        for seed in c.P['seeds']:
            a=idx[case,seed,'Parameter-PC','Fresh-C4','all','full']; b=idx[case,seed,'Parameter-PC','Fresh-C4','all','t5']
            high=idx[case,seed,'Parameter-PC','Fresh-C4','high','full']
            number=lambda v:'NOT_RUN' if v is None else f'{v:+.6e}'
            text.append(f'|{case%1000}/{seed}|{fmt(a["omega_MSE_improvement_pct"])}|{fmt(b["omega_MSE_improvement_pct"])}|'
                f'{number(a["mean_abs_K_bias_change"])}|{number(a["mean_abs_Z_bias_change"])}|{number(high["mean_abs_Z_bias_change"])}|')
    text+=['','偏差变化为candidate−baseline，正值表示绝对偏差增加；不是改善百分比。',
        'PAIRED_LONG_SUMMARY.csv包含全频/低/中/高/角部、全窗/早窗/后半窗/终点的PC与Fresh绝对MSE及改善百分比、平均绝对K/Z相对偏差变化；WINDOW_SUMMARY.csv给出u/v/omega相对L2和绝对MSE。FRESH_ENDPOINT_DISTANCES.csv直接比较不同自主轨迹的终点距离，不能当成真值误差。',
        'FRESH_CONSISTENCY.csv比较同一个History自主状态上的历史修正与fresh修正，属于模型一致性。参考态Source误差另放REFERENCE_SOURCE_REUSED.csv；不得将参考场的SGS标签当成自主场的同状态真值。','',
        '## D. 实际成本','', '|流动/seed|PC相对Fresh的实测CNN节省|实测轨迹循环加速比|','|---|---:|---:|']
    for r in speeds:
        if r['candidate']=='Parameter-PC' and r['baseline']=='Fresh-C4':
            saving='NOT_RUN' if r['CNN_saved_fraction'] is None else f'{100*r["CNN_saved_fraction"]:.3f}%'
            speed='NOT_RUN' if r['measured_loop_speedup'] is None else f'{r["measured_loop_speedup"]:.3f}x'
            text.append(f'|{r["case"]%1000}/{r["seed"]}|{saving}|{speed}|')
    text+=['','CNN/RHS来自逐调用追加日志，包括中断作废调用。轨迹循环计时包括刷新、检查、保存；solver计时包含stage与刷新，CNN单独同步计时写在逐步元数据。共享启动另计，不重复分摊。失败短轨迹不得称作t5加速。硬中断若缺少完整wall结束记录，则加速留NOT_RUN。COSTS.csv另含显存/RSS和存储。','',
        '## E. 物理与数值失败','', '|流动/seed/方法|状态|首次失败时间|stage|分支/原因|','|---|---|---:|---:|---|']
    for r in statuses:
        details=r.get('details',{}); branches=[name for name,key in [('qd','qd_envelope_pass'),('qb amplitude','signed_amplitude_pass'),('qb transfer','signed_transfer_pass')] if details.get(key) is False]
        text.append(f'|{r["case"]%1000}/{r["seed"]}/{r["method"]}|{r["status"]}|{r.get("first_failed_time","")}|{r.get("first_failed_stage","")}|{", ".join(branches)} {r.get("reason","")}|')
    text+=['','STAGE_PHYSICAL.csv及失败NPZ/收据记录分支越界点数、最大比例、c/A证书和原状态。Fresh群平均signed域没有普遍凸闭合保证；本轮仅报告实际所测包络。所有运行均严格停止，无修复或越界后续轨迹。',
        'K/Z总量、高频径向谱尾、矩形角部均保留；正SGS传递和非单调谱尾不自动判不稳定。没有既有长期异常谱尾数值门槛，因此只做配对描述；100倍总K/Z安全界来自旧协议。','',
        '## F. 研究结论和证据边界','',f'{category}。总体{decision}。',
        '分类按运行前协议的方向性证据规则产生，不是补造的JCP验收阈值。若不能完成控制组则无法完整分类；所有失败仍保留。此前联合物理门槛FAIL并未被改写。',
        '完整accepted状态和四stage RHS可无CNN复算RK4与每个步的离散K/Z收支；RK4二次型缺陷单列，不强行要求RK4精确保守。每个0.05检查点的通量、Source、证书、预算和谱被独立重构。其余stage的完整通量独立再推理NOT_RUN，不能把运行时证书冒称全部独立重算。',
        '104的后半窗缺少密集参考；未插值、未补算DNS。所有原生步有状态/K/Z，但误差只在原档精确时间计算。参考时空收敛仍未证明。']
    (out/'REPORT_CN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def plot(rows,spectra,out,synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scope='SYNTHETIC SOFTWARE TEST: ' if synthetic else ''
    fig,axs=plt.subplots(3,2,figsize=(12,11))
    for i,case in enumerate(c.P['cases']):
        for j,seed in enumerate(c.P['seeds']):
            ax=axs[i,j]
            for method in c.METHODS:
                rs=[r for r in rows if (r['case'],r['seed'],r['method'],r['band'])==(case,seed,method,'all') and r['omega_MSE'] is not None]
                ax.semilogy([r['time_s'] for r in rs],[max(r['omega_MSE'],1e-30) for r in rs],label=method)
            ax.set_title(f'{case%1000}/{seed}'); ax.set_xlabel('t'); ax.set_ylabel('omega absolute MSE'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(scope+'Autonomous LES; REFERENCE_NOT_CONVERGED'); fig.tight_layout()
    fig.savefig(out/'TRAJECTORY_ERRORS.pdf'); fig.savefig(out/'TRAJECTORY_ERRORS.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(12,10))
    for i,case in enumerate(c.P['cases']):
        for j,seed in enumerate(c.P['seeds']):
            ax=axs[i,j]
            for method in c.METHODS:
                rs=[r for r in rows if (r['case'],r['seed'],r['method'],r['band'])==(case,seed,method,'high')]
                ax.plot([r['time_s'] for r in rs],[r['Z'] for r in rs],label=method)
            ax.set_title(f'{case%1000}/{seed}: high-shell Z'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(scope+'Radial high shell >=30; corner is separate in CSV'); fig.tight_layout()
    fig.savefig(out/'SPECTRAL_TAIL.pdf'); fig.savefig(out/'SPECTRAL_TAIL.png',dpi=150); plt.close(fig)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out/'PHYSICS_TIME_SERIES.pdf') as pdf:
        for case in c.P['cases']:
            for seed in c.P['seeds']:
                fig,axes=plt.subplots(2,3,figsize=(13,7))
                specs=[('all','omega_MSE'),('mid','omega_MSE'),('all','K_relative_bias'),
                       ('all','Z_relative_bias'),('high','Z'),('corner','Z')]
                for ax,(band,quantity) in zip(axes.flat,specs):
                    for method in c.METHODS:
                        rs=[r for r in rows if (r['case'],r['seed'],r['method'],r['band'])==(case,seed,method,band) and r.get(quantity) is not None]
                        ax.plot([r['time_s'] for r in rs],[r[quantity] for r in rs],label=method)
                    ax.set_title(band+': '+quantity); ax.grid(alpha=.2); ax.set_xlabel('t'); ax.legend(fontsize=7)
                fig.suptitle(scope+f'{case%1000}/{seed}; partial curves end at first failure; REFERENCE_NOT_CONVERGED')
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(14,11))
    for i,case in enumerate(c.P['cases']):
        for j,quantity in enumerate(('E','T_K_SGS','T_Z_SGS')):
            ax=axs[i,j]
            for seed in c.P['seeds']:
                for method in c.METHODS:
                    rs=[r for r in spectra if (r['case'],r['seed'],r['method'],r['step'])==(case,seed,method,800)]
                    ax.plot([r['shell'] for r in rs],[r[quantity] for r in rs],label=method+'/'+str(seed))
            if quantity=='E': ax.set_yscale('log')
            ref=[r for r in spectra if (r['case'],r['method'],r['step'])==(case,'REFERENCE',800)]
            if ref: ax.plot([r['shell'] for r in ref],[r[quantity] for r in ref],color='black',ls='--',lw=1.7,label='reference')
            ax.set_title(f'{case%1000} t=5: {quantity}'); ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.suptitle(scope+'Only completed t=5 endpoint spectra; absent curves = NOT_RUN'); fig.tight_layout()
    fig.savefig(out/'SPECTRA_T5.pdf'); plt.close(fig)
