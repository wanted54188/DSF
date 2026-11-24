"""Actual-data plots and verdict generation; blocked results remain blocked."""
from pathlib import Path
import numpy as np
import common as c

def blocked(run,reason):
    qualities=[]
    for s in c.SEEDS:
        path=run/'quality'/str(s)/'QUALITY.json'
        qualities.append(f'- {s}: '+(c.read(path)['status'] if path.exists() else 'NOT RUN'))
    report='# SSC_LONG_HORIZON_VERDICT_CN\n\n状态：INCONCLUSIVE_BLOCKED_REFERENCE_QUALITY / NOT RUN。\n\n'+reason+'\n\n'+'\n'.join(qualities)
    report+='\n\n没有生成可信长期参考与真实误差前，不回答Clark长期误差增长、SSC优势或JCP成功。已有1.2–5秒有限性结果不能替代匹配精度。所有锁定和最终测试仍封存。\n'
    (run/'SSC_LONG_HORIZON_VERDICT_CN.md').write_text(report,encoding='utf-8')

def render(run,summaries,rows,paths,failures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=run/'figures';figdir.mkdir(exist_ok=True);e=c.engine()
    names={'SSC_0.5_24':'SSC-Fixed','SSC_1_16':'SSC-Optimized'};label=lambda m:names.get(m,m)
    colors=dict(zip(c.METHODS,['#777777','#009e73','#cc79a7','#0072b2','#000000','#d55e00','#56b4e9']))
    main=('FULL','Clark','SSC_0.5_24','SSC_1_16')
    for key,title in [('omega_relative_L2_all','Vorticity relative L2'),('omega_relative_L2_low','Low-band relative L2'),('omega_relative_L2_high','High-band relative L2'),('E_relative_bias','Energy relative bias'),('Z_relative_bias','Enstrophy relative bias'),('radial_tail_fraction','Radial spectral tail')]:
        fig,ax=plt.subplots(2,2,figsize=(11,7))
        for axis,(s,w) in zip(ax.flat,[(s,w) for s in c.SEEDS for w in (2026,2028)]):
            for m in c.METHODS:
                data=paths[s,w,m,.005][2];axis.plot([r['time_s'] for r in data],[r[key] for r in data],label=label(m),color=colors[m])
            axis.set(title=f'IC {s}, weight {w}',xlabel='time [s]',ylabel=title);axis.grid(alpha=.2)
            if key.startswith('omega_'):axis.set_yscale('symlog',linthresh=.01)
            elif key=='radial_tail_fraction':axis.set_yscale('symlog',linthresh=1e-12)
        h,l=ax[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=4);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(figdir/(key+'.png'),dpi=150);plt.close(fig)
    # Same exact timestamps, blank cell for a failed/missing state, never interpolation.
    for seed in c.SEEDS:
        for t in c.policy()['field_plot_times']:
            fig,ax=plt.subplots(2,len(main)+1,figsize=(15,6))
            _,_,_,op,a=paths[seed,2026,'FULL',.005];ref=a['reference_omega'][c.index(a['time_s'],t)]
            vmax=float(abs(ref).max());ax[0,0].imshow(ref,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax);ax[0,0].set_title('Reference');ax[1,0].axis('off')
            for col,m in enumerate(main,1):
                times,states,_,_,_=paths[seed,2026,m,.005];hit=np.where(abs(times-t)<1e-9)[0]
                if not len(hit):
                    for ar in ax[:,col]:ar.text(.5,.5,'No valid state',ha='center');ar.axis('off')
                    continue
                w=states[hit[0]];ax[0,col].imshow(w,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax);ax[0,col].set_title(label(m))
                image=ax[1,col].imshow(w-ref,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
                ax[1,col].set_title('Error: same scale as reference')
            fig.suptitle(f'IC {seed}, weight2026, t={t:g}; shared physical color scale');fig.tight_layout();fig.savefig(figdir/f'FIELDS_{seed}_{t:g}.png',dpi=150);plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(13,4))
            for m in main:
                times,states,_,_,_=paths[seed,2026,m,.005];hit=np.where(abs(times-t)<1e-9)[0]
                if not len(hit):continue
                w=states[hit[0]];err=op.shell_power(w-ref);Z=.5*op.shell_power(w);E=.5*op.shell_cospectrum(w,op.psi(w))
                for axis,power in zip(axes,(err,E,Z)):axis.semilogy(np.maximum(power,1e-30),label=label(m),color=colors[m])
            axes[1].semilogy(np.maximum(.5*op.shell_cospectrum(ref,op.psi(ref)),1e-30),'k--',label='Reference')
            axes[2].semilogy(np.maximum(.5*op.shell_power(ref),1e-30),'k--',label='Reference')
            for axis,title in zip(axes,('Vorticity error power','Energy spectrum','Enstrophy spectrum')):axis.set(title=title,xlabel='shell');axis.legend(fontsize=7);axis.grid(alpha=.2)
            fig.tight_layout();fig.savefig(figdir/f'SPECTRA_{seed}_{t:g}.png',dpi=150);plt.close(fig)
    # Aggregate only complete independent cases; failed full-window scores remain blank.
    aggregate=[]
    for dt in c.policy()['les_dt']:
        for m in c.METHODS:
            rr=[r for r in summaries if r['method']==m and r['dt']==dt];complete=all(r['accuracy_window_complete'] for r in rr)
            out=dict(method=m,dt=dt,groups_completed=sum(r['accuracy_window_complete'] for r in rr),groups_total=len(rr),independent_flows=2)
            for key in ('omega_relative_L2_all','omega_relative_L2_low','omega_relative_L2_high','E_relative_bias','Z_relative_bias'):
                for suffix in ('full_window','terminal5'):out[key+'_'+suffix]=float(np.mean([r[key+'_'+suffix] for r in rr])) if complete else None
            aggregate.append(out)
    c.csvout(run/'AGGREGATE.csv',aggregate)
    def value(m,k):return next(r[k] for r in aggregate if r['method']==m and r['dt']==.005)
    def percent(v):return '—（未完成）' if v is None else f'{100*v:.5f}%'
    # Predeclared paired decision, with observed numerical comparisons as diagnostics.
    paired=[];refdelta=0.
    for s in c.SEEDS:
        conv=c.csvread(run/'quality'/str(s)/'CONVERGENCE.csv')
        refdelta=max(refdelta,max(float(x['max_frame_relative_RMS']) for x in conv if x['field']=='reference_omega' and x['band']=='all'))
    for dt in c.policy()['les_dt']:
        for s in c.SEEDS:
            for w in c.policy()['weights']:
                def get(m,h):return next(r for r in summaries if r['seed']==s and r['weight']==w and r['method']==m and r['dt']==h)
                ss=get('SSC_1_16',dt)
                for base in ('Clark','FULL'):
                    bb=get(base,dt)
                    for suffix in ('full_window','terminal5'):
                        key='omega_relative_L2_all_'+suffix;gain=None if ss[key] is None or bb[key] is None else bb[key]-ss[key]
                        sa,sb=get('SSC_1_16',.005),get('SSC_1_16',.0025);ba,bb2=get(base,.005),get(base,.0025)
                        drift=None if any(x[key] is None for x in (sa,sb,ba,bb2)) else abs(sa[key]-sb[key])+abs(ba[key]-bb2[key])
                        paired.append(dict(seed=s,weight=w,dt=dt,baseline=base,metric=suffix,absolute_error_reduction=gain,
                            observed_numerical_margin=None if drift is None else 2*refdelta+drift,
                            gain_positive=gain is not None and gain>0,beyond_observed_margin=gain is not None and drift is not None and gain>2*refdelta+drift))
    c.csvout(run/'PAIRED_ADVANTAGE.csv',paired)
    if not all(r['accuracy_window_complete'] for r in summaries if r['method']=='SSC_1_16'):decision='NO_SUPPORT_LONG_ROBUSTNESS'
    elif not all(r['gain_positive'] for r in paired):decision='NO_SUPPORT_CONSISTENT_SUPERIORITY_TO_CLARK_AND_FULL'
    elif not all(r['beyond_observed_margin'] for r in paired):decision='INCONCLUSIVE_OBSERVED_NUMERICAL_MARGIN'
    else:decision='DEVELOPMENT_ACCURACY_ADVANTAGE_NOT_INDEPENDENT_VALIDATION'
    c.dump(run/'DECISION.json',dict(status=decision,reference_difference_observed=refdelta,
        caveat='Observed reference differences are NOT certified continuum bounds; JCP novelty is not an automatic numerical gate',policy=c.policy()['verdict_rule']))
    lines=['# SSC_LONG_HORIZON_VERDICT_CN','', '状态：COMPLETED_LONG_MATCHED_EVALUATION。仅两个development独立初值；四个初值/权重组合不是四个独立流动。', '',
        '## 1. 原参考与质量', '',
        '同一原独立NS、原始谱/种子、域长/黏性/滤宽；优先从原complex128 Fourier checkpoint 1.2秒继续。原1–1.2秒前缀按SHA复用并由原高分辨快照和checkpoint独立重算核验。source主标签为过滤高网格RHS−同态低网格RHS，另算通量散度；两项一致性不替代数值收敛。', '']
    for s in c.SEEDS:
        q=c.read(run/'quality'/str(s)/'QUALITY.json');lines.append(f'- {s}: {q["status"]}；gates={q["gates"]}。')
    lines+=['', '空间1024主步长/1536主步长，时间1536主/半步长：全部密集低场各频带与原始native快照全模差分别核验，另有candidate raw尾谱/CFL门禁；质量数字和失败项见quality/。这是经验数值收敛证据，不是连续真值/DNS普适认证。', '',
        '## 2–4. 长期涡量与E/Z实际比较', '', '|方法|5秒涡量误差|全窗口平均|5秒E绝对相对偏差|5秒Z绝对相对偏差|', '|---|---:|---:|---:|---:|']
    for m in c.METHODS:lines.append('|'+label(m)+'|'+'|'.join(percent(value(m,k)) for k in ('omega_relative_L2_all_terminal5','omega_relative_L2_all_full_window','E_relative_bias_terminal5','Z_relative_bias_terminal5'))+'|')
    sc=value('SSC_1_16','omega_relative_L2_all_full_window');cl=value('Clark','omega_relative_L2_all_full_window');full=value('FULL','omega_relative_L2_all_full_window')
    lines+=['', f'SSC-Optimized相对Clark的全窗口误差收益：{100*(cl-sc)/cl:.4f}%；相对FULL：{100*(full-sc)/full:.4f}%。' if all(x is not None and x>0 for x in (sc,cl,full)) else '至少一个主要方法未完成，不能给出完整5秒优势。', '',
        'Clark短窗高精度有共同态source误差较低的既有证据；是否长期增长须看ERROR_TIMEWISE与SPECTRA，不能仅由解析构型断言因果。长期source诊断另列SOURCE_TIMEWISE，不与不同自主状态混合。', '',
        '## 5. 固定版安全停止与优化版的差别', '',
        'FAILURES.json保留实际失败时刻和安全条件。高频谱尾、误差谱、E/Z及两步长差异可定位失稳伴随信号；它们不能唯一证明半离散闭合不稳定或排除显式RK4刚性。不能由source负交叉项单独推出自主稳定性。没有进行耗散扫描或调整参数。', '',
        '## 6. 有效时长、频带与步长', '',
        'PREDICTION_HORIZONS.csv报告5/10/20%全场relative L2首次越界的观测区间，未越界按右删失报告，不插值出精确秒数。COMMON_INTERVAL.csv只比较全部方法共有的真实时间节点；完整5秒评分对失败方法留空，不用其较短有效前缀冒充完整窗口。DT_COMPARISON.csv为同一空间网格的时间步敏感性。分带误差采用1–16、17–71壳并同时输出可加的误差平方。', '',
        '## 7. JCP研究判断与边界', '',f'预先固定配对规则的实际结果：**{decision}**。数值差异参照量为实测参考差异与时间步效应；不是连续真值误差界。逐条件见PAIRED_ADVANTAGE.csv。', '',
        '这些是开发数据上的冻结参数评价，没有新独立泛化验证。若SSC的长期误差、E/Z和稳定性不能在可信数值差异之外优于Clark与FULL，就不支持仅靠固定谱混合作为JCP独立主方法。即便开发结果有优势，也不能自动认定创新性或解锁验证。现阶段可据本表和逐初值配对结果决定是否继续；不以挑选窗口、均值掩盖失败或调参挽救结论。', '',
        f'实际数值失败/安全停止条件行数：{len(failures)}。两权重为模型重复，Clark的权重列是同一解析轨迹别名。未训练、未访问locked/reserved或STAR最终场。', '',
        '## 复现与成本', '',
        'FREEZE.json、METADATA_AUDIT.json、reference/*/DONE/PROGRESS/CHECKPOINT、quality/*.csv、TRAJECTORY_REUSE.json、SOURCE_COST.json、RHS_COST_PROVENANCE.json、EXECUTION.log给出输入SHA、协议、每任务成本和CPU公平计时口径。旧5秒轨迹优先复用；不会把本批后处理成本当作新LES推进成本。']
    lines+=['', '## 附：七问的附加定量证据', '']
    for s in c.SEEDS:
        for w in (2026,2028):
            clrows=paths[s,w,'Clark',.005][2]
            early=next(r['omega_relative_L2_all'] for r in clrows if abs(r['time_s']-1.2)<1e-9);last=clrows[-1]['omega_relative_L2_all']
            lines.append(f'- IC{s}/权重{w} Clark：1.2秒误差{100*early:.5f}%，最后有效时刻{clrows[-1]["time_s"]:g}秒误差{100*last:.5f}%，比值{last/max(early,1e-30):.4g}（比值本身不代表模型失效）。')
    for base in ('FULL','Clark'):
        for stat in ('E_relative_bias','Z_relative_bias'):
            a=value('SSC_1_16',stat+'_full_window');b=value(base,stat+'_full_window')
            lines.append(f'- SSC优化版相对{base}的全窗口{stat}改善：'+('不可计算（有失败条件）' if a is None or b is None else f'{100*(b-a)/max(b,1e-30):.4f}%')+'。')
    for m in ('SSC_0.5_24','SSC_1_16'):
        selected=[r for r in summaries if r['method']==m]
        lines.append(f'- {label(m)}：完成条件{sum(r["accuracy_window_complete"] for r in selected)}/{len(selected)}；失败时刻'+str([r['failure_time'] for r in selected if r['failure_time'] is not None])+'；频带末值/窗口值及有效时长均在CSV逐条件保留。')
    lines+=['', '固定版在|k|=33处Clark权重约0.486、仍保留约0.514的Sb；优化版在该处Clark权重接近1。这个构成差异可以精确确认，但不能仅凭两组结果将安全停止唯一归因于残留Sb，或排除其与显式积分的耦合。', '',
        '### 预测有效时长的实际记录（主步长）', '',
        '|方法|阈值|首次越界时刻范围(s)|未越界右删失条件数|数值失败删失条件数|', '|---|---:|---:|---:|---:|']
    hh=c.csvread(run/'PREDICTION_HORIZONS.csv')
    for m in c.METHODS:
        for threshold in c.policy()['prediction_error_thresholds']:
            rr=[r for r in hh if r['method']==m and float(r['dt'])==.005 and float(r['threshold'])==threshold]
            times=[float(r['first_exceedance']) for r in rr if r['first_exceedance']]
            span=f'{min(times):.4f}–{max(times):.4f}' if times else '未观测到'
            lines.append(f'|{label(m)}|{threshold:.0%}|{span}|{sum(r["censor"]=="RIGHT_CENSORED_AT_END" for r in rr)}|{sum(r["censor"]=="CENSORED_BY_NUMERICAL_FAILURE" for r in rr)}|')
    lines+=['', '范围包含两个开发初值×两个权重；Clark别名不能增加独立样本数。准确的前后观测界、相对于1秒初始时刻的有效时长下/上界在CSV中。', '',
        '### 参考态source全窗口误差（单独口径）', '', '|方法|全频relative L2时间均值|低频|高频|', '|---|---:|---:|---:|']
    src=c.csvread(run/'SOURCE_TIMEWISE.csv')
    for m in c.METHODS:
        means=[]
        for b in ('all','low','high'):
            group=[]
            for s in c.SEEDS:
                for w in c.policy()['weights']:
                    rr=sorted([r for r in src if r['method']==m and r['band']==b and int(r['seed'])==s and int(r['weight'])==w],key=lambda r:float(r['time_s']))
                    group.append(float(np.trapezoid([float(r['relative_L2']) for r in rr],[float(r['time_s']) for r in rr])/4))
            means.append(float(np.mean(group)))
        lines.append('|'+label(m)+'|'+'|'.join(f'{x:.6g}' for x in means)+'|')
    (run/'SSC_LONG_HORIZON_VERDICT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.dump(run/'REPORT_IDENTITY.json',dict(report_code_sha256=c.sha(Path(__file__)),generated_time=__import__('time').strftime('%Y-%m-%d %H:%M:%S')))
