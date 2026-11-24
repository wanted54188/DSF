"""One complete report and real-data figures, with explicit missing stages."""
import csv
from collections import defaultdict
import numpy as np
import common as c
import rollout


def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def generate(a,out):
    reports=out/'reports';reports.mkdir(exist_ok=True);i=1
    while (reports/f'report_{i:03d}').exists():i+=1
    d=reports/f'report_{i:03d}';d.mkdir();training=[];histories=[]
    for row in c.matrix(a):
        folder=out/'training'/f'{row["mode"]}_{row["seed"]}'
        status=c.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else dict(status='NOT_RUN')
        done=c.read(folder/'DONE.json') if (folder/'DONE.json').exists() else {}
        training.append(dict(mode=row['mode'],seed=row['seed'],**status,checkpoint_sha256=done.get('best_sha256')))
        if (folder/'HISTORY.json').exists():
            for h in c.read(folder/'HISTORY.json'):
                histories.append(dict(mode=row['mode'],seed=row['seed'],epoch=h['epoch'],
                    train_source_MSE=h['training']['source'],validation_source_MSE=h['validation']['source'] if h['validation'] else None,
                    validation_source_L2=h['validation']['source_relative_L2'] if h['validation'] else None))
    c.csvout(d/'TRAINING_SUMMARY.csv',training);c.csvout(d/'TRAINING_HISTORY.csv',histories)
    apriori=[];file=out/'apriori/SOURCE.csv'
    if file.exists():
        done=c.read(out/'apriori/DONE.json')
        if c.sha(file)!=done['outputs']['SOURCE.csv']:raise ValueError('A-priori CSV modified')
        grouped=defaultdict(list)
        for row in csvread(file):grouped[(row['initial'],row['training_seed'],row['method'],row['band'])].append(row)
        for keys,rr in sorted(grouped.items()):
            z=dict(zip(('initial','training_seed','method','band'),keys));z['frames']=len(rr)
            for k in ('source_relative_L2','MSE','correlation','SGS_E','SGS_Z','reference_E','reference_Z'):
                z[k]=float(np.mean([float(r[k]) for r in rr]))
            for q in ('E','Z'):z[q+'_budget_MAE']=float(np.mean([abs(float(r['SGS_'+q])-float(r['reference_'+q])) for r in rr]))
            apriori.append(z)
    c.csvout(d/'SOURCE_SUMMARY.csv',apriori)
    state=[];curves=[];missing=[]
    for stage in ('short','long'):
        for row in rollout.matrix(a,stage):
            base=dict(row)
            if row['analytic_alias']:base['training_seed']=2026
            dest=out/stage/rollout.name(base)
            if not (dest/'DONE.json').exists():missing.append(row);continue
            done=c.read(dest/'DONE.json')
            for name in ('SUMMARY.json','ERROR_TIMEWISE.csv'):
                if c.sha(dest/name)!=done['outputs'][name]:raise ValueError('Rollout result changed')
            for r in c.read(dest/'SUMMARY.json'):
                state.append({**r,'training_seed':row['training_seed'],'analytic_alias':row['analytic_alias']})
            if not row['analytic_alias']:curves+=csvread(dest/'ERROR_TIMEWISE.csv')
    c.csvout(d/'ROLLOUT_SUMMARY.csv',state);c.csvout(d/'NOT_RUN_MATRIX.csv',missing)
    complete=sum(x['status']=='TRAINING_COMPLETE' for x in training)
    lines=['# 新NS统一训练与评估基准','', '**REFERENCE_NOT_CONVERGED：使用已有离散参考的探索性基准，不是DNS收敛认证。**','',
        '固定101训练、102开发验证；1–5 s。原历史development身份保留，新训练用途由用户2026-11-03明确授权。',
        '101属于训练轨迹；102曾参与方法开发，本轮作held-out开发验证，不能称最终独立测试。',
        '',f'训练完成 {complete}/8；source评价 '+('COMPLETED' if apriori else 'NOT_RUN')+'。',
        '', '|方法|种子|状态|已训练epoch|选中epoch|','|---|---:|---|---:|---:|']
    for r in training:lines.append(f'|{r["mode"]}|{r["seed"]}|{r["status"]}|{r.get("epoch","—")}|{r.get("best_epoch","—")}|')
    if apriori:
        lines+=['','开发验证102，两个训练种子的描述性平均：','','|方法|source全频L2|低频L2|高频L2|','|---|---:|---:|---:|']
        for method in ('FULL-C2','FULL-Prior4','FULL-Input4','FULL-C4','Clark4','P1','P1-C4','NoSGS'):
            vals=[]
            for band in ('all','low','high'):
                rr=[x['source_relative_L2'] for x in apriori if int(x['initial'])==110240102 and x['method']==method and x['band']==band]
                vals.append(f'{100*np.mean(rr):.4f}%' if rr else 'NOT_RUN')
            lines.append('|'+method+'|'+'|'.join(vals)+'|')
    lines+=['','|阶段|实际完成独立任务|计划独立任务|','|---|---:|---:|']
    for stage in ('short','long'):
        actual=sum((out/stage/rollout.name(r)/'DONE.json').exists() for r in rollout.matrix(a,stage) if not r['analytic_alias'])
        lines.append(f'|{stage}|{actual}|56|')
    lines+=['','全部同数据重训练完成并冻结BASELINE_REGISTRY之后，才运行source/short/long。原PoF及旧冻结迁移结果不进入这次公平训练主表。',
        'P1/P1-C4只使用新训练FULL-C2作神经基底；Clark4与NoSGS无参数可训练。',
        '逐seed、逐初值、逐dt、逐参考的原始指标全部保留；种子/时间帧/参考版本不增加独立流动数。',
        '失败方案的完整窗口指标留空；安全停止不等于一般物理失稳定理。',
        '如果source改善而rollout恶化，应报告SOURCE_GAIN_ROLLOUT_LOSS；本程序不据单个指标自动授予科学GO。']
    (d/'NS_BASELINE_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.dump(d/'STATUS.json',dict(training_completed=complete,total_training=8,apriori='COMPLETED' if apriori else 'NOT_RUN',
        reference_quality='REFERENCE_NOT_CONVERGED',independent_final_test='NOT_RUN',scientific_success='NOT_ESTABLISHED'))
    if histories or apriori or curves:plot(d,histories,apriori,curves)
    return d


def plot(d,histories,source,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if histories:
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        for mode in c.MODES:
            for seed in (2026,2028):
                rr=[r for r in histories if r['mode']==mode and r['seed']==seed]
                if not rr:continue
                ax[0].plot([r['epoch'] for r in rr],[r['train_source_MSE'] for r in rr],label=f'{mode}/{seed}')
                vv=[r for r in rr if r['validation_source_MSE'] is not None]
                ax[1].plot([r['epoch'] for r in vv],[r['validation_source_MSE'] for r in vv],label=f'{mode}/{seed}')
        for z,title in zip(ax,('101 train source MSE','102 development validation source MSE')):z.set_title(title);z.set_yscale('log');z.grid(alpha=.2)
        ax[1].legend(fontsize=7);fig.suptitle('REFERENCE_NOT_CONVERGED');fig.tight_layout();fig.savefig(d/'TRAINING_CURVES.png',dpi=160);plt.close(fig)
    if source:
        methods=('FULL-C2','FULL-Prior4','FULL-Input4','FULL-C4','Clark4','P1','P1-C4','NoSGS')
        fig,ax=plt.subplots(1,3,figsize=(13,4))
        for z,band in zip(ax,('all','low','high')):
            values=[np.mean([r['source_relative_L2'] for r in source if int(r['initial'])==110240102 and r['method']==m and r['band']==band]) for m in methods]
            z.bar(methods,values);z.set_title('102 '+band+' source relative L2');z.tick_params(axis='x',rotation=65)
        fig.suptitle('REFERENCE_NOT_CONVERGED');fig.tight_layout();fig.savefig(d/'SOURCE_COMPARISON.png',dpi=160);plt.close(fig)
    for initial in (110240101,110240102):
        for stage in ('short','long'):
            rr=[r for r in curves if int(r['initial'])==initial and r['stage']==stage and float(r['dt'])==.005 and r['reference']=='n1536_half']
            if not rr:continue
            fig,ax=plt.subplots(2,3,figsize=(13,7));ax=ax.ravel()
            for method,seed in sorted({(x['method'],x['training_seed']) for x in rr}):
                x=sorted([r for r in rr if (r['method'],r['training_seed'])==(method,seed)],key=lambda r:float(r['time_s']))
                for z,key in zip(ax,('omega_relative_L2_all','omega_relative_L2_low','omega_relative_L2_high',
                                    'E_signed_relative_bias','Z_signed_relative_bias','tail_fraction')):
                    z.plot([float(r['time_s']) for r in x],[float(r[key]) for r in x],label=f'{method}/{seed}');z.set_title(key);z.grid(alpha=.2)
            ax[0].legend(fontsize=6);fig.suptitle(f'{initial} {stage} REFERENCE_NOT_CONVERGED');fig.tight_layout();fig.savefig(d/f'{initial}_{stage}_CURVES.png',dpi=160);plt.close(fig)
