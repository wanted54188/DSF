"""Chinese evidence report from the bounded audit artifacts, zero model calls."""
from pathlib import Path
import csv,json
import numpy as np
from analyze import HERE,ROOT,read,dump,sha,csv as writecsv

def main():
    out=HERE/'results/audit01'
    R=lambda name:list(csv.DictReader((out/name).open(encoding='utf-8-sig')))
    pairs=R('HOLD_DIRECT_PAIRED.csv');absolute=R('WINDOW_ABSOLUTES.csv');cost=R('HOLD_MEASURED_COST.csv')
    cf=R('SAME_STATE_HEAD_INTERVENTIONS.csv');hd=R('HEAD_FULL_WINDOW.csv');temporal=R('HEAD_TEMPORAL_DIAGNOSTIC.csv')
    ref=list(csv.DictReader((HERE/'results/reference01/EXISTING_REFERENCE_RANKING.csv').open(encoding='utf-8-sig')))
    aidx={(r['case'],r['seed'],r['method'],r['band'],r['window']):r for r in absolute}
    idx={(r['case'],r['seed'],r['state_path'],r['time_s'],r['variant'],r['band']):r for r in cf}
    derived=[];attribution=[]
    for band in ('all','mid','high'):
        for q in ('Source_power','Sb_power','Z_rate_b','K_rate','signed_injection_mean'):
            dif=[];rel=[]
            for k,h in idx.items():
                if k[-2:]!=('hold',band):continue
                t=idx[k[:-2]+('trend',band)];delta=float(h[q])-float(t[q]);dif.append(delta)
                # Ratios only for nonnegative powers/injection. Signed-rate
                # percentages close to zero are not an interpretable diagnostic.
                if q in ('Source_power','Sb_power','signed_injection_mean') and float(t[q]):rel.append(100*delta/float(t[q]))
            derived.append(dict(band=band,quantity=q,samples=len(dif),mean_Hold_minus_PC=float(np.mean(dif)),
                negative_count=sum(v<0 for v in dif),positive_count=sum(v>0 for v in dif),
                mean_pct_for_positive_power_only=float(np.mean(rel)) if rel else None))
            for ch in range(7):
                dif=[];total=[]
                for k,h in idx.items():
                    if k[-2:]!=('hold',band):continue
                    v=idx[k[:-2]+(f'hold_channel_{ch}_trend',band)];t=idx[k[:-2]+('trend',band)]
                    dif.append(float(v[q])-float(h[q]));total.append(float(t[q])-float(h[q]))
                attribution.append(dict(band=band,quantity=q,channel=ch,mean_single_channel_change=float(np.mean(dif)),
                    mean_full_PC_minus_Hold=float(np.mean(total)),
                    ratio_of_mean_changes=float(np.mean(dif)/np.mean(total)) if np.mean(total) else None,
                    caveat='NONADDITIVE one-channel interventions; fractions may exceed1 or cancel; not a Shapley decomposition'))
    writecsv(out/'CONTROLLED_EFFECT_SUMMARY.csv',derived);writecsv(out/'CHANNEL_EFFECT_SUMMARY.csv',attribution)
    lines=['# Hold与PC：独立复算和机制核验','',
        '**判断：CONDITIONAL。Hold是当前开发数据上的更强候选；论文级独立泛化和参考收敛仍未完成。**','',
        'REFERENCE_NOT_CONVERGED。101是训练流动，102与104均为已用于开发判断的流动。此次没有训练、CNN forward、新RHS或新积分。',
        '一、直接从绝对MSE与保存状态核对','',
        '原统一全窗为80个时刻1.05至5的绝对涡量MSE算术均值。独立FFT重算全部24条轨迹的81帧并与原始逐帧绝对MSE核对，不由已报告百分比换算。保留MSE和相对L2的区别。',
        'WINDOW_ABSOLUTES.csv另提供包含t=1的梯形时间平均；不同汇总定义不混用。','',
        '|流动/seed|Hold全窗绝对MSE|Fresh全窗绝对MSE|MSE改善|t5 MSE改善|对PC全窗MSE改善|',
        '|---|---:|---:|---:|---:|---:|']
    six=[r for r in pairs if (r['baseline'],r['band'],r['window'])==('Fresh-C4','all','full')]
    for r in six:
        get=lambda b,w:next(x for x in pairs if (x['case'],x['seed'],x['baseline'],x['band'],x['window'])==(r['case'],r['seed'],b,'all',w))
        lines.append(f'|{int(r["case"])%1000}/{r["seed"]}|{float(r["Hold_MSE"]):.8f}|{float(r["baseline_MSE"]):.8f}|{float(r["MSE_improvement_pct"]):.4f}%|{float(get("Fresh-C4","t5")["MSE_improvement_pct"]):.4f}%|{float(get("Parameter-PC","full")["MSE_improvement_pct"]):.4f}%|')
    lines+=['','六组全窗与终点均正收益。下表所有比较仍以Fresh为基线。K/Z列为平均绝对相对偏差，单位%；速度列才是相对L2指标的改善百分比，不是速度MSE。',
        '', '|流动/seed|u相对L2改善|v相对L2改善|K偏差Hold/Fresh|Z偏差Hold/Fresh|高频MSE改善|平均高频Z/参考Hold/Fresh|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in six:
        key=r['case'],r['seed'];h=aidx[(*key,'Parameter-Hold','all','full')];f=aidx[(*key,'Fresh-C4','all','full')]
        ht=aidx[(*key,'Parameter-Hold','high','full')];ft=aidx[(*key,'Fresh-C4','high','full')]
        hi=next(x for x in pairs if (x['case'],x['seed'],x['baseline'],x['band'],x['window'])==(*key,'Fresh-C4','high','full'))
        lines.append(f'|{int(key[0])%1000}/{key[1]}|{float(r["u_relative_L2_improvement_pct"]):.3f}%|{float(r["v_relative_L2_improvement_pct"]):.3f}%|{100*float(h["abs_K_bias"]):.4f}/{100*float(f["abs_K_bias"]):.4f}|{100*float(h["abs_Z_bias"]):.4f}/{100*float(f["abs_Z_bias"]):.4f}|{float(hi["MSE_improvement_pct"]):.3f}%|{float(ht["Z"])/float(ht["ref_Z"]):.3f}/{float(ft["Z"])/float(ft["ref_Z"]):.3f}|')
    lines+=['','Hold降低了谱尾误差与积累，但高频Z仍约参考的2.00至2.30倍；不能把相对改善写成谱尾问题已经解决。矩形角部单独列在HOLD_DIRECT_PAIRED.csv，不与径向频带重复相加。',
        '', '二、Hold为什么可能优于PC','',
        'PC对Fresh head的预测明显更准：全窗七个通道各自的均方误差都较Hold低，不能解释为PC“没有预测到趋势”。该统计来自各自轨迹；同状态对照见下面独立干预。',
        '通道0为耗散c，1/2/3为Cholesky，4为signed gate，5/6为学习方向的两个raw分量。不能跨通道按raw数值大小判断物理作用。',
        '', '|head通道|PC对Fresh平均MSE|Hold对Fresh平均MSE|', '|---|---:|---:|']
    for ch in range(7):
        avg=lambda method:np.mean([float(r['pooled_MSE']) for r in hd if r['method']==method and int(r['channel'])==ch])
        lines.append(f'|{ch}|{avg("Parameter-PC"):.6e}|{avg("Parameter-Hold"):.6e}|')
    lines+=['','离线固定36个保存状态：六组×PC/Hold两条自身轨迹×t=2、3、5。使用同一对真实head锚点，当前梯度/C4/support不变，分别解码Hold、PC、fresh，以及逐通道替换。共1440次纯代数原decoder，0 CNN、0 RHS、0积分。全部混合候选物理包络通过；只用于诊断，不作为新部署方法。',
        '', '|同状态量|Hold−PC均值|36样本负/正|', '|---|---:|---:|']
    for row in derived:
        if row['band'] in ('all','high'):
            lines.append(f'|{row["band"]}/{row["quantity"]}|{row["mean_Hold_minus_PC"]:+.7e}|{row["negative_count"]}/{row["positive_count"]}|')
    lines+=['','关键证据：全部36个样本中，Hold的全频/高频Source功率及高频Sb功率均小于PC；高频signed enstrophy作用也全部更负。平均高频Source功率减少约1.79%。局部signed负pi注入量35/36降低。这里pi_b=-qb·grad(omega)，负值才是局部回输；不能把signed分支整体都称作backscatter。',
        '全频SGS能量作用Hold−PC反而在36/36为正，不能用“Hold全面增加耗散”概括。分带传递与全局PSD不是同一命题。自主态平均预算同样显示Hold的中高频Sb作用更负，但不同轨迹上的对比只属统计证据。',
        '逐通道同状态干预显示：高频Source功率变化，单独替换学习方向通道5/6分别约对应完整PC−Hold平均变化的34%/42%；gate约13%，qd四通道作用较小。对高频signed Z作用，方向5/6单独替换约52%/41%，gate约7%。这些是非可加干预比，不能相加成严格百分比分解；原C4方向在相同状态上固定，只有学习head被替换。',
        '全频Source功率则gate作用较大；全频Z上gate与学习方向的效果有抵消。不能把所有频带、所有预算归因于同一头。',
        '', '可检验机制：Hold保留了一种本构时间滞后；PC更准确地重建了Fresh，但也更完整地恢复其较强的高频Source响应。Hold可能通过减弱某些方向反馈改善开发集轨迹，而非提高瞬时Fresh逼近精度。原数学形式及严格证据边界见MECHANISM_CN.md。尚未在固定完整路径上单独干预头并检验长期因果，因此标为候选机制。没有宣称非正规不稳定、最大奇异值或NS物理记忆律。',
        '', '三、参考质量、独立初态和全窗时间步','',
        '对101/102已有1024、1536及1536半步三套参考重新评分，Hold对Fresh的全频全窗排名全部保持。原参考时间步两档之间改善百分比的漂移很小；空间1024/1536之间仍有差异。详见../reference01/EXISTING_REFERENCE_RANKING.csv。',
        '但是101/102旧门槛filtered_time_space、native_time_space、candidate_tail仍FAIL；算子恒等式和CFL通过不构成参考收敛。104仅单套参考，原生谱尾最大功率比例0.021862858，仍远超原1e-4门槛。因此所有研究结论继续REFERENCE_NOT_CONVERGED。',
        'IC105初始Fourier系数已冻结，CPU100/128一致性通过；用户已确认此前未用于训练、验证或模型选择，来源收据另存independent_ic105/PROVENANCE_ATTESTATION.json，不修改原FROZEN.json。它是待执行的独立同族初态，目前DNS/LES尚未生成，不能宣称已经获得盲测性能证据，也不等于跨物理参数泛化。',
        '已提供全窗dt=.0025手动驱动paper_dt.py：保持原.005历史单位与刷新物理时刻，两个细步组成每个粗时间单元，四方法均自主运行至5。CPU四方法、8个stage、固定刷新、原驱动中断续算逐位一致均PASS。真实全窗NOT_RUN，已有72个局部半步检查不能代替它。',
        '进一步参考3072空间及时间细化仅完成预登记PAPER_PROTOCOL.json，尚未实现/执行高成本新参考生成与资格驱动。最多新增2080729个高分辨率RHS，此外还有原生场保存/IO；未给出虚构wall时间。不能说“论文级验证已经完成”。',
        '', '四、实测成本','', '|流动/seed|Hold/Fresh CNN|Hold wall秒|Fresh wall秒|实测加速|', '|---|---:|---:|---:|---:|']
    for r in cost:lines.append(f'|{int(r["case"])%1000}/{r["seed"]}|{r["Hold_CNN"]}/{r["Fresh_CNN"]}|{float(r["Hold_loop_seconds"]):.2f}|{float(r["Fresh_loop_seconds"]):.2f}|{float(r["Hold_measured_speedup"]):.3f}x|')
    lines+=['','Hold与PC CNN数完全相同。相对Fresh节省90.022%的模型特异CNN，但实测轨迹循环加速为1.349至2.895倍。包括刷新/检查/保存，公共12 CNN/4 RHS启动另记；不能把90%调用减少写成10倍实测加速。',
        '这些是单次既有运行的实测值；硬件负载/执行顺序/I/O差异明显，尚无重复计时置信区间。完整命令耗时也包含审计、报告和共享开销，不可按方法重复分摊。',
        '', '五、本轮准确状态','',
        '- VERIFIED：绝对MSE双路径复算、速度/KZ/谱尾与成本核对、36状态同状态head干预、已有参考重评分。',
        '- FAIL：旧参考资格中的空间/原生谱尾门槛；均值signed域的普遍保证仍不存在，既有实际24轨迹检查通过不改写该定理边界。',
        '- NOT_RUN：独立IC105的参考/模型轨迹、3072参考资格、全窗口dt细化和重复性能基准。',
        '- 研究判断：Hold值得继续做固定的独立验证；当前不能升级为已完成论文级泛化/稳定性证明。不调整权重、head、刷新频率或损失。']
    (out/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in ('Parameter-PC','Parameter-Hold'):
        vals=[np.mean([float(r['pooled_MSE']) for r in hd if r['method']==method and int(r['channel'])==ch]) for ch in range(7)]
        axes[0].semilogy(range(7),vals,'o-',label=method)
    axes[0].set(title='Raw-head error to fresh on own paths',xlabel='head channel',ylabel='MSE');axes[0].legend()
    vals=[next(r['mean_single_channel_change'] for r in attribution if (r['band'],r['quantity'],r['channel'])==('high','Z_rate_b',ch)) for ch in range(7)]
    axes[1].bar(range(7),vals);axes[1].set(title='Same-state one-channel hold -> trend',xlabel='replaced channel',ylabel='change in high-band signed Z rate')
    fig.suptitle('36 archived states; nonadditive interventions; REFERENCE_NOT_CONVERGED');fig.tight_layout()
    fig.savefig(out/'HEAD_MECHANISM.pdf');fig.savefig(out/'HEAD_MECHANISM.png',dpi=150);plt.close(fig)
    dump(out/'SUMMARY_SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='SUMMARY_SHA256.json'})
    print('REPORT:',out/'REPORT_CN.md')

if __name__=='__main__':main()
