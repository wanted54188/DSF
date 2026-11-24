"""One reference report: time/space, source definition bridge, tail and balances."""
import csv
import numpy as np
from rp_common import write, sha


def csvout(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def comparison(op, a, b, axis, quality):
    out = []
    for field in ('reference_omega', 'reference_source', 'reference_dot'):
        if not np.allclose(a['time_s'], b['time_s'], rtol=0, atol=1e-10):
            raise ValueError('Unaligned reference times')
        for band in ('low', 'high', 'all'):
            target = op.band(b[field], band); diff = op.band(a[field]-b[field], band)
            floor = quality['normalization_floor']*rms(b[field])
            den = max(rms(target), floor, 1e-30)
            relative = rms(diff)/den
            details = []
            for t, x, y, full in zip(b['time_s'], diff, target, b[field]):
                f = quality['normalization_floor']*rms(full)
                d = max(rms(y), f, 1e-30)
                details.append((rms(x)/d, float(t), rms(x), rms(y), rms(y) < f))
            peak = max(details)
            out.append(dict(axis=axis, field=field, band=band, relative_RMS=relative,
                max_frame_relative_RMS=peak[0], peak_time_s=peak[1],
                peak_difference_RMS=peak[2], peak_reference_RMS=peak[3],
                floor_active_frames=sum(x[4] for x in details),
                threshold=quality['relative_time_space'],
                passes=max(relative, peak[0]) <= quality['relative_time_space']))
    return out


def qualify(run, frozen):
    from engine import operator
    from rp_compute import task_verified
    op = operator(frozen['config']['audit']); nu = frozen['config']['nu']; q = frozen['protocol']['quality']
    target = run/'quality'
    if target.exists():
        raise FileExistsError('Quality output exists; retain it and choose a new explicit output revision')
    data, metadata, identities = {}, {}, []
    for name, item in frozen['reused'].items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Reused reference hash mismatch')
        with np.load(item['path'], allow_pickle=False) as z:
            a = {k: np.array(z[k]) for k in ('time_s', 'reference_omega', 'reference_source', 'reference_dot')}
        for field in ('reference_omega', 'reference_source', 'reference_dot'):
            a[field] = op.project(a[field])
        a['legacy_reference_source'] = a['reference_source']
        resolved = np.stack([op.nonlinear(w)+nu*op.lap(w) for w in a['reference_omega']])
        a['reference_source'] = a['reference_dot']-resolved
        # No high-grid moments exist in old archives: this is an algebraic gap, not an independent flux check.
        data[name] = a; metadata[name] = item['metadata']
    for name in frozen['protocol']['new_variants']:
        m = task_verified(run/name); metadata[name] = m['metadata']
        frames = []
        for file in m['metadata']['frames']:
            with np.load(run/name/file, allow_pickle=False) as z:
                frames.append({k: np.array(z[k]) for k in z.files})
        a = {k: np.stack([x[k] for x in frames]) for k in frames[0]}
        if not all(np.isfinite(x).all() for x in a.values()):
            raise ValueError('Nonfinite reference arrays')
        if a['reference_omega'].shape != (41, op.ny, op.nx):
            raise ValueError('Unexpected reference window/grid')
        for i, t in enumerate(a['time_s']):
            src = a['reference_source'][i]
            scale = max(rms(src), 1e-30)
            qx, qy = a['reference_qx'][i], a['reference_qy'][i]
            flux = op.source(np.stack((qx, qy)))
            resolved = op.nonlinear(a['reference_omega'][i])+nu*op.lap(a['reference_omega'][i])
            identities.append(dict(variant=name, time_s=float(t),
                source_flux_relative=rms(flux-src)/scale,
                source_RHS_gap_relative=rms(src-a['reference_source_rhs_gap'][i])/scale,
                source_mean_relative=abs(float(src.mean()))/scale,
                equation_residual_relative=rms(a['reference_dot'][i]-resolved-src)/scale))
        data[name] = a
    convergence = []
    for a, b, axis in (('space_fine', 'both_fine', 'time_512_reused'),
                       ('space_fine', 'n1024_dt', 'space_512_1024_common_dt'),
                       ('n1024_dt', 'n1024_half_dt', 'time_1024')):
        convergence.extend(comparison(op, data[a], data[b], axis, q))
    tails = [dict(variant=name, n=m['n'], max_CFL=m['max_CFL'], max_tail=m['max_tail'],
                  CFL_limit=q['CFL'], tail_limit=q['tail_fraction'],
                  CFL_pass=m['max_CFL'] <= q['CFL'], tail_pass=m['max_tail'] <= q['tail_fraction'],
                  role='candidate_reference' if name.startswith('n1024') else 'coarse_control_v1_failure_preserved')
             for name, m in metadata.items()]
    bridge, balances = [], []
    for name, a in data.items():
        for i, t in enumerate(a['time_s']):
            w = a['reference_omega'][i]
            for band in ('low', 'high', 'all'):
                primary, legacy = op.band(a['reference_source'][i], band), op.band(a['legacy_reference_source'][i], band)
                dot = op.band(a['reference_dot'][i], band)
                gap = primary-legacy
                bridge.append(dict(variant=name, time_s=float(t), band=band,
                    primary_source_RMS=rms(primary), legacy_source_RMS=rms(legacy),
                    definition_difference_RMS=rms(gap),
                    difference_over_primary=rms(gap)/max(rms(primary), 1e-30),
                    legacy_equation_residual_over_RHS=rms(gap)/max(rms(dot), 1e-30)))
            values = op.value(w)
            balances.append(dict(variant=name, time_s=float(t), **values,
                primary_Z_budget=float(np.mean(w*a['reference_source'][i])),
                legacy_Z_budget=float(np.mean(w*a['legacy_reference_source'][i])),
                primary_E_budget=float(np.mean(op.psi(w)*a['reference_source'][i])),
                legacy_E_budget=float(np.mean(op.psi(w)*a['legacy_reference_source'][i]))))
    identities_ok = all(max(x[k] for k in ('source_flux_relative', 'source_RHS_gap_relative',
                       'equation_residual_relative')) <= q['source_flux_identity']
                       and x['source_mean_relative'] <= q['mean_relative'] for x in identities)
    candidate_tail_ok = all(x['tail_pass'] and x['CFL_pass'] for x in tails if x['role'] == 'candidate_reference')
    convergence_ok = all(x['passes'] for x in convergence)
    passed = convergence_ok and identities_ok and candidate_tail_ok
    status = 'PASS_FILTERED_REFERENCE_QUALITY_V2' if passed else 'INCONCLUSIVE_BLOCKED_REFERENCE_QUALITY'
    # The original legacy-source comparison remains a diagnostic; no historical result is replaced.
    legacy_convergence = []
    for left, right, axis in (('space_fine', 'n1024_dt', 'legacy_space_512_1024'),
                              ('n1024_dt', 'n1024_half_dt', 'legacy_time_1024')):
        a, b = dict(data[left]), dict(data[right])
        a['reference_source'] = a['legacy_reference_source']; b['reference_source'] = b['legacy_reference_source']
        legacy_convergence.extend(comparison(op, a, b, axis, q))
    result = dict(status=status, initial_seed=frozen['seed'],
        reference_gates=dict(time_space=convergence_ok, candidate_tail_CFL=candidate_tail_ok,
                             independent_source_flux_and_RHS_identity=identities_ok),
        failed_convergence=[x for x in convergence if not x['passes']],
        source_label_definition=frozen['protocol']['reference_source'],
        original_v1_quality='UNCHANGED_FAIL_REFERENCE_QUALITY',
        H1_H4='NOT_RUN', mechanism='INCONCLUSIVE_NOT_RUN', method='NOT_RUN',
        distribution_coverage='NOT_ESTABLISHED', training='NOT_RUN',
        DNS_certification=False, continuum_error_certification=False,
        uncertainty='Empirical time/space sensitivities, not rigorous error bounds',
        reused_source_caveat='n512 S_eff reconstructed from saved dot and state; no independent n512 high-product flux available',
        threshold_changes='None of the numerical thresholds changed. V2 uses the new candidate tail and a separately versioned consistent source. V1 fails remain archived.')
    target.mkdir(parents=True)
    csvout(target/'CONVERGENCE.csv', convergence)
    csvout(target/'LEGACY_CONVERGENCE.csv', legacy_convergence)
    csvout(target/'TAIL_AND_CFL.csv', tails)
    csvout(target/'OPERATOR_IDENTITIES.csv', identities)
    csvout(target/'SOURCE_DEFINITION_BRIDGE.csv', bridge)
    csvout(target/'BUDGETS.csv', balances)
    write(target/'REFERENCE_QUALITY_V2.json', result, exclusive=True)
    plot(target, convergence, metadata, bridge, balances)
    report = report_cn(result, convergence, tails, identities, metadata)
    (target/'REFERENCE_PLATFORM_REPORT_CN.md').write_text(report, encoding='utf-8')
    write(target/'OUTPUT_MANIFEST.json', {x.name: sha(x) for x in target.iterdir() if x.is_file()}, exclusive=True)
    return result


def plot(target, convergence, metadata, bridge, balances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    spatial = [x for x in convergence if x['axis'] == 'space_512_1024_common_dt']
    labels = [x['field'].replace('reference_', '')+'/'+x['band'] for x in spatial]
    axs[0, 0].bar(range(len(spatial)), [100*x['max_frame_relative_RMS'] for x in spatial])
    axs[0, 0].axhline(1, color='r', linestyle='--'); axs[0, 0].set_xticks(range(len(spatial)), labels, rotation=55, ha='right')
    axs[0, 0].set_ylabel('Maximum frame relative RMS [%]'); axs[0, 0].set_title('Space: 512 vs 1024; consistent source')
    for name, m in metadata.items():
        if name.startswith('n1024'):
            hist = m['history']
            axs[0, 1].semilogy([x['time_s'] for x in hist], [x['tail_fraction'] for x in hist], label=name)
            for key in ('E_balance_relative', 'Z_balance_relative'):
                axs[1, 1].semilogy([x['time_s'] for x in hist], [max(x[key], 1e-18) for x in hist], label=name+'/'+key[0])
        else:
            axs[0, 1].axhline(m['max_tail'], alpha=.4, linestyle=':', label=name+' max')
    axs[0, 1].axhline(1e-4, color='r', linestyle='--'); axs[0, 1].set_title('Raw high-grid vorticity tail'); axs[0, 1].legend(fontsize=7)
    for name in metadata:
        rows = [x for x in bridge if x['variant'] == name and x['band'] == 'high']
        axs[1, 0].plot([x['time_s'] for x in rows], [100*x['legacy_equation_residual_over_RHS'] for x in rows], label=name)
    axs[1, 0].set_title('Legacy high-band equation residual / RHS [%]'); axs[1, 0].legend(fontsize=7)
    axs[1, 1].set_title('Fine-grid E/Z RK-quadrature balance residual'); axs[1, 1].legend(fontsize=7)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(target/'REFERENCE_QUALITY_OVERVIEW.png', dpi=170); plt.close(fig)


def report_cn(result, convergence, tails, identities, metadata):
    rows = ['# 独立 NS 参考平台 v2：开发初值质量报告', '',
        f'初值：{result["initial_seed"]}；状态：**{result["status"]}**。', '',
        '本报告只判定参考平台。FULL/A0B1 推理、Original/RemoveD 演化、H1–H4、网络训练均未运行。', '',
        '## 检查结果', '', '| 检查 | 最大实际值 | 阈值 | 结果 |', '|---|---:|---:|---|']
    for axis in dict.fromkeys(x['axis'] for x in convergence):
        xs = [x for x in convergence if x['axis'] == axis]
        maximum = max(max(x['relative_RMS'], x['max_frame_relative_RMS']) for x in xs)
        rows.append(f'| {axis} | {maximum:.9g} | 0.01 | {"PASS" if all(x["passes"] for x in xs) else "FAIL"} |')
    for t in tails:
        rows.append(f'| {t["variant"]} raw tail ({t["role"]}) | {t["max_tail"]:.9g} | 0.0001 | {"PASS" if t["tail_pass"] else "FAIL"} |')
    ident = max(max(x[k] for k in ('source_flux_relative','source_RHS_gap_relative','equation_residual_relative')) for x in identities)
    rows.append(f'| 独立通量 / RHS 定义一致性 | {ident:.9g} | 1e-10 | {"PASS" if result["reference_gates"]["independent_source_flux_and_RHS_identity"] else "FAIL"} |')
    rows += ['', '## 如何理解', '',
        '1. 1% 空间/时间比较、1e-4 谱尾、0.5 CFL、1e-10 恒等式门槛均沿用原数值值。',
        '2. 新参考的 source 使用同一过滤投影 F 下的高网格通量矩与低网格解析输运之差；另存原 PoF 标签及两者差异。',
        '3. 一致性恒等式通过只说明方程和标签口径相符；它本身不能证明空间收敛。',
        '4. 已有 512² 的 source gap 从保存的状态和 RHS 代数重建；没有伪造高网格矩。空间比较仍受实际粗细网格差控制。',
        '5. 粗网格谱尾 FAIL 保留为对照诊断。v2 要求两份新候选参考谱尾通过；这属于明确版本化的候选参考规则，不能回填 v1 的 PASS。',
        '6. 质量通过也不意味着 PoF 权重处于训练分布内、不意味着机制成立、不能直接称为 DNS。', '',
        '## 下一步', '',
        ('本开发初值满足预先声明的 v2 参考要求。可审查训练分布覆盖，再决定是否进行最小冻结模型机制实验；本程序不自动启动。'
         if result['status'].startswith('PASS') else
         '当前仍被参考质量阻断。失败项在 CONVERGENCE.csv 与 TAIL_AND_CFL.csv 中逐项保留；本程序停止，不自动增加网格、改黏度或放宽门槛。'), '']
    return '\n'.join(rows)
