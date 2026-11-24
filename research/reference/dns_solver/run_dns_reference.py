"""Metadata preflight by default; explicit --execute for every real-array stage."""
import argparse
import json
from rp_common import HERE, protocol, config, scope, reused, run_path, prepare, verify


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    for action in ('check-only', 'prepare', 'synthetic', 'generate', 'quality', 'run'):
        actions.add_argument('--'+action, action='store_true')
    p.add_argument('--seed', type=int, default=110240101)
    p.add_argument('--run-dir')
    p.add_argument('--variant', choices=list(protocol()['new_variants']))
    p.add_argument('--device', default='cuda', choices=('cpu','cuda'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv); scope(a.seed)
    if (a.run or a.generate or a.quality) and not a.execute:
        print('REFUSED_EXECUTE_REQUIRED: no arrays decoded, no GPU kernels or evolution started')
        return 2
    if a.synthetic:
        import subprocess, sys
        return subprocess.call([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            str(HERE/'tests'), '--basetemp', str(HERE/'test_tmp_cli')])
    run = run_path(a.run_dir, a.seed)
    if a.check_only:
        c = config(a.seed); old = reused(a.seed)
        print(json.dumps(dict(status='PASS_METADATA_ONLY', seed=a.seed,
            physical=dict(nu=c['nu'], length=c['length'], delta=c['delta'], les_n=c['les_n']),
            window=c['window'], new_tasks=protocol()['new_variants'],
            reused={name: dict(n=x['metadata']['n'], dt=x['metadata']['dt_s'], max_tail=x['metadata']['max_tail'],
                               previous_seconds=x['metadata']['seconds']) for name,x in old.items()},
            output=str(run), arrays_decoded=False, GPU_started=False,
            no_model_training=True, no_locked_or_reserved_access=True), ensure_ascii=False, indent=2))
        return 0
    if a.prepare or (a.run and not run.exists()):
        f = prepare(run, a.seed)
        if a.prepare:
            print('PREPARED '+str(run)); return 0
    else:
        f = verify(run, a.seed)
    if a.generate and not a.variant:
        p.error('--generate requires --variant')
    if a.run or a.generate:
        from rp_compute import generate
        for tag in ([a.variant] if a.generate else f['protocol']['new_variants']):
            generate(run, f, tag, a.device, a.resume)
    if a.run or a.quality:
        from rp_quality import qualify
        if (run/'quality/OUTPUT_MANIFEST.json').exists():
            from rp_common import read, sha
            for name, digest in read(run/'quality/OUTPUT_MANIFEST.json').items():
                if sha(run/'quality'/name) != digest:
                    raise ValueError('Existing quality result changed')
            result = read(run/'quality/REFERENCE_QUALITY_V2.json')
        else:
            result = qualify(run, f)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
