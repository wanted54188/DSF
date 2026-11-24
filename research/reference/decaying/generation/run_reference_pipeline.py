"""One manual --execute: original NS continuation -> quality -> LES -> plots/verdict."""
from pathlib import Path
import argparse, json, os, platform, shutil, sys, time, traceback
import common as c

def process_alive(pid):
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
        h=kernel.OpenProcess(0x1000,False,pid)
        if not h:return False
        code=ctypes.c_ulong();kernel.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong))
        ok=kernel.GetExitCodeProcess(h,ctypes.byref(code));kernel.CloseHandle.argtypes=(ctypes.c_void_p,);kernel.CloseHandle(h)
        return bool(ok and code.value==259)
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def recover(run,restart_uncommitted):
    for lock in run.rglob('RUNNING.lock'):
        if process_alive(int(lock.read_text())):raise RuntimeError('A live process holds '+str(lock))
        lock.unlink();c.log(run,'RECOVER stale lock '+str(lock))
    if restart_uncommitted:
        for s in c.SEEDS:
            for v in c.VARIANTS:
                d=run/'reference'/str(s)/v
                if (d/'reference_omega.npy').exists() and not (d/'CHECKPOINT.json').exists() and not (d/'DONE.json').exists():
                    if not d.resolve().is_relative_to(run.resolve()):raise PermissionError('Invalid recovery directory')
                    backup=d.with_name(d.name+'_interrupted_'+str(time.time_ns()));d.rename(backup)
                    c.log(run,'PRESERVE uncommitted task '+str(backup))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--output',type=Path,default=c.HERE/'results/run01');p.add_argument('--reference-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--stage',choices=['all','reference','quality','evaluate'],default='all');p.add_argument('--resume',action='store_true')
    p.add_argument('--recover-lock',action='store_true');p.add_argument('--restart-uncommitted',action='store_true')
    args=p.parse_args();run=c.output_guard(args.output)
    if not args.execute or args.check_only:
        a=c.audit();estimate=c.estimate(a);deps=c.dependencies()
        c.dump(c.HERE/'METADATA_AUDIT.json',a);c.dump(c.HERE/'RESOURCE_ESTIMATE.json',estimate);c.dump(c.HERE/'PROTOCOL.json',c.policy())
        c.dump(c.HERE/'DEPENDENCY_AUDIT.json',dict(status='PASS_REAL_SOURCE_FILES',files=deps))
        print(json.dumps(dict(status='PASS_METADATA_ONLY_NO_NUMERICS',audit=str(c.HERE/'METADATA_AUDIT.json'),dependency_files=len(deps),estimate=estimate,policy=c.policy()),ensure_ascii=True,indent=2));return
    run.mkdir(parents=True,exist_ok=True)
    if (run/'FREEZE.json').exists():
        if not args.resume:raise FileExistsError('Existing output requires --resume or a fresh output directory')
        f=c.verify_frozen(run)
        if f['reference_device']!=args.reference_device:raise ValueError('Resume must use frozen reference device')
    else:
        a=c.audit();estimate=c.estimate(a);free=shutil.disk_usage(run).free/1e9
        if free<estimate['minimum_free_GB']:raise RuntimeError(f'Need {estimate["minimum_free_GB"]:.1f} GB free; have {free:.1f}')
        c.dump(run/'FREEZE.json',dict(policy=c.policy(),audit=a,dependencies=c.dependencies(),reference_device=args.reference_device,
            python=sys.version,platform=platform.platform(),argv=sys.argv,created_time=time.strftime('%Y-%m-%d %H:%M:%S')))
        c.dump(run/'METADATA_AUDIT.json',a);c.dump(run/'RESOURCE_ESTIMATE.json',estimate);f=c.read(run/'FREEZE.json')
    from reporting import blocked
    if args.recover_lock:recover(run,args.restart_uncommitted)
    elif args.restart_uncommitted:raise ValueError('--restart-uncommitted also requires --recover-lock')
    global_lock=run/'RUNNING.lock'
    with global_lock.open('x') as lock:lock.write(str(os.getpid()))
    try:
        c.log(run,'START stage='+args.stage+' device='+args.reference_device)
        if args.stage in ('all','reference'):
            from reference import integrate
            for case in f['audit']['cases']:
                for v in case['variants']:integrate(run,case,v,args.reference_device,args.resume)
        if args.stage in ('all','reference','quality'):
            from quality import qualify
            qualities=[qualify(run,case) for case in f['audit']['cases']]
            c.dump(run/'COMPLETE_LONG_REFERENCE_MANIFEST.json',dict(status='PASS_LONG_REFERENCE_QUALITY' if all(q['status']=='PASS_LONG_REFERENCE_QUALITY' for q in qualities) else 'INCONCLUSIVE_BLOCKED_REFERENCE_QUALITY',cases=qualities))
            if not all(q['status']=='PASS_LONG_REFERENCE_QUALITY' for q in qualities):
                blocked(run,'实际长期质量门禁未通过，详细失败项见quality/；已停止LES精度结论。不得放宽阈值或静默加密。');c.log(run,'BLOCKED_REFERENCE_QUALITY');return 2
        if args.stage in ('all','evaluate'):
            from evaluation import run as evaluate
            evaluate(run,args.resume)
        else:blocked(run,'参考阶段已完成；尚未运行匹配的长期精度评估。执行同目录 --stage evaluate --resume --execute。')
        c.verify_frozen(run);c.log(run,'COMPLETED stage='+args.stage);return 0
    except BaseException as exc:
        c.dump(run/f'PIPELINE_ERROR_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),stage=args.stage))
        blocked(run,'执行未完成：'+repr(exc)+'；已保留checkpoint、原错误和日志，尚不能给出科学结论。')
        c.log(run,'FAILED '+repr(exc));raise
    finally:
        global_lock.unlink(missing_ok=True)
        report=run/'SSC_LONG_HORIZON_VERDICT_CN.md'
        if report.exists():
            shutil.copyfile(report,c.HERE/report.name)
            c.dump(c.HERE/'LATEST_RESULT.json',dict(run=str(run),report=str(report),report_sha256=c.sha(report),published_time=time.strftime('%Y-%m-%d %H:%M:%S')))

if __name__=='__main__':sys.exit(main())
