"""Static identity audit and bounded CPU synthetic acceptance; zero real CNN."""
from pathlib import Path
import time
import numpy as np
import torch
import pc_common as c


def audit(reuse=False):
    inventory=c.inventory(); identity=c.local_identity(); phase=c.HERE/'phase1'; acceptance=phase/'ACCEPTANCE.json'
    if reuse and acceptance.exists():
        old=c.read(acceptance)
        c.require(old['code_identity']==identity and old['inventory']==inventory and old['tests']['status']=='PASS',
                  'Static acceptance/code/input changed')
        return old
    phase.mkdir(parents=True,exist_ok=True)
    c.require(not acceptance.exists(),'Static acceptance exists; do not overwrite')
    from pc_tests import run
    tests=run(phase/'synthetic')
    checkpoints=[]; initials=[]
    for row in inventory['checkpoints']:
        s=c.model(row,'cpu')
        h=s.closure.register_forward_pre_hook(lambda *_:(_ for _ in ()).throw(AssertionError('Static audit called CNN')))
        checkpoints.append(dict(seed=row['seed'],epoch=row['epoch'],sha256=row['sha256'],
            parameters=sum(p.numel() for p in s.parameters()),strict_load='VERIFIED'))
        if row['seed']==2026:
            from pc_engine import basic_metrics
            for case,desc in inventory['references'].items():
                r=c.References(desc); first=r.at(0)
                w=s.project(torch.tensor(first['reference_omega'],dtype=torch.float64))
                initials.append(dict(case=int(case),**basic_metrics(s,w)))
        h.remove()
    c.csvout(phase/'INITIAL_STATES.csv',initials)
    c.dump(phase/'CHECKPOINTS.json',checkpoints)
    from pc_report import build_report
    template=build_report(phase/'NOT_RUN_TEMPLATE',template=True)
    result=dict(status='VERIFIED_STATIC_SYNTHETIC',real_CNN=0,real_full_LES='NOT_RUN',
        code_identity=identity,inventory=inventory,tests=tests,checkpoint_loads=checkpoints,
        reference_quality=c.QUALITY,report_template=str(template.relative_to(c.HERE)),
        checkpoint_model_reachability='NOT_RUN',universal_signed_history_closure='FAIL_GATE_NO_UNIVERSAL_GUARANTEE')
    c.dump(acceptance,result)
    return result


if __name__=='__main__':
    torch.set_num_threads(2)
    result=audit()
    print('VERIFIED static/synthetic; real CNN=0; full LES=NOT_RUN; reference frames:',
          {k:len(v['steps']) for k,v in result['inventory']['references'].items()})
