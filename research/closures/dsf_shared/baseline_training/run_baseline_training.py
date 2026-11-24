"""One explicit entry for the new data/initialization/selection benchmark."""
import argparse
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import common as c
import data
import training
import evaluation
import rollout
import report


def require_tests(a):
    import xml.etree.ElementTree as ET
    receipt=c.read(HERE/'results/TEST_ACCEPTANCE.json')
    if receipt['code_sha256']!=a['code_sha256'] or c.sha(receipt['xml'])!=receipt['xml_sha256']:
        raise RuntimeError('CPU test/code identity mismatch')
    for s in ET.parse(receipt['xml']).getroot().iter('testsuite'):
        if int(s.get('errors',0)) or int(s.get('failures',0)):raise RuntimeError('CPU tests did not pass')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('audit','prepare','train','evaluate','short','long','report'))
    p.add_argument('--output',type=Path,default=HERE/'results/run01')
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--stop-after-epoch',type=int)
    args=p.parse_args(argv);out=c.safe(args.output,True)
    if args.stop_after_epoch is not None and not 1<=args.stop_after_epoch<=c.protocol()['epochs']:
        p.error('--stop-after-epoch must be within the fixed 1..100 budget')
    if args.stage not in ('audit','report') and (args.dry_run or not args.execute):
        a=dict(protocol=c.protocol())
        rows=rollout.matrix(a,args.stage) if args.stage in ('short','long') else c.matrix(a) if args.stage=='train' else []
        print(__import__('json').dumps(dict(status='DRY_RUN_NO_DATA_DECODE_NO_TRAINING_NO_LES',stage=args.stage,matrix=rows,
            quality='REFERENCE_NOT_CONVERGED',time_range=[1.,5.]),indent=2));return
    if args.stage=='audit':
        if (out/'AUDIT.json').exists():raise FileExistsError('Immutable audit exists; use existing run or new output')
        out.mkdir(parents=True,exist_ok=True);a=c.identity();c.dump(out/'AUDIT.json',a)
        c.csvout(out/'TRAINING_MATRIX.csv',c.matrix(a));c.csvout(out/'ROLLOUT_MATRIX.csv',rollout.matrix(a,'short')+rollout.matrix(a,'long'))
        print('PASS_METADATA_ONLY '+str(out/'AUDIT.json'))
    else:
        a=c.read(out/'AUDIT.json');c.verify(a)
        if args.stage!='report':require_tests(a)
        try:
            if args.stage=='prepare':data.prepare(a,out,args.resume)
            elif args.stage=='train':training.run(a,out,args.device,args.resume,args.stop_after_epoch)
            elif args.stage=='evaluate':evaluation.apriori(a,out,args.device,args.resume)
            elif args.stage in ('short','long'):rollout.run(a,out,args.stage,args.device,args.resume)
        except BaseException:
            report.generate(a,out);raise
    print(report.generate(a,out))


if __name__=='__main__':main()
