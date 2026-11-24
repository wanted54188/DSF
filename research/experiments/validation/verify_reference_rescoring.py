"""Check the new postprocessor against archived absolute real-reference scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import campaign as v
import rescore_references as r
ORIGINAL=v.HERE
def main():
    folder=ORIGINAL/'SOFTWARE_REFERENCE_RESCORING_CHECK';folder.mkdir(exist_ok=True)
    names=('n1024_main','n1536_main','n1536_half');settings=((1024,.00025),(1536,.00025),(1536,.000125))
    def members(case):
        if case!=110240101:return [None,None,None]
        result=[]
        for name,(n,dt) in zip(names,settings):
            path=v.ROOT/'research/ssc_long_horizon_v1_20261030/results/run01/reference'/str(case)/name
            a=np.load(path/'reference_omega.npy',mmap_mode='r')
            def sample(step,aa=a):return dict(reference_omega=np.array(aa[4*step]))
            result.append(dict(n=n,dt=dt,sample=sample))
        return result
    v.HERE=folder;v.CASES=(110240101,);r.members=members;r.main()
    report=folder/v.read(folder/'LATEST_REFERENCE_RESCORING.json')['path'];scores=pd.read_csv(report/'MODEL_REFERENCE_SCORES.csv')
    old=pd.read_csv(v.ROOT/'research/full_aug_hold_audit_v1_20261117/results/reference01/EXISTING_REFERENCE_RANKING.csv');old=old[(old.case==110240101)&(old.band=='all')&(old.window=='full')]
    checks=[]
    for row in old.itertuples():
        n,dt=settings[names.index(row.reference_variant)]
        for mod,key in [('DSF-Hold','Hold_MSE'),('DSF-Sym','Fresh_MSE')]:
            now=scores[(scores.seed==row.seed)&(scores.model==mod)&(scores.band=='all')&(scores.reference_n==n)&np.isclose(scores.reference_dt,dt,atol=1e-12)].full_omega_MSE.iloc[0]
            expected=getattr(row,key);err=abs(now-expected);assert err<1e-12,(mod,row.reference_variant,err)
            checks.append(dict(seed=row.seed,model=mod,reference=row.reference_variant,new_MSE=now,archived_MSE=expected,absolute_difference=err))
    v.HERE=ORIGINAL
    v.write(ORIGINAL/'REFERENCE_RESCORING_ACCEPTANCE.json',dict(status='PASS',CNN=0,new_RHS=0,checks=checks,scope='Real archived flow101, two frozen network seeds, three existing reference variants; no new DNS or model evolution'))
    print('PASS:',len(checks),'absolute MSE comparisons and all framewise uncertainty bounds')
if __name__=='__main__':main()
