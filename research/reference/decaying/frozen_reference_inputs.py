"""Immutable, scoped campaign IO. Importing this module never decodes arrays."""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parent.parent
# Reuse the audited local operators/loaders, without changing their files.
for p in (ROOT,STUDY.parent,STUDY.parent/"dissipation_check_20261024",STUDY):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
sys.path.insert(0,str(HERE))

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def rules():return read(HERE/"DECISION_RULES.json")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def csvread(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def scope(seed):
    for role,seeds in rules()["roles"].items():
        if seed in seeds:return role
    raise PermissionError("Reserved/unregistered initial seed is forbidden")
def campaign(p):
    p=Path(p).resolve()
    if not p.is_relative_to(HERE/"results"):raise ValueError("Campaign must be inside decisive/results")
    return p
def seal(directory,meta,files):
    meta=dict(meta,outputs_sha256={n:sha(directory/n) for n in files})
    dump(directory/"DONE.json",meta);return meta
def verified(directory):
    m=read(directory/"DONE.json")
    for name,digest in m["outputs_sha256"].items():
        p=(directory/name).resolve()
        if not p.is_relative_to(directory.resolve()) or sha(p)!=digest:raise ValueError("Artifact hash mismatch "+name)
    return m
def freeze_verify(run):
    f=read(run/"FREEZE.json")
    for path,digest in f["identities"].items():
        if sha(path)!=digest:raise ValueError("Frozen source/config changed: "+path)
    return f
def matrix():
    r=rules();rows=[]
    for role,seeds in r["roles"].items():
        for s in seeds:
            for v in r["reference_variants"]:
                rows.append(dict(task_id=f"ref_{s}_{v}",action="reference",role=role,initial_seed=s,model="",weight_seed="",condition="",dt_s=r["reference_variants"][v]["dt"],variant=v,dependency="FREEZE",state="NOT_RUN"))
            rows.append(dict(task_id=f"quality_{s}",action="quality",role=role,initial_seed=s,model="",weight_seed="",condition="",dt_s="",variant="",dependency=f"four reference variants for {s}",state="NOT_RUN"))
            dts=[r["main_LES_dt"]]+([r["fine_LES_dt"]] if s in r["fine_initial_seeds"] else [])
            for dt in dts:
                rows.append(dict(task_id=f"traj_{s}_NoSGS_0_NoSGS_{dt}",action="trajectory",role=role,initial_seed=s,model="NoSGS",weight_seed=0,condition="NoSGS",dt_s=dt,variant="both_fine",dependency=f"quality_{s}",state="NOT_RUN"))
                for m in r["models"]:
                    for w in r["weight_seeds"]:
                        for c in ("Original","RemoveD"):
                            rows.append(dict(task_id=f"traj_{s}_{m}_{w}_{c}_{dt}",action="trajectory",role=role,initial_seed=s,model=m,weight_seed=w,condition=c,dt_s=dt,variant="both_fine",dependency=f"quality_{s}",state="NOT_RUN"))
            for m in r["models"]:
                for w in r["weight_seeds"]:
                    rows.append(dict(task_id=f"instant_{s}_{m}_{w}",action="instant",role=role,initial_seed=s,model=m,weight_seed=w,condition="same_reference_state",dt_s="",variant="both_fine",dependency=f"quality_{s}",state="NOT_RUN"))
            rows.append(dict(task_id=f"analyze_{s}",action="analyze",role=role,initial_seed=s,model="",weight_seed="",condition="",dt_s="",variant="",dependency=f"all trajectories and instant for {s}",state="NOT_RUN"))
    return rows
def taskdir(run,row):return run/"tasks"/row["task_id"]
def taskbyid(identifier):
    q=[r for r in matrix() if r["task_id"]==identifier]
    if len(q)!=1:raise ValueError("Unknown or unregistered task ID")
    return q[0]
