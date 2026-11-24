"""Bounded, counted resource calibration; not reference qualification."""
import sys,json,time,hashlib,ctypes,traceback,gc
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ENGINE=ROOT/'A0B1 full/jcp_dissipation_20261024/decisive/reference_platform_v2_20261025'
sys.path[:0]=[str(ROOT),str(ENGINE)]
from rp_spectral import SpectralRK4
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((ROOT/'research/ssc_long_horizon_v1_20261030/results/run01/reference/110240101/n1536_half/DONE.json').read_text())['config']
    rows=[]
    for n in (1536,3072):
        row=dict(n=n,case=110240105,dt=.000125,steps=12,planned_RHS=48,actual_RHS=0,scope='calibration only; never scored as reference')
        try:
            torch.cuda.reset_peak_memory_stats()
            s=SpectralRK4(n,cfg['length'],cfg['nu'],'cuda')
            h=s.initial(110240105,cfg['initial'])
            torch.cuda.synchronize();begin=time.perf_counter()
            for i in range(12):
                row['actual_RHS']+=4
                h,_,_=s.step(h,.000125)
                if i==1:
                    torch.cuda.synchronize();start=time.perf_counter()
            torch.cuda.synchronize()
            row.update(status='PASS',total_seconds=time.perf_counter()-begin,seconds_per_step=(time.perf_counter()-start)/10,peak_GPU_bytes=torch.cuda.max_memory_allocated(),finite=bool(torch.isfinite(h).all()))
            del h,s
        except BaseException as ex:
            row.update(status='FAILED_RESOURCE',error=str(ex),traceback=traceback.format_exc())
        rows.append(row)
        (HERE/'RESOURCE_CALIBRATION.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
        gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
