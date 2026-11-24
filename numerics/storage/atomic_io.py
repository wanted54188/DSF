"""Retry transient Windows atomic-replace failures without weakening atomicity."""
import os
from pathlib import Path
import time

DELAYS=(.02,.05,.1,.2,.4,.8,1.,1.,1.,1.)
RETRY_WINERRORS=frozenset((5,32,33))


class RetryReplace:
    def __init__(self,replace=os.replace,delays=DELAYS,sleep=time.sleep):
        self.native=replace; self.delays=tuple(delays); self.sleep=sleep; self.events=[]

    def __call__(self,source,destination,*args,**kwargs):
        errors=[]; begin=time.perf_counter()
        for attempt in range(len(self.delays)+1):
            try:
                result=self.native(source,destination,*args,**kwargs)
            except OSError as error:
                code=getattr(error,'winerror',None)
                errors.append(dict(attempt=attempt+1,winerror=code,errno=error.errno))
                if code not in RETRY_WINERRORS or attempt==len(self.delays):
                    self.events.append(dict(source=str(source),destination=str(destination),recovered=False,
                        errors=errors,elapsed_seconds=time.perf_counter()-begin))
                    if hasattr(error,'add_note'):
                        error.add_note('Atomic replace retained the old target and the new temp file. No delete/truncate/ACL change was attempted.')
                    raise
                self.sleep(self.delays[attempt])
                continue
            if errors:
                self.events.append(dict(source=str(source),destination=str(destination),recovered=True,
                    errors=errors,elapsed_seconds=time.perf_counter()-begin))
            return result


class ScopedOS:
    """Proxy ONLY pc_common.os; the actual global os module stays untouched."""
    def __init__(self,native,root,replacer):
        self.native=native; self.root=Path(root).resolve(); self.replacer=replacer
    def __getattr__(self,name): return getattr(self.native,name)
    def replace(self,source,destination,*args,**kwargs):
        source_path=Path(source).resolve(); destination_path=Path(destination).resolve()
        if source_path.is_relative_to(self.root) and destination_path.is_relative_to(self.root):
            return self.replacer(source,destination,*args,**kwargs)
        return self.native.replace(source,destination,*args,**kwargs)
