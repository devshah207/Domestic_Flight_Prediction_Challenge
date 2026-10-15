"""Reserve distinct future scopes before parallel execution; no overlap allowed."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
RUNNERS={'classic_gradient':18802,'adaboost':18804}


def main():
    reserved=[];suspended=[]
    state=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),scheduling_only=True,
               changed_model_choices=False,extra_model_fits=0,jobs=[],sequential_runners_resumed=False)
    try:
        for family,pid in RUNNERS.items():
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            assert b'--family\x00'+family.encode()+b'\x00' in command
            os.kill(pid,signal.SIGSTOP);suspended.append(pid)
            directory=ROOT/'results'/family
            last=max(int(p.name[-1]) for p in directory.glob('outer[0-4]'))
            assert not (directory/'full_training').exists()
            future=[f'outer{i}' for i in range(last+1,5)]+['full_training']
            for scope in future:
                assert not (directory/scope).exists(),'Future scope already exists; refuse overlap'
                reserved.append((family,scope))
            state[family]=dict(pid=pid,reserved_sequential_prefix_through=f'outer{last}',parallel_future_scopes=future)
        (ROOT/'SCOPE_SCHEDULING_STARTED.json').write_text(json.dumps(state,indent=2)+'\n')
        def worker(family,scope):
            with (ROOT/f'scope_{family}_{scope}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(ROOT/'scope_worker.py'),'--family',family,'--scope',scope],
                    cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=False)
            return dict(family=family,scope=scope,returncode=result.returncode)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,*job) for job in reserved]
            for future in as_completed(futures):state['jobs'].append(future.result())
        assert all(row['returncode']==0 for row in state['jobs']),state
    finally:
        for pid in suspended:os.kill(pid,signal.SIGCONT)
        state['sequential_runners_resumed']=True
        state['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        (ROOT/'SCOPE_SCHEDULING.json').write_text(json.dumps(state,indent=2)+'\n')
    print('All reserved future scopes completed; sequential runners resumed without extra fits.',flush=True)


if __name__=='__main__':main()
