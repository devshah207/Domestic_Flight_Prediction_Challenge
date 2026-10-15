"""Parallelize reserved future histogram scopes using the unchanged scope fitter."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def coordinate(pid):
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'--family\x00hist_gradient\x00' in command
    state=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),family='hist_gradient',
        configuration_changes=False,extra_model_fits=0,jobs=[])
    os.kill(pid,signal.SIGSTOP)
    try:
        directory=ROOT/'results/hist_gradient'
        last=max(int(p.name[-1]) for p in directory.glob('outer[0-4]'))
        assert not (directory/'full_training').exists()
        scopes=[f'outer{i}' for i in range(last+1,5)]+['full_training']
        for scope in scopes:assert not (directory/scope).exists(),'Refuse overlapping scope'
        state.update(sequential_prefix_through=f'outer{last}',parallel_future_scopes=scopes)
        (ROOT/'HIST_SCHEDULING_STARTED.json').write_text(json.dumps(state,indent=2)+'\n')
        def worker(scope):
            with (ROOT/f'scope_hist_gradient_{scope}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__)),'--scope',scope],
                    cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=False)
            return dict(scope=scope,returncode=result.returncode)
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(worker,scope) for scope in scopes]
            for job in as_completed(jobs):state['jobs'].append(job.result())
        assert all(job['returncode']==0 for job in state['jobs']),state
    finally:
        os.kill(pid,signal.SIGCONT)
        state['original_runner_resumed']=True;state['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        (ROOT/'HIST_SCHEDULING.json').write_text(json.dumps(state,indent=2)+'\n')
    print('Reserved histogram scopes finished; original scope resumes without duplicated fits.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int)
    parser.add_argument('--scope',choices=[f'outer{i}' for i in range(5)]+['full_training'])
    args=parser.parse_args()
    if args.scope:
        from scope_worker import main,e
        main(e.base.TRAIN,'hist_gradient',args.scope)
    else:
        assert args.pid
        coordinate(args.pid)
