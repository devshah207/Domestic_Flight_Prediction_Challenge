"""Run independent model families concurrently without duplicating any fit.

The original all-family runner is temporarily suspended (not terminated). Each
worker owns distinct family output directories. Resume the original runner only
after every worker exits; its existing completion checks skip finished families.
This changes execution scheduling only, never configurations, seeds or budgets.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main(pid):
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'airfare_boosting_sweep/study.py' in command and b'--family' not in command
    state=dict(original_runner_pid=pid,original_runner_suspended=True,
        scheduling_only=True,configuration_changes=False,additional_research_fits=0,
        started_at_utc=datetime.now(timezone.utc).isoformat(),workers={})
    def worker(label,families):
        records=[]
        for family in families:
            path=ROOT/f'worker_{family}.log'
            with path.open('w') as log:
                result=subprocess.run([sys.executable,str(ROOT/'study.py'),'--stage','run','--family',family],
                    cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=False)
            records.append(dict(family=family,returncode=result.returncode))
            if result.returncode!=0:break
        return label,records
    os.kill(pid,signal.SIGSTOP)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(worker,'classic',['classic_gradient']),
                  pool.submit(worker,'ada',['adaboost']),
                  pool.submit(worker,'native',['lgb_gbdt','lgb_dart','cat_symmetric','cat_leafwise','hist_gradient'])]
            for job in jobs:
                label,records=job.result();state['workers'][label]=records
    finally:
        os.kill(pid,signal.SIGCONT)
        state['original_runner_suspended']=False
        state['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        (ROOT/'SCHEDULING.json').write_text(json.dumps(state,indent=2)+'\n')
    assert all(r['returncode']==0 for records in state['workers'].values() for r in records),state
    print('Independent workers completed; original all-family runner resumed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pid',required=True,type=int)
    args=parser.parse_args();main(args.pid)
