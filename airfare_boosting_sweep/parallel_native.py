"""Overlap remaining independent families; pause only the serial shell launcher.

The current DART child continues uninterrupted. Its parent shell cannot start
future families until all dedicated family workers have completed, preventing
overlapping scopes. Scheduling never changes registered fitting procedures.
"""
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
FUTURE=['cat_symmetric','cat_leafwise','hist_gradient']


def main(parent):
    command=Path(f'/proc/{parent}/cmdline').read_bytes()
    assert b'for family in lgb_gbdt lgb_dart cat_symmetric cat_leafwise hist_gradient' in command
    state=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),parent_shell_pid=parent,
        current_dart_child_continues=True,configuration_changes=False,extra_model_fits=0,jobs=[])
    os.kill(parent,signal.SIGSTOP)
    try:
        for family in FUTURE:assert not (ROOT/'results'/family).exists(),'Refuse existing future-family output'
        def worker(family):
            with (ROOT/f'parallel_{family}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(ROOT/'study.py'),'--stage','run','--family',family],
                    cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=False)
            return dict(family=family,returncode=result.returncode)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,family) for family in FUTURE]
            for future in as_completed(futures):state['jobs'].append(future.result())
        assert all(job['returncode']==0 for job in state['jobs']),state
    finally:
        os.kill(parent,signal.SIGCONT)
        state['parent_shell_resumed']=True;state['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        (ROOT/'NATIVE_SCHEDULING.json').write_text(json.dumps(state,indent=2)+'\n')
    print('Native family workers completed; serial shell resumed with completed outputs.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent-pid',required=True,type=int)
    main(parser.parse_args().parent_pid)
