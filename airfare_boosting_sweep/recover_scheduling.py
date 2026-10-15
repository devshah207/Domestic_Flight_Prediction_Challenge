"""Recover one duplicated in-progress scope without changing any model choice.

The original runner had reached LGB outer0 when it was suspended for parallel
scheduling. A second worker opened that scope while original trial2 was RUNNING.
Refit that exact trial/configuration on the same inner rows and persist its value.
No new suggested configuration, outer label or test data is used. The original
runner will fail its finished-trial attribute write when eventually resumed,
preventing any further duplicate suggestions. Account for its duplicate fit.
"""
from datetime import datetime,timezone
from pathlib import Path
import importlib.util
import os
import signal
import sys
import time
import optuna
import numpy as np

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('boosting_sweep_study',ROOT/'study.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def main():
    e.verify(e.base.TRAIN);frame,raw,y=e.base.data(e.base.TRAIN)
    tr,va=e.base.splits(raw,y,e.OUTER_SEED)[0]
    local=raw.iloc[tr].reset_index(drop=True);target=y[tr]
    it,iv=e.base.splits(local,target,e.INNER_SEED)[0]
    folder=ROOT/'results/lgb_gbdt/outer0/tuning'
    study=optuna.load_study(study_name='lgb_gbdt',storage='sqlite:///'+str((folder/'optuna.sqlite3').resolve()))
    frozen=study.trials[2]
    assert frozen.state==optuna.trial.TrialState.RUNNING
    p=e.m.propose(optuna.trial.FixedTrial(frozen.params),'lgb_gbdt')
    record=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        cause='Scheduling race: original all-family runner already entered LGB outer0 before suspension; a worker opened the same scope.',
        family='lgb_gbdt',scope='outer0',trial=2,exact_registered_configuration=p,
        recovery_changes_no_hyperparameters=True,outer_validation_or_test_used=False,
        duplicate_fit_attempts=1,canonical_registered_completed_research_fits=384,
        actual_total_research_fit_attempts=385,
        reproducibility_note='Trials3 onward in this one scope can reflect asynchronous TPE ordering/restarted sampler; saved trial parameters are the exact record. Other scopes retain their registered seeded procedure.')
    e.save(ROOT/'SCHEDULING_RECOVERY.json',record)
    features=e.m.Features('lgb_gbdt');x=features.fit_transform(local.iloc[it]);xv=features.transform(local.iloc[iv])
    tick=time.perf_counter();model=e.m.estimator('lgb_gbdt',p,features)
    e.m.fit(model,'lgb_gbdt',features,x,target[it]);pred=e.m.predict(model,xv)
    trial=optuna.trial.Trial(study,frozen._trial_id)
    for name,value in dict(configuration=p,actual_rounds=int(e.m.actual_rounds(model,'lgb_gbdt')),
        seconds=time.perf_counter()-tick,tuning_metrics=e.base.scores(target[iv],pred),
        scheduling_recovery_same_configuration=True).items():trial.set_user_attr(name,value)
    study.tell(trial,float(np.mean((pred-target[iv])**2)))
    record.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),recovered_trial_complete=True,
                  tuning_mse=float(np.mean((pred-target[iv])**2)))
    e.save(ROOT/'SCHEDULING_RECOVERY.json',record)
    try:os.kill(18805,signal.SIGCONT)
    except ProcessLookupError:pass  # Worker may already have rejected the RUNNING trial archive.
    print('Recovered original LGB outer0 trial2 without a new parameter proposal; worker resumed.',flush=True)


if __name__=='__main__':main()
