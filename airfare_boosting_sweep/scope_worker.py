"""Execute one untouched registered scope, using the unchanged tuner/models.

Used only after suspending the family's sequential runner and reserving future
scope directories. This is execution parallelism, not a different search or CV.
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import time
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('boosting_sweep_study',ROOT/'study.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
m=e.m;base=e.base


def main(train,family,scope):
    reg=e.verify(train);frame,raw,y=base.data(train)
    folder=ROOT/'results'/family/scope;folder.mkdir(parents=True,exist_ok=True)
    if scope=='full_training':
        assert not (folder/'configuration.json').exists()
        selected=e.tune(raw,y,family,folder/'tuning',e.FULL_SEED)
        features=m.Features(family);x=features.fit_transform(raw)
        model=m.estimator(family,selected['configuration'],features);m.fit(model,family,features,x,y)
        artifact=folder/'model.joblib'
        joblib.dump(dict(features=features,model=model,family=family,scale=m.SCALE),artifact,compress=3)
        e.save(folder/'configuration.json',dict(family=family,selection=selected,
            constructor_parameters=m.parameters(model,family),actual_rounds=int(m.actual_rounds(model,family)),
            feature_schema=features.description(),source_hashes=e.sources(),training_sha256=reg['training_sha256'],
            training_rows=len(y),training_r2=base.scores(y,m.predict(model,x))['r2'],model_sha256=e.sha(artifact),
            target_policy='price/10000 squared error; inverse10000; clip below0',test_accessed=False,
            execution_helper_sha256=e.sha(Path(__file__))))
    else:
        index=int(scope[-1]);tr,va=base.splits(raw,y,e.OUTER_SEED)[index]
        assert not (folder/'metrics.json').exists()
        assert set(base.group_ids(raw)[tr]).isdisjoint(base.group_ids(raw)[va])
        e.save(folder/'outer_indices.json',dict(fit=tr.tolist(),validation=va.tolist(),group_overlap=0))
        selected=e.tune(raw.iloc[tr].reset_index(drop=True),y[tr],family,folder/'tuning',e.INNER_SEED+index)
        tick=time.perf_counter();features=m.Features(family)
        x=features.fit_transform(raw.iloc[tr]);xv=features.transform(raw.iloc[va])
        model=m.estimator(family,selected['configuration'],features);m.fit(model,family,features,x,y[tr])
        pred=m.predict(model,xv);pt=m.predict(model,x)
        pd.DataFrame(dict(row_position=va,prediction=pred)).to_csv(folder/'predictions.csv',index=False)
        np.savez_compressed(folder/'train_predictions.npz',row_position=tr,prediction=pt)
        e.save(folder/'metrics.json',dict(family=family,fold=index,source_hashes=e.sources(),selection=selected,
            **base.scores(y[va],pred),train_r2=base.scores(y[tr],pt)['r2'],n_fit=len(tr),n_validation=len(va),
            group_overlap=0,actual_rounds=int(m.actual_rounds(model,family)),refit_seconds=time.perf_counter()-tick,
            execution_helper_sha256=e.sha(Path(__file__))))
    print('REGISTERED_SCOPE_COMPLETE',family,scope,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',type=Path,default=base.TRAIN)
    parser.add_argument('--family',choices=['classic_gradient','adaboost'],required=True)
    parser.add_argument('--scope',choices=[f'outer{i}' for i in range(5)]+['full_training'],required=True)
    args=parser.parse_args();main(args.train,args.family,args.scope)
