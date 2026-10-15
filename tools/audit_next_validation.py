"""Independently match stored tuning identities to training-only partitions."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'airfare_business'))
import specialist as s


def raw_hash(raw):return hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest()
def target_hash(y):return hashlib.sha256(y.tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train',type=Path,default=s.base.TRAIN);args=parser.parse_args()
    frame,raw,y=s.base.data(args.train);audits=[]
    for label,seed in [('primary',20261020),('sensitivity',20261027)]:
        root=REPO/'airfare_business'/label
        if not (root/'summary.json').exists():continue
        for i,(tr,va) in enumerate(s.base.splits(raw,y,seed)):
            dest=root/f'outer{i}';local=raw.iloc[tr].reset_index(drop=True)
            global_selected=json.loads((dest/'global_selected.json').read_text())
            specialist=json.loads((dest/'selected.json').read_text())
            g=s.base.group_ids(raw)
            assert set(g[tr]).isdisjoint(g[va])
            assert global_selected['signature']['row_hash']==raw_hash(local)
            assert global_selected['signature']['target_hash']==target_hash(y[tr])
            assert specialist['signature']['local_raw_sha256']==raw_hash(local)
            assert specialist['signature']['local_y_sha256']==target_hash(y[tr])
            inner_seed=(20262020 if label=='primary' else 20276020)+i
            assert global_selected['signature']['seed']==inner_seed
            assert specialist['signature']['inner_seed']==inner_seed
            reconstructions=json.loads((dest/'global_inner_reconstructions.json').read_text())
            local_groups=s.base.group_ids(local)
            for k,(it,iv) in enumerate(s.base.splits(local,y[tr],inner_seed)[:2]):
                assert set(local_groups[it]).isdisjoint(local_groups[iv])
                bt=it[local.iloc[it].flight_class.to_numpy()=='Business']
                bv=iv[local.iloc[iv].flight_class.to_numpy()=='Business']
                assert hashlib.sha256(bt.tobytes()).hexdigest()==reconstructions[k]['fit_positions_sha256']
                assert hashlib.sha256(bv.tobytes()).hexdigest()==reconstructions[k]['tune_positions_sha256']
                assert len(bt)==specialist['inner_splits'][k]['n_fit']
                assert len(bv)==specialist['inner_splits'][k]['n_tune']
                assert abs(reconstructions[k]['global_r2_reconstruction_error'])<1e-9
            audits.append(dict(study='business',split=label,fold=i,global_and_specialist_input_hashes_match=True,
                global_and_specialist_seed_identities_match=True,inner_business_fit_indices_match=True,group_overlap=0))
    root=REPO/'airfare_round2/primary/cat64_direct'
    for i,(tr,va) in enumerate(s.base.splits(raw,y,20261020)):
        selected=json.loads((root/f'outer{i}/selected.json').read_text())
        assert selected['signature']['raw_sha256']==raw_hash(raw.iloc[tr].reset_index(drop=True))
        assert selected['signature']['target_sha256']==target_hash(y[tr])
        assert selected['signature']['seed']==20262220+i and selected['threshold']==64
        audits.append(dict(study='direct_cat64',split='primary',fold=i,input_hashes_and_seed_match=True))
    for phase in ['airfare_round2','airfare_business']:
        reg=json.loads((REPO/phase/'REGISTRATION.json').read_text())
        for name,digest in reg['source_hashes'].items():assert s.base.sha(REPO/name)==digest,name
    direct_full=json.loads((root/'full_training/selected.json').read_text())
    business_full=json.loads((REPO/'airfare_business/full_training/selected.json').read_text())
    assert direct_full['signature']['raw_sha256']==raw_hash(raw) and direct_full['signature']['target_sha256']==target_hash(y)
    assert business_full['signature']['local_raw_sha256']==raw_hash(raw) and business_full['signature']['local_y_sha256']==target_hash(y)
    output=dict(training_sha256=s.base.sha(args.train),training_rows=len(y),checks=audits,full_training_input_identities_verified=True,
        conclusion='Stored global and specialist tuning inputs correspond to outer training partitions; group and inner specialist fitting identities verified.',
        limitation='This audits leakage boundaries, not hidden-test performance or absence of statistical overfitting.',test_accessed=False)
    (REPO/'NEXT_VALIDATION_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Validation identities verified for',len(audits),'outer scopes; no test/sample files read.')


if __name__=='__main__':main()
