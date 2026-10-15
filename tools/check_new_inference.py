"""Synthetic check of new candidate inference; no competition data or model fits."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'airfare_round2';sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('new_inference',ROOT/'inference.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Features:
    def transform(self,x):return x[['value']]


class Model:
    def __init__(self,multiplier):self.multiplier=multiplier
    def predict(self,x):return x.value.to_numpy()*self.multiplier


def main():
    with tempfile.TemporaryDirectory(dir='/tmp',prefix='airfare-inference-synthetic-') as directory:
        p=Path(directory);out=p/'out';out.mkdir();fresh=p/'fresh';(fresh/'submissions').mkdir(parents=True)
        ids=pd.Series([str(i) for i in range(3898)])
        raw=pd.DataFrame({'id':ids.iloc[::-1].to_numpy(),'value':np.arange(3898)/100-1,
            'flight_class':np.where(np.arange(3898)%2==0,'Business','Economy')})
        schema=pd.DataFrame({'id':ids,'price':0});raw.to_csv(p/'test.csv',index=False);schema.to_csv(p/'sample.csv',index=False)
        audit={k:hashlib.sha256((p/v).read_bytes()).hexdigest() for k,v in [('test_sha256','test.csv'),('sample_sha256','sample.csv')]}
        (fresh/'submissions/inference_audit.json').write_text(json.dumps(audit));module.run.FRESH=fresh
        bundles={k:{'features':Features(),'model':Model(v)} for k,v in
            [('xgb',1),('lgb',2),('cat64_direct',4),('cat_onehot64',3),('business_specialist',5)]}
        def fake_freeze(train,out,include_business=False):
            if (out/'INFERENCE_STARTED.json').exists():raise RuntimeError('already accessed')
            (out/'frozen_manifest.json').write_text('{}');return {},bundles
        module.freeze=fake_freeze
        module.infer(p/'unused.csv',p/'test.csv',p/'sample.csv',out,True)
        # Compute expected values from the same synthetic CSV parsing boundary.
        parsed=pd.read_csv(p/'test.csv');g=np.maximum(0,parsed.value.to_numpy()*10000)[::-1]
        mask=parsed.flight_class.eq('Business').to_numpy()[::-1]
        augmented=g.copy();augmented[mask]*=3
        expected={'submission_catboost64_direct_tuned.csv':4*g,'submission_equal_blend_direct64.csv':7*g/3,
            'submission_business_augmented_xgb.csv':augmented,'submission_business_equal_blend64.csv':(augmented+5*g)/3}
        for filename,want in expected.items():
            actual=pd.read_csv(out/filename,dtype={'id':str})
            assert actual.id.equals(ids) and list(actual)==['id','price'] and len(actual)==3898
            np.testing.assert_allclose(actual.price,want,rtol=1e-12,atol=1e-8)
        # Production code already asserts bit-exact Economy invariance in memory;
        # CSV decimal serialization/parsing permits insignificant roundoff here.
        actual=pd.read_csv(out/'submission_business_augmented_xgb.csv')
        np.testing.assert_allclose(actual.price[~mask],g[~mask],rtol=0,atol=1e-8)
        try:module.infer(p/'unused.csv',p/'test.csv',p/'sample.csv',out,True)
        except RuntimeError:pass
        else:raise AssertionError('Repeated inference was not refused')
        print('PASS: four outputs, scaling/clipping, ID alignment, Business gating, Economy invariance, blends and access guard.')


if __name__=='__main__':main()
