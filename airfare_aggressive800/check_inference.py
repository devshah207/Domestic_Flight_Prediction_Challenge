"""Synthetic one-file inference check; no competition data or model fitting."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('aggressive_inference_test',ROOT/'inference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Features:
    def transform(self,x):return x[['value']]


class Model:
    def __init__(self,multiplier):self.multiplier=multiplier
    def predict(self,x):return x.value.to_numpy()*self.multiplier


def main():
    with tempfile.TemporaryDirectory(dir='/tmp',prefix='airfare800-synthetic-') as directory:
        p=Path(directory);out=p/'out';out.mkdir();fresh=p/'fresh';(fresh/'submissions').mkdir(parents=True)
        ids=pd.Series([str(i) for i in range(3898)])
        raw=pd.DataFrame({'id':ids.iloc[::-1].to_numpy(),'value':np.arange(3898)/100-1,
            'flight_class':np.where(np.arange(3898)%2==0,'Business','Economy')})
        sample=pd.DataFrame({'id':ids,'price':0});raw.to_csv(p/'test.csv',index=False);sample.to_csv(p/'sample.csv',index=False)
        old_audit={k:hashlib.sha256((p/v).read_bytes()).hexdigest() for k,v in [('test_sha256','test.csv'),('sample_sha256','sample.csv')]}
        (fresh/'submissions/inference_audit.json').write_text(json.dumps(old_audit));m.e.old.common.FRESH=fresh
        bundles={k:{'features':Features(),'model':Model(v)} for k,v in [('global_xgb',1),('specialist',5)]}
        def fake_freeze(train,out):
            if (out/'INFERENCE_STARTED.json').exists():raise RuntimeError('already accessed')
            (out/'frozen_manifest.json').write_text('{}');return {},bundles
        m.freeze=fake_freeze;m.infer(p/'unused.csv',p/'test.csv',p/'sample.csv',out)
        result=pd.read_csv(out/m.FILENAME,dtype={'id':str});assert result.id.equals(ids)
        parsed=pd.read_csv(p/'test.csv');g=np.maximum(0,parsed.value.to_numpy()*10000)[::-1]
        business=parsed.flight_class.eq('Business').to_numpy()[::-1];expected=g.copy();expected[business]*=3
        np.testing.assert_allclose(result.price,expected,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(result.price[~business],g[~business],rtol=0,atol=1e-8)
        assert len(list(out.glob('submission*.csv')))==1
        try:m.infer(p/'unused.csv',p/'test.csv',p/'sample.csv',out)
        except RuntimeError:pass
        else:raise AssertionError('Repeated inference was allowed')
        print('PASS: one CSV, sample ordering, clipping/scaling, Business-only mixing, Economy invariance and repeat guard.')


if __name__=='__main__':main()
