"""Synthetic inference and target-encoding leakage checks, no regressor fitting."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import inference as i

class DummyFeatures:
    def transform(self,raw):return raw[['days_left']].to_numpy(float)
class DummyModel:
    def __init__(self,k):self.k=k
    def predict(self,x):return x[:,0]*self.k

def main():
    n=3898;ids=np.array([f'id{k}' for k in range(n)]);order=np.random.default_rng(42).permutation(n)
    raw=pd.DataFrame(dict(id=ids,days_left=np.arange(n)-10));sample=pd.DataFrame(dict(id=ids[order],price=np.zeros(n)))
    bundles={f:dict(model=DummyModel(k+1),features=DummyFeatures()) for k,f in enumerate(i.m.FAMILIES)}
    result=i.outputs(raw,sample,bundles);assert len(result)==7
    expected={f:np.maximum(0.,(np.arange(n)-10)*(k+1)*10000.) for k,f in enumerate(i.m.FAMILIES)}
    for name,weights in i.m.BLENDS.items():expected[name]=sum(w*expected[f] for f,w in weights.items())
    for name,values in expected.items():
        frame=result[i.FILES[name]];assert frame.id.equals(sample.id)
        np.testing.assert_allclose(frame.price.to_numpy(),values[order],rtol=0,atol=1e-7)
    with tempfile.TemporaryDirectory(prefix='peer-synthetic-') as d:
        p=Path(d);(p/'INFERENCE_STARTED.json').write_text('{}')
        try:i.infer(Path('/missing_train'),Path('/missing_test'),Path('/missing_sample'),p)
        except AssertionError as exc:assert 'already started' in str(exc)
        else:raise AssertionError('Repeat test read not blocked')
    k=np.arange(120);group=k%30
    frame=pd.DataFrame(dict(airline='Air',flight=[f'AA {g}' for g in group],source_city='A',
        departure_time='Morning',stops='zero',arrival_time='Evening',destination_city='B',
        flight_class='Business',duration=2.,days_left=group))
    y=10000.+1000.*group;mask=group==0;changed=y.copy();changed[mask]+=1e7
    a=i.m.Features('rf_smooth_auto');x=a.fit_transform(frame,y)
    b=i.m.Features('rf_smooth_auto');z=b.fit_transform(frame,changed)
    np.testing.assert_array_equal(x[mask,:len(a.cats)],z[mask,:len(a.cats)])
    assert a.encoder.smooth=='auto' and a.encoder.target_type=='continuous'
    assert all(p['group_overlap']==0 for p in a.encoding_folds)
    i.c.save(i.ROOT/'synthetic_checks.json',dict(inference_rows=n,outputs=7,target_inverse=True,
        shuffled_sample_order=True,clipping=True,equal_weights=True,repeat_guard=True,
        own_group_target_perturbation_cannot_change_its_training_encoding=True,
        regressor_fits=0,competition_inputs_read=False,synthetic_encoder_mapping_fits=12))
    print('Synthetic checks passed: 7 outputs, inverse/order/clip/weights/repeat guard and grouped target isolation.')
if __name__=='__main__':main()
