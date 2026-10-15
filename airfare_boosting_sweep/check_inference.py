"""Synthetic tests of inversion, clipping, equal blends, IDs and one-use guard."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import inference as i


class IdentityFeatures:
    def transform(self,raw):return raw[['days_left']].to_numpy(float)


class DummyModel:
    def __init__(self,factor):self.factor=factor
    def predict(self,x):return x[:,0]*self.factor


def main():
    n=3898;ids=np.array([f'id_{k}' for k in range(n)])
    rng=np.random.default_rng(42);order=rng.permutation(n)
    raw=pd.DataFrame({'id':ids,'days_left':np.arange(n,dtype=float)-10})
    schema=pd.DataFrame({'id':ids[order],'price':np.zeros(n)})
    bundles={family:dict(features=IdentityFeatures(),model=DummyModel(k+1))
             for k,family in enumerate(i.m.FAMILIES)}
    outputs=i.make_outputs(raw,schema,bundles)
    assert len(outputs)==10
    values={family:np.maximum(0.,(np.arange(n)-10)*(k+1)*10000.)
            for k,family in enumerate(i.m.FAMILIES)}
    for family,expected in values.items():
        out=outputs[i.FILES[family]]
        assert out.id.to_numpy().tolist()==schema.id.to_numpy().tolist()
        np.testing.assert_array_equal(out.price.to_numpy(),expected[order])
    for name,weights in i.m.BLENDS.items():
        expected=sum(weight*values[family] for family,weight in weights.items())
        np.testing.assert_allclose(outputs[i.FILES[name]].price.to_numpy(),expected[order],rtol=0,atol=1e-7)
    with tempfile.TemporaryDirectory(prefix='airfare-highround-check-') as directory:
        out=Path(directory);(out/'INFERENCE_STARTED.json').write_text('{}')
        try:i.infer(Path('/missing_train'),Path('/missing_test'),Path('/missing_sample'),out)
        except AssertionError as exc:assert 'already started' in str(exc)
        else:raise AssertionError('Repeat inference was not blocked')
    i.e.save(i.ROOT/'inference_smoke.json',dict(synthetic_rows=n,output_count=10,
        exact_shuffled_sample_id_order=True,original_price_inversion=True,nonnegative_clipping=True,
        fixed_equal_weights=True,repeated_test_access_blocked_before_read=True,
        competition_test_sample_or_previous_submissions_read=False,new_model_fits=0))
    print('Synthetic inference checks passed for all10 outputs; no competition test data read.')


if __name__=='__main__':main()
