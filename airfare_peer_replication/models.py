"""Disclosed peer settings; no invented smooth feature or leaderboard tuning."""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'airfare_fresh'))
from features import TabularFeatures
FAMILIES=['rf_ohe','xgb_ohe','rf_smooth_auto','xgb_smooth_auto','cat']
SCALE=10000.
BLENDS={'equal_ohe':{f:1/3 for f in ['rf_ohe','xgb_ohe','cat']},
        'equal_smooth_auto':{f:1/3 for f in ['rf_smooth_auto','xgb_smooth_auto','cat']}}

class Features:
    def __init__(self,family):self.family=family
    def fit_transform(self,raw,y=None):
        self.base=TabularFeatures('context','cat')
        frame=self.base.fit_transform(raw)
        self.cats=self.base.cat_columns_;self.nums=self.base.numeric_columns_
        if self.family.endswith('smooth_auto'):
            assert y is not None
            from pandas import util
            groups=util.hash_pandas_object(raw.drop(columns=['airline','duration']).astype(str),index=False).to_numpy()
            self.encoder=TargetEncoder(smooth='auto',target_type='continuous',cv=5,shuffle=True,random_state=42)
            self.encoder.fit(frame[self.cats].astype(str),np.asarray(y)/SCALE)
            self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            numeric=self.imputer.fit_transform(frame[self.nums]).astype(np.float32)
            encoded=np.empty((len(frame),len(self.cats)),float);self.encoding_folds=[]
            for tr,va in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(frame,groups=groups):
                assert set(groups[tr]).isdisjoint(groups[va])
                encoder=TargetEncoder(smooth='auto',target_type='continuous',random_state=42)
                encoder.fit(frame.iloc[tr][self.cats].astype(str),np.asarray(y)[tr]/SCALE)
                encoded[va]=encoder.transform(frame.iloc[va][self.cats].astype(str))
                self.encoding_folds.append(dict(fit=tr.tolist(),validation=va.tolist(),group_overlap=0))
            return np.column_stack([encoded,numeric])
        if self.family!='cat':
            self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
            self.encoder.fit(frame[self.cats].astype(str))
            self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            self.imputer.fit(frame[self.nums])
        return self.transform(raw)
    def transform(self,raw):
        frame=self.base.transform(raw)
        if self.family=='cat':return frame
        if self.family.endswith('smooth_auto'):
            return np.column_stack([self.encoder.transform(frame[self.cats].astype(str)),self.imputer.transform(frame[self.nums])])
        return sparse.hstack([self.encoder.transform(frame[self.cats].astype(str)),
            sparse.csr_matrix(self.imputer.transform(frame[self.nums]).astype(np.float32))],format='csr')
    def description(self):
        return dict(features=self.base.feature_columns_,categorical=self.cats,numerical=self.nums,
            representation='native CatBoost categories' if self.family=='cat' else 'group-cross-fitted TargetEncoder + numeric medians' if self.family.endswith('smooth_auto') else 'training-fitted sparse one-hot + numeric medians',
            smooth='sklearn TargetEncoder smooth=auto; interpretation, not confirmed peer implementation' if self.family.endswith('smooth_auto') else 'not applied',mode='context; assumption, not confirmed peer feature list')

def estimator(family):
    if family.startswith('rf'):
        return RandomForestRegressor(n_estimators=200,max_depth=2,max_leaf_nodes=4,
            max_features=.7,bootstrap=False,min_samples_leaf=1,min_samples_split=2,
            criterion='squared_error',random_state=42,n_jobs=4)
    if family.startswith('xgb'):
        return XGBRegressor(n_estimators=964,max_depth=9,max_leaves=512,
            learning_rate=.054,random_state=42,n_jobs=4,objective='reg:squarederror',
            tree_method='hist',grow_policy='depthwise',max_bin=256,min_child_weight=1.,
            reg_lambda=1.,reg_alpha=0.,gamma=0.,subsample=1.,colsample_bytree=1.,
            colsample_bylevel=1.,colsample_bynode=1.,verbosity=0)
    assert family=='cat'
    return CatBoostRegressor(iterations=1000,depth=6,learning_rate=.139,
        random_seed=42,thread_count=4,loss_function='RMSE',eval_metric='RMSE',
        grow_policy='SymmetricTree',l2_leaf_reg=3.,random_strength=1.,
        bootstrap_type='Bayesian',bagging_temperature=1.,rsm=1.,border_count=254,
        one_hot_max_size=2,max_ctr_complexity=1,ctr_leaf_count_limit=512,
        verbose=False,allow_writing_files=False)

def fit(model,family,features,x,y):
    model.fit(x,np.asarray(y)/SCALE,**({'cat_features':features.cats} if family=='cat' else {}))
    return model

def predict(model,x):return np.maximum(0.,np.asarray(model.predict(x),float)*SCALE)
def count(model,family):
    return len(model.estimators_) if family.startswith('rf') else model.get_booster().num_boosted_rounds() if family.startswith('xgb') else model.tree_count_
