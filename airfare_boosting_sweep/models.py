"""Bounded global boosting models and fitting-partition-only preprocessing."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from scipy import sparse
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'airfare_fresh'))
from features import TabularFeatures

FAMILIES = ['xgb', 'lgb_gbdt', 'lgb_dart', 'cat_symmetric', 'cat_leafwise',
            'hist_gradient', 'classic_gradient', 'adaboost']
TRIALS = dict(zip(FAMILIES, [8, 8, 6, 8, 8, 6, 6, 6]))
ROUNDS = [800, 1100, 1400]
SCALE = 10000.
THREADS = 4
BLENDS = {
    'equal_main3': {'xgb': 1/3, 'lgb_gbdt': 1/3, 'cat_symmetric': 1/3},
    'equal_dart3': {'xgb': 1/3, 'lgb_dart': 1/3, 'cat_symmetric': 1/3},
}


class Features:
    """All learned mappings fit solely on the rows passed to fit_transform."""
    def __init__(self, family):
        self.family = family

    def fit_transform(self, raw):
        kind = ('xgb' if self.family == 'xgb' else
                'lgb' if self.family.startswith('lgb') else 'cat')
        self.base = TabularFeatures('context', kind)
        frame = self.base.fit_transform(raw)
        self.cats = self.base.cat_columns_
        self.nums = self.base.numeric_columns_
        if self.family in ['classic_gradient', 'adaboost', 'hist_gradient']:
            self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
            self.imputer.fit(frame[self.nums])
            if self.family == 'hist_gradient':
                self.levels = {}
                for c in self.cats:
                    counts = frame[c].astype(str).value_counts()
                    # Stable count descending / lexical tie-break; reserve zero for OTHER.
                    levels = sorted(counts.index, key=lambda v: (-int(counts[v]), v))[:127]
                    self.levels[c] = {v: i+1 for i, v in enumerate(levels)}
            else:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True,
                                             dtype=np.float32)
                self.encoder.fit(frame[self.cats].astype(str))
        return self.transform(raw)

    def transform(self, raw):
        frame = self.base.transform(raw)
        if self.family.startswith('cat'):
            # CatBoost cannot accept NaN numerical categories, but numerical NaN is valid.
            return frame
        if self.family in ['xgb', 'lgb_gbdt', 'lgb_dart']:
            return frame
        numeric = self.imputer.transform(frame[self.nums]).astype(np.float32)
        if self.family == 'hist_gradient':
            cats = np.column_stack([frame[c].astype(str).map(self.levels[c]).fillna(0)
                                    .to_numpy(np.float32) for c in self.cats])
            return np.column_stack([cats, numeric])
        categorical = self.encoder.transform(frame[self.cats].astype(str))
        return sparse.hstack([categorical, sparse.csr_matrix(numeric)], format='csr')

    def description(self):
        result = dict(mode='context', raw_features=list(self.base.RAW_COLUMNS),
                      context_features=self.base.feature_columns_,
                      categorical_features=self.cats, numerical_features=self.nums)
        if self.family == 'hist_gradient':
            result.update(representation='native categorical top127 plus OTHER',
                          maximum_category_levels=128)
        elif self.family in ['classic_gradient', 'adaboost']:
            result.update(representation='sparse one-hot plus numeric median',
                          encoded_columns=int(sum(len(v) for v in self.encoder.categories_)
                                              + len(self.nums)))
        else:
            result['representation'] = 'native categorical'
        return result


def propose(trial, family):
    p = dict(rounds=trial.suggest_categorical('rounds', ROUNDS))
    if family == 'adaboost':
        p.update(depth=trial.suggest_int('depth', 4, 8),
                 leaves=trial.suggest_categorical('leaves', [24, 48, 64, 96]),
                 min_child=trial.suggest_int('min_child', 2, 16),
                 learning_rate=trial.suggest_float('learning_rate', .003, .06, log=True),
                 loss=trial.suggest_categorical('loss', ['linear', 'square']),
                 column_sample=trial.suggest_float('column_sample', .75, 1.))
        return p
    p.update(depth=trial.suggest_int('depth', 4, 8),
             learning_rate=trial.suggest_float('learning_rate', .012, .05, log=True))
    if family not in ['cat_symmetric']:
        p['leaves'] = trial.suggest_categorical('leaves', [16, 32, 48, 64])
    if family in ['xgb', 'lgb_gbdt', 'lgb_dart', 'cat_symmetric', 'cat_leafwise']:
        p.update(l2=trial.suggest_float('l2', 1., 25., log=True),
                 row_sample=trial.suggest_float('row_sample', .75, 1.),
                 column_sample=trial.suggest_float('column_sample', .75, 1.))
    if family in ['xgb', 'lgb_gbdt', 'lgb_dart']:
        p.update(min_child=trial.suggest_int('min_child', 3 if family == 'xgb' else 10,
                                           24 if family == 'xgb' else 60),
                 l1=trial.suggest_float('l1', .001, 1., log=True),
                 split_gain=trial.suggest_float('split_gain', .001, .08, log=True))
        if family.startswith('lgb'):
            p['cat_smooth'] = trial.suggest_float('cat_smooth', 2., 30., log=True)
        if family == 'lgb_dart':
            p.update(drop_rate=trial.suggest_float('drop_rate', .03, .15),
                     skip_drop=trial.suggest_float('skip_drop', .4, .8))
    elif family.startswith('cat'):
        p.update(random_strength=trial.suggest_float('random_strength', .05, 1.5, log=True),
                 one_hot=trial.suggest_categorical('one_hot', [2, 64]))
        if family == 'cat_leafwise':
            p['min_child'] = trial.suggest_int('min_child', 5, 40)
    elif family == 'hist_gradient':
        p.update(min_child=trial.suggest_int('min_child', 10, 60),
                 l2=trial.suggest_float('l2', .1, 20., log=True))
    else:
        p.update(min_child=trial.suggest_int('min_child', 5, 40),
                 row_sample=trial.suggest_float('row_sample', .75, 1.),
                 column_sample=trial.suggest_float('column_sample', .6, 1.),
                 split_gain=trial.suggest_float('split_gain', .00001, .003, log=True))
    return p


def initial(family, index):
    p = dict(rounds=800 if index == 0 else 1400, depth=6 if index == 0 else 7,
             learning_rate=.03 if index == 0 else .018, leaves=48 if index == 0 else 64,
             min_child=12 if family == 'xgb' else 25, l2=8., row_sample=.9,
             column_sample=.9, l1=.1, split_gain=.015, cat_smooth=10.,
             drop_rate=.08, skip_drop=.6, random_strength=.4, one_hot=64)
    if family == 'hist_gradient':
        p['min_child'] = 25
    if family == 'classic_gradient':
        p['split_gain'] = .0001
    if family == 'adaboost':
        p.update(depth=6 if index == 0 else 8, leaves=48 if index == 0 else 96,
                 min_child=5, learning_rate=.01 if index == 0 else .005,
                 loss='square' if index == 0 else 'linear')
    # Avoid queued unused parameters in Optuna's saved parameter dictionaries.
    class Recorder:
        def __init__(self): self.names = []
        def suggest_categorical(self, name, choices): self.names.append(name); return p[name]
        def suggest_int(self, name, low, high): self.names.append(name); return p[name]
        def suggest_float(self, name, low, high, **kw): self.names.append(name); return p[name]
    recorder = Recorder(); propose(recorder, family)
    return {name:p[name] for name in recorder.names}


def estimator(family, p, features):
    n = int(p['rounds']); depth = int(p['depth'])
    leaves = min(int(p.get('leaves', 2**depth)), 2**depth)
    lr = p['learning_rate']
    if family == 'xgb':
        return XGBRegressor(n_estimators=n, max_depth=depth, max_leaves=leaves,
            learning_rate=lr, min_child_weight=p['min_child'], reg_lambda=p['l2'],
            reg_alpha=p['l1'], gamma=p['split_gain'], subsample=p['row_sample'],
            colsample_bytree=p['column_sample'], objective='reg:squarederror',
            tree_method='hist', grow_policy='depthwise', max_bin=192,
            enable_categorical=True, max_cat_to_onehot=4, max_cat_threshold=32,
            random_state=42, n_jobs=THREADS, verbosity=0)
    if family.startswith('lgb'):
        extra = (dict(drop_rate=p['drop_rate'], skip_drop=p['skip_drop'], max_drop=50,
                      uniform_drop=False, xgboost_dart_mode=False, drop_seed=42)
                 if family == 'lgb_dart' else {})
        return LGBMRegressor(n_estimators=n, max_depth=depth, num_leaves=leaves,
            boosting_type='dart' if family == 'lgb_dart' else 'gbdt',
            learning_rate=lr, min_child_samples=p['min_child'], reg_lambda=p['l2'],
            reg_alpha=p['l1'], min_split_gain=p['split_gain'], subsample=p['row_sample'],
            subsample_freq=1, colsample_bytree=p['column_sample'], max_bin=191,
            max_cat_threshold=32, max_cat_to_onehot=4, cat_smooth=p['cat_smooth'],
            cat_l2=5., min_data_per_group=5, objective='regression', random_state=42,
            n_jobs=THREADS, verbosity=-1, deterministic=True, force_col_wise=True, **extra)
    if family.startswith('cat'):
        extra = (dict(grow_policy='Lossguide', max_leaves=leaves,
                      min_data_in_leaf=p['min_child']) if family == 'cat_leafwise'
                 else dict(grow_policy='SymmetricTree'))
        return CatBoostRegressor(iterations=n, depth=depth, learning_rate=lr,
            loss_function='RMSE', eval_metric='RMSE', l2_leaf_reg=p['l2'],
            random_strength=p['random_strength'], bootstrap_type='Bernoulli',
            subsample=p['row_sample'], rsm=p['column_sample'],
            one_hot_max_size=p['one_hot'], max_ctr_complexity=1,
            ctr_leaf_count_limit=512, model_size_reg=1., border_count=128,
            random_seed=42, thread_count=THREADS, verbose=False,
            allow_writing_files=False, **extra)
    if family == 'hist_gradient':
        return HistGradientBoostingRegressor(max_iter=n, max_depth=depth,
            max_leaf_nodes=leaves, learning_rate=lr, min_samples_leaf=p['min_child'],
            l2_regularization=p['l2'], max_bins=255, early_stopping=False,
            categorical_features=[True]*len(features.cats)+[False]*len(features.nums),
            random_state=42, loss='squared_error')
    if family == 'classic_gradient':
        return GradientBoostingRegressor(n_estimators=n, max_depth=depth,
            max_leaf_nodes=leaves, learning_rate=lr, min_samples_leaf=p['min_child'],
            min_samples_split=2*p['min_child'], min_impurity_decrease=p['split_gain'],
            subsample=p['row_sample'], max_features=p['column_sample'],
            random_state=42, loss='squared_error', criterion='friedman_mse')
    tree = DecisionTreeRegressor(max_depth=depth, max_leaf_nodes=leaves,
        min_samples_leaf=p['min_child'], min_samples_split=2*p['min_child'],
        max_features=p['column_sample'], min_impurity_decrease=.00001, random_state=42)
    return AdaBoostRegressor(estimator=tree, n_estimators=n, learning_rate=lr,
                             loss=p['loss'], random_state=42)


def fit(model, family, features, x, y):
    kwargs = dict(cat_features=features.cats) if family.startswith('cat') else {}
    model.fit(x, np.asarray(y)/SCALE, **kwargs)
    return model


def predict(model, x):
    p = np.maximum(0., np.asarray(model.predict(x), float).reshape(-1)*SCALE)
    assert np.isfinite(p).all(), 'Nonfinite prediction'
    return p


def actual_rounds(model, family):
    if family == 'xgb': return model.get_booster().num_boosted_rounds()
    if family.startswith('cat'): return model.tree_count_
    if family == 'hist_gradient': return model.n_iter_
    if family == 'adaboost': return len(model.estimators_)
    if family.startswith('lgb'): return model.n_iter_
    return model.n_estimators_


def parameters(model, family):
    if family.startswith('cat'): return model.get_all_params()
    if family == 'adaboost':
        return dict(n_estimators=model.n_estimators, learning_rate=model.learning_rate,
                    loss=model.loss, random_state=model.random_state,
                    base_estimator=model.estimator.get_params())
    return model.get_params()
