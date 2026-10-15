"""Read fitted trees to verify actual counts and effective structural bounds."""
import json
import numpy as np


def binary_shape(node,kind,depth=0):
    if kind=='xgb':
        children=node.get('children',[])
    else:
        children=[node['left_child'],node['right_child']] if 'left_child' in node else []
    if not children:return 1,depth
    parts=[binary_shape(child,kind,depth+1) for child in children]
    return sum(n for n,d in parts),max(d for n,d in parts)


def inspect(model,family,config):
    p=config['selection']['configuration'];depth=p['depth']
    cap=min(p.get('leaves',2**depth),2**depth)
    if family=='xgb':
        shapes=[binary_shape(json.loads(tree),'xgb') for tree in model.get_booster().get_dump(dump_format='json')]
    elif family.startswith('lgb'):
        shapes=[binary_shape(tree['tree_structure'],'lgb') for tree in model.booster_.dump_model()['tree_info']]
    elif family.startswith('cat'):
        leaves=model.get_tree_leaf_counts()
        shapes=[(int(n),int(round(np.log2(n))) if family=='cat_symmetric' else None) for n in leaves]
    elif family=='hist_gradient':
        shapes=[(int(round_trees[0].nodes['is_leaf'].sum()),int(round_trees[0].nodes['depth'].max()))
                for round_trees in model._predictors]
    else:
        trees=model.estimators_ if family=='adaboost' else model.estimators_.ravel()
        shapes=[(int(tree.get_n_leaves()),int(tree.get_depth())) for tree in trees]
    observed_leaves=max(n for n,d in shapes)
    observed_depth=max(d for n,d in shapes if d is not None) if any(d is not None for n,d in shapes) else None
    assert len(shapes)==config['actual_rounds'] and observed_leaves<=cap
    if observed_depth is not None:assert observed_depth<=depth
    return dict(family=family,requested_round_cap=p['rounds'],actual_trees=len(shapes),
        configured_max_depth=depth,observed_max_depth=observed_depth,
        effective_leaf_cap=cap,observed_max_leaves=observed_leaves,
        total_fitted_leaves=int(sum(n for n,d in shapes)),actual_tree_structure_bounds_passed=True,
        note='Lossguide CatBoost depth is constructor-bounded; this audit extracts its leaf counts, not node depths.' if family=='cat_leafwise' else '')
