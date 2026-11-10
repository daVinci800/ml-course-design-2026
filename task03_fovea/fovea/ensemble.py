"""A small, predeclared set of OOF ensembles; never mix validation folds."""
import numpy as np


def candidate_recipes():
    # Fixed before either v4 run is trained. Names identify run and stage.
    recipes={name:{name:1.} for name in ('baseline_coarse','baseline_fine','resnet_coarse','resnet_fine','convnext_coarse','convnext_fine')}
    recipes.update(
        resnet_stages={'resnet_coarse':.5,'resnet_fine':.5},
        convnext_stages={'convnext_coarse':.5,'convnext_fine':.5},
        new_fine_ensemble={'resnet_fine':.5,'convnext_fine':.5},
        new_coarse_ensemble={'resnet_coarse':.5,'convnext_coarse':.5},
        new_all_stages={name:.25 for name in ('resnet_coarse','resnet_fine','convnext_coarse','convnext_fine')},
        three_fine_ensemble={name:1/3 for name in ('baseline_fine','resnet_fine','convnext_fine')})
    return recipes


def combine_predictions(components,weights):
    if not weights or any(name not in components for name in weights):
        raise ValueError('Missing ensemble components.')
    coefficients=np.array(list(weights.values()),dtype=float)
    if not np.isfinite(coefficients).all() or (coefficients<0).any() or coefficients.sum()<=0:
        raise ValueError('Invalid ensemble weights.')
    ids=set(components[next(iter(weights))])
    if any(set(components[name])!=ids for name in weights):
        raise ValueError('Component IDs do not match.')
    result={}
    for ident in ids:
        points=np.array([components[name][ident]['xy'] for name in weights],dtype=float)
        if points.shape!=(len(weights),2) or not np.isfinite(points).all():
            raise ValueError('Invalid component coordinate.')
        result[ident]=dict(xy=np.average(points,axis=0,weights=coefficients).tolist())
    return result
