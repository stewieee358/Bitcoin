"""Three extra tree classifiers; registration leaves the upstream factories intact."""
import hashlib
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from src.models import MODEL_REGISTRY, ModelDefinition


def dt(params):
    return DecisionTreeClassifier(**{'random_state': 42, **params})


def ada(params):
    params = dict(params)
    depth = params.pop('base_max_depth', 1)
    leaf = params.pop('base_min_samples_leaf', 10)
    estimator = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=42)
    return AdaBoostClassifier(estimator=estimator, **{'random_state': 42, **params})


def gbm(params):
    if params.get('n_iter_no_change') is not None:
        raise ValueError('Random early-stopping validation is disabled in this temporal study')
    return GradientBoostingClassifier(**{'random_state': 42, **params})


def register_tree_models():
    # Factory helpers and their defaults are included in every new model identity.
    version = 'tree-v1-' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, factory in [('dt', dt), ('ada', ada), ('gbm', gbm)]:
        MODEL_REGISTRY[name] = ModelDefinition(factory, version=version)
