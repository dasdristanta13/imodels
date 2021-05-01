import os
import pickle as pkl
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_openml


MODEL_COMPARISON_PATH = os.path.dirname(os.path.realpath(__file__)) + "/comparison_data/"

DATASET_PATH = os.path.dirname(os.path.realpath(__file__)) + "/data/"


class Model:
    def __init__(self, name: str, cls, vary_param: str, vary_param_val: Any, 
                 fixed_param: str = None, fixed_param_val: Any = None,
                 other_params: Dict[str, Any] = {}):
        self.name = name
        self.cls = cls
        self.fixed_param = fixed_param
        self.fixed_param_val = fixed_param_val
        self.vary_param = vary_param
        self.vary_param_val = vary_param_val
        self.kwargs = {self.vary_param: self.vary_param_val}
        if self.fixed_param is not None:
            self.kwargs[self.fixed_param] = self.fixed_param_val
        self.kwargs = {**self.kwargs, **other_params}


def get_openml_dataset(data_id: int) -> pd.DataFrame:
    dataset = fetch_openml(data_id=data_id, as_frame=False)
    X = dataset.data
    if issparse(X):
        X = X.toarray()
    y = (dataset.target == dataset.target[0]).astype(int)
    feature_names = dataset.feature_names

    target_name = dataset.target_names
    if target_name[0].lower() == 'class':
        target_name = [dataset.target[0]]

    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame(y, columns=target_name)
    return pd.concat((X_df, y_df), axis=1)


def get_clean_dataset(path: str) -> Tuple[np.array]:
    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    feature_names = df.columns.values[:-1]
    return np.nan_to_num(X.astype('float32')), y, feature_names


def get_comparison_result(path: str, estimator_name: str, prefix='val', low_data=False, easy=False) -> Dict[str, Any]:
    path += 'low_data/' if low_data else 'reg_data/'
    path += 'easy/' if easy else 'hard/'
    if prefix == 'test':
        result_file = path + 'test/' + f'{estimator_name}_test_comparisons.pkl'
    elif prefix == 'cv':
        result_file = path + 'cv/' + f'{estimator_name}_comparisons.pkl'
    else:
        result_file = path + 'val/' + f'{estimator_name}_comparisons.pkl'
    return pkl.load(open(result_file, 'rb'))


def get_best_model_under_complexity(c: int, model_name: str, model_cls: BaseEstimator,
                                    curve_param = None,
                                    metric: str = 'mean_ROCAUC',
                                    kwargs: dict = {},
                                    prefix: str = 'cv',
                                    easy: bool = False) -> BaseEstimator:
    # init_models = []
    # for m_name, m_cls in models:
    result = get_comparison_result(MODEL_COMPARISON_PATH, model_name, prefix=prefix, easy=easy)
    df, auc_metric = result['df'], result['meta_auc_df'][f'{metric}_auc']

    if curve_param:
        # specify which curve to use
        df_best_curve = df[df.iloc[:, 1] == curve_param]
    else:
        # detect which curve to use
        df_best_curve = df[df.index == auc_metric.idxmax()]
    
    df_under_c = df_best_curve[df_best_curve['mean_complexity'] < c]
    if df_under_c.shape[0] == 0:
        warnings.warn(f'{model_name} skipped for complexity limit {c}')
        return None

    best_param = df_under_c.iloc[:, 0][df_under_c[metric].argmax()]
    kwargs[df_under_c.columns[0]] = best_param

    # if there is a second param that was varied
    if auc_metric.shape[0] > 1:
        kwargs[df_under_c.columns[1]] = int(df_under_c.iloc[0, 1])

    return model_cls(**kwargs)