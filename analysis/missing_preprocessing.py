"""Training-only imputation for the model pipeline. No CSV modification.

Use a fixed feature schema supplied by compact/full_feature_columns.json.
Example:
    transformer = make_numeric_preprocessor(feature_columns)
    X_train = transformer.fit_transform(train[feature_columns])
    X_val   = transformer.transform(val[feature_columns])
    X_test  = transformer.transform(test[feature_columns])

Important: do NOT fit on validation/test/2026-27 fixtures; fit on train only.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


def make_numeric_preprocessor(feature_columns):
    if not feature_columns:
        raise ValueError('Supply predictor columns from feature_columns.json')
    # SimpleImputer keeps completely empty training features (fill zero),
    # though default pipeline should preferably drop unavailable features.
    # Using keep_empty_features requires sklearn >=1.2.
    return Pipeline([
        ('median',SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('drop_constant',VarianceThreshold(threshold=0.0)),
    ])


def split_seasons(df):
    train=df[df.season.between('2014-15','2023-24')].copy()
    valid=df[df.season=='2024-25'].copy()
    test=df[df.season=='2025-26'].copy()
    return train,valid,test
