"""Provisional preprocessing for UMAP splitting (unsupervised only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

META_COLUMNS = {"compound_id", "canonical_smiles", "activity", "original_row_index"}


def get_descriptor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def provisional_preprocess_for_umap(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Provisional unsupervised preprocessing for UMAP/clustering only.

    This pipeline is separate from the final supervised-model preprocessing.
    Activity is never used. Used only to create structural clusters for splitting.
    """
    desc_cols = get_descriptor_columns(df)
    X = df[desc_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    numeric_cols = []
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().any():
            X[col] = converted
            numeric_cols.append(col)
    X = X[numeric_cols]

    completely_missing = X.columns[X.isna().all()]
    X = X.drop(columns=completely_missing)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index
    )

    constant_cols = X_imputed.columns[X_imputed.std() == 0]
    X_imputed = X_imputed.drop(columns=constant_cols)

    if X_imputed.shape[1] == 0:
        raise ValueError("No descriptors remain after provisional preprocessing for UMAP.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_imputed, X_scaled
