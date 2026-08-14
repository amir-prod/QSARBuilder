"""Train-only descriptor preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from qsar_agent.config import PreprocessingConfig
from qsar_agent.schemas.preprocessing import PreprocessingResult
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.tools.descriptor_calculation import META_COLUMNS


class DescriptorPreprocessor(BaseEstimator, TransformerMixin):
    """Fitted on training data only; transforms train, val, and test consistently."""

    def __init__(
        self,
        missing_threshold: float = 0.20,
        near_constant_std: float = 0.01,
        correlation_threshold: float = 0.95,
    ):
        self.missing_threshold = missing_threshold
        self.near_constant_std = near_constant_std
        self.correlation_threshold = correlation_threshold
        self.retained_columns_: list[str] = []
        self.imputer_input_columns_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.removed_records_: list[dict] = []
        self.imputation_values_: dict[str, float] = {}
        self.scaler_means_: dict[str, float] = {}
        self.scaler_scales_: dict[str, float] = {}

    def _descriptor_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in META_COLUMNS]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DescriptorPreprocessor":
        self.removed_records_ = []
        desc_cols = self._descriptor_cols(X)
        Xt = X[desc_cols].copy().replace([np.inf, -np.inf], np.nan)

        numeric_cols = []
        for col in Xt.columns:
            converted = pd.to_numeric(Xt[col], errors="coerce")
            if converted.notna().any():
                Xt[col] = converted
                numeric_cols.append(col)
            else:
                self.removed_records_.append({"descriptor": col, "reason": "nonnumeric"})
        Xt = Xt[numeric_cols]

        missing_frac = Xt.isna().mean()
        drop_missing = missing_frac[missing_frac > self.missing_threshold].index.tolist()
        for col in drop_missing:
            self.removed_records_.append(
                {"descriptor": col, "reason": f"missing>{self.missing_threshold}"}
            )
        Xt = Xt.drop(columns=drop_missing)
        self.imputer_input_columns_ = Xt.columns.tolist()

        self.imputer_ = SimpleImputer(strategy="median")
        Xt_imp = pd.DataFrame(
            self.imputer_.fit_transform(Xt), columns=self.imputer_input_columns_, index=Xt.index
        )

        constant_cols = Xt_imp.columns[Xt_imp.std() == 0].tolist()
        for col in constant_cols:
            self.removed_records_.append({"descriptor": col, "reason": "constant"})
        Xt_imp = Xt_imp.drop(columns=constant_cols)

        near_const = Xt_imp.columns[Xt_imp.std() < self.near_constant_std].tolist()
        for col in near_const:
            self.removed_records_.append(
                {"descriptor": col, "reason": f"near_constant_std<{self.near_constant_std}"}
            )
        Xt_imp = Xt_imp.drop(columns=near_const)

        if y is not None:
            activity = pd.to_numeric(y, errors="coerce")
        else:
            activity = None

        corr = Xt_imp.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop: set[str] = set()
        pairs = []
        for col in upper.columns:
            for row in upper.index:
                if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > self.correlation_threshold:
                    pairs.append((row, col, upper.loc[row, col]))

        pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
        for a, b, r in pairs:
            if a in to_drop or b in to_drop:
                continue
            miss_a = Xt_imp[a].isna().sum()
            miss_b = Xt_imp[b].isna().sum()
            if miss_a != miss_b:
                drop = a if miss_a > miss_b else b
            elif activity is not None:
                corr_a = abs(Xt_imp[a].corr(activity))
                corr_b = abs(Xt_imp[b].corr(activity))
                if corr_a != corr_b:
                    drop = a if corr_a < corr_b else b
                else:
                    drop = max(a, b)
            else:
                drop = max(a, b)
            to_drop.add(drop)
            self.removed_records_.append(
                {"descriptor": drop, "reason": f"correlated_with_{a if drop == b else b}", "r": r}
            )

        Xt_imp = Xt_imp.drop(columns=list(to_drop))

        if Xt_imp.shape[1] == 0:
            raise ValueError("No descriptors remain after preprocessing.")

        self.scaler_ = StandardScaler()
        self.scaler_.fit(Xt_imp)
        self.retained_columns_ = Xt_imp.columns.tolist()
        self.imputation_values_ = {
            col: float(self.imputer_.statistics_[self.imputer_input_columns_.index(col)])
            for col in self.retained_columns_
        }
        self.scaler_means_ = dict(zip(self.retained_columns_, self.scaler_.mean_))
        self.scaler_scales_ = dict(zip(self.retained_columns_, self.scaler_.scale_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        meta = X[list(META_COLUMNS)].copy()
        desc_cols = self._descriptor_cols(X)
        Xt = X[desc_cols].copy().replace([np.inf, -np.inf], np.nan)
        for col in self.imputer_input_columns_:
            if col not in Xt.columns:
                Xt[col] = np.nan
        Xt = Xt[self.imputer_input_columns_]
        for col in Xt.columns:
            Xt[col] = pd.to_numeric(Xt[col], errors="coerce")
        Xt_imp = pd.DataFrame(
            self.imputer_.transform(Xt),
            columns=self.imputer_input_columns_,
            index=Xt.index,
        )
        Xt_final = Xt_imp[self.retained_columns_]
        Xt_scaled = pd.DataFrame(
            self.scaler_.transform(Xt_final),
            columns=self.retained_columns_,
            index=Xt.index,
        )
        return pd.concat([meta.reset_index(drop=True), Xt_scaled.reset_index(drop=True)], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


def _count_reasons(removed: list[dict]) -> dict[str, int]:
    counts = {
        "nonnumeric": 0,
        "missing": 0,
        "constant": 0,
        "near_constant": 0,
        "correlated": 0,
    }
    for r in removed:
        reason = r["reason"]
        if reason == "nonnumeric":
            counts["nonnumeric"] += 1
        elif reason.startswith("missing"):
            counts["missing"] += 1
        elif reason == "constant":
            counts["constant"] += 1
        elif reason.startswith("near_constant"):
            counts["near_constant"] += 1
        elif reason.startswith("correlated"):
            counts["correlated"] += 1
    return counts


def fit_descriptor_preprocessor(
    train_path: str | Path,
    test_path: str | Path,
    run_dir: Path,
    preprocessing_config: PreprocessingConfig | None = None,
    val_path: str | Path | None = None,
) -> PreprocessingResult:
    """Fit preprocessing on training set only; apply to train, val, and test."""
    cfg = preprocessing_config or PreprocessingConfig()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path) if val_path is not None else None
    if len(train_df) == 0:
        raise ValueError(
            "Training descriptor set is empty; check UMAP split and descriptor calculation."
        )
    if len(test_df) == 0:
        raise ValueError(
            "External test descriptor set is empty; check the train/val/test split "
            "(small clusters were previously assigned entirely to train)."
        )
    if val_df is not None and len(val_df) == 0:
        raise ValueError(
            "Validation descriptor set is empty; check the train/val/test split "
            "(small clusters were previously assigned entirely to train)."
        )

    initial_count = len([c for c in train_df.columns if c not in META_COLUMNS])

    preprocessor = DescriptorPreprocessor(
        missing_threshold=cfg.missing_value_threshold,
        near_constant_std=cfg.near_constant_std_threshold,
        correlation_threshold=cfg.correlation_threshold,
    )
    train_pp = preprocessor.fit_transform(train_df, y=train_df["activity"])
    test_pp = preprocessor.transform(test_df)
    val_pp = preprocessor.transform(val_df) if val_df is not None else None

    train_out = run_dir / "preprocessed_train_descriptors.csv"
    val_out = run_dir / "preprocessed_val_descriptors.csv"
    test_out = run_dir / "preprocessed_test_descriptors.csv"
    train_pp.to_csv(train_out, index=False)
    test_pp.to_csv(test_out, index=False)
    if val_pp is not None:
        val_pp.to_csv(val_out, index=False)
    else:
        val_out = train_out

    preproc_path = run_dir / "descriptor_preprocessor.joblib"
    joblib.dump(preprocessor, preproc_path)

    removed_path = run_dir / "removed_descriptors.csv"
    pd.DataFrame(preprocessor.removed_records_).to_csv(removed_path, index=False)

    retained_path = run_dir / "retained_descriptors.json"
    with open(retained_path, "w", encoding="utf-8") as f:
        json.dump({"retained_descriptors": preprocessor.retained_columns_}, f, indent=2)

    reason_counts = _count_reasons(preprocessor.removed_records_)
    report = {
        "initial_descriptor_count": initial_count,
        "removed_nonnumeric": reason_counts["nonnumeric"],
        "removed_missing": reason_counts["missing"],
        "removed_constant": reason_counts["constant"],
        "removed_near_constant": reason_counts["near_constant"],
        "removed_correlated": reason_counts["correlated"],
        "final_descriptor_count": len(preprocessor.retained_columns_),
        "missing_value_threshold": cfg.missing_value_threshold,
        "near_constant_threshold": cfg.near_constant_std_threshold,
        "correlation_threshold": cfg.correlation_threshold,
        "imputation_values": preprocessor.imputation_values_,
        "scaler_means": preprocessor.scaler_means_,
        "scaler_scales": preprocessor.scaler_scales_,
        "methodological_note": (
            "Near-constant filtering occurs before StandardScaler because scaling "
            "would mask near-zero variance descriptors."
        ),
    }
    report_path = run_dir / "descriptor_preprocessing_report.json"
    save_json(report_path, report)

    return PreprocessingResult(
        initial_descriptor_count=initial_count,
        removed_nonnumeric=reason_counts["nonnumeric"],
        removed_missing=reason_counts["missing"],
        removed_constant=reason_counts["constant"],
        removed_near_constant=reason_counts["near_constant"],
        removed_correlated=reason_counts["correlated"],
        final_descriptor_count=len(preprocessor.retained_columns_),
        missing_value_threshold=cfg.missing_value_threshold,
        near_constant_threshold=cfg.near_constant_std_threshold,
        correlation_threshold=cfg.correlation_threshold,
        preprocessed_train_path=str(train_out),
        preprocessed_val_path=str(val_out),
        preprocessed_test_path=str(test_out),
        preprocessor_path=str(preproc_path),
        preprocessing_report_path=str(report_path),
        removed_descriptors_path=str(removed_path),
        retained_descriptors_path=str(retained_path),
    )
