"""The specialist agents of the QSAR workflow.

Each role owns one stage and calls only deterministic tools. They hold no LLM
reference: judgement about *what to try next* belongs to
:mod:`qsar_agent.agents.strategist`, which keeps the measurement path free of
anything a model could influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from qsar_agent.schemas.agentic import (
    ADSummary,
    CVSummary,
    FeatureRecipe,
    IterationPlan,
    SplitMethod,
    TaskDetection,
    ValidationReport,
)
from qsar_agent.tools.ad_distance import (
    evaluate_applicability_domain,
    training_outlier_mask,
)
from qsar_agent.tools.dataset_ingest import IngestResult, ingest_dataset
from qsar_agent.tools.metrics import CV_SCORER, compute_metrics, primary_metric_name
from qsar_agent.tools.model_zoo import build_estimator, decision_scores
from qsar_agent.tools.rdkit_features import compute_features
from qsar_agent.tools.splitting import make_split
from qsar_agent.tools.y_scrambling import run_y_scrambling


@dataclass
class DatasetBundle:
    """The split dataset every later stage works from."""

    frame: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray
    task: str
    task_detection: TaskDetection
    split_method: str
    ingest: IngestResult

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def smiles(self) -> list[str]:
        return self.frame["canonical_smiles"].tolist()

    def compound_ids(self, indices: np.ndarray) -> list[str]:
        return self.frame.iloc[indices]["compound_id"].astype(str).tolist()

    def summary(self) -> dict:
        activity = self.frame["activity"]
        info = {
            "n_compounds": int(len(self.frame)),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "split_method": self.split_method,
            "task": self.task,
            "task_detection": self.task_detection.model_dump(),
        }
        if self.task == "regression":
            numeric = pd.to_numeric(activity)
            info["activity_stats"] = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "std": float(numeric.std()),
            }
        else:
            info["class_counts"] = {
                str(k): int(v) for k, v in activity.value_counts().items()
            }
        return info


@dataclass
class FeatureMatrices:
    """Train/test matrices built from one recipe, with preprocessing fitted on train only."""

    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_raw_features: int
    dropped_outlier_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class DataAgent:
    """Ingests the CSV, characterises the endpoint, and splits the data."""

    def __init__(self, run_dir: Path, random_seed: int = 42):
        self.run_dir = run_dir
        self.random_seed = random_seed

    def prepare(
        self,
        dataset_path: str | Path,
        smiles_column: str,
        activity_column: str,
        id_column: str | None = None,
        forced_task: str | None = None,
        split_method: SplitMethod = "random",
        test_size: float = 0.2,
        min_compounds: int = 20,
    ) -> DatasetBundle:
        ingest = ingest_dataset(
            dataset_path=dataset_path,
            smiles_column=smiles_column,
            activity_column=activity_column,
            run_dir=self.run_dir,
            id_column=id_column,
            forced_task=forced_task,
            min_compounds=min_compounds,
        )
        frame = pd.read_csv(ingest.cleaned_dataset_path)
        task = ingest.task_detection.task
        if task == "classification":
            frame["activity"] = frame["activity"].astype(str)

        train_idx, test_idx, method_used = make_split(
            method=split_method,
            frame=frame,
            task=task,
            test_size=test_size,
            random_seed=self.random_seed,
        )

        assignments = pd.DataFrame(
            {
                "compound_id": frame["compound_id"],
                "canonical_smiles": frame["canonical_smiles"],
                "activity": frame["activity"],
                "split": ["train"] * len(frame),
            }
        )
        assignments.loc[test_idx, "split"] = "test"
        assignments.to_csv(self.run_dir / "agentic_split_assignments.csv", index=False)

        return DatasetBundle(
            frame=frame,
            train_idx=train_idx,
            test_idx=test_idx,
            task=task,
            task_detection=ingest.task_detection,
            split_method=method_used,
            ingest=ingest,
        )


class DescriptorAgent:
    """Turns a feature recipe into train/test matrices.

    Every preprocessing decision — imputation values, retained columns, scaling
    parameters, univariate ranking — is learned from the training rows only and
    then applied unchanged to the test rows.
    """

    def __init__(self, bundle: DatasetBundle):
        self.bundle = bundle
        self._raw_cache: dict[str, pd.DataFrame] = {}

    def _raw_features(self, recipe: FeatureRecipe) -> pd.DataFrame:
        key = "+".join(recipe.blocks) + f"|{recipe.fingerprint_bits}|{recipe.fingerprint_radius}"
        if key not in self._raw_cache:
            self._raw_cache[key] = compute_features(self.bundle.smiles(), recipe)
        return self._raw_cache[key]

    def build(self, recipe: FeatureRecipe, y_train_encoded: np.ndarray | None = None) -> FeatureMatrices:
        raw = self._raw_features(recipe)
        n_raw = raw.shape[1]
        notes: list[str] = []

        train_idx = np.asarray(self.bundle.train_idx)
        test_idx = np.asarray(self.bundle.test_idx)
        dropped_ids: list[str] = []

        train_raw = raw.iloc[train_idx]
        test_raw = raw.iloc[test_idx]

        # Drop columns that are entirely missing in training; they carry nothing.
        usable = train_raw.columns[train_raw.notna().any()].tolist()
        if len(usable) < n_raw:
            notes.append(f"Dropped {n_raw - len(usable)} all-missing descriptors.")
        train_raw = train_raw[usable]
        test_raw = test_raw[usable]

        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_train = imputer.fit_transform(train_raw)
        X_test = imputer.transform(test_raw)
        names = list(usable)

        X_train, X_test, names, note = self._drop_near_constant(
            X_train, X_test, names, recipe.variance_threshold
        )
        if note:
            notes.append(note)

        if recipe.correlation_threshold is not None:
            X_train, X_test, names, note = self._drop_correlated(
                X_train, X_test, names, recipe.correlation_threshold
            )
            if note:
                notes.append(note)

        if recipe.scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if recipe.univariate_top_k is not None and y_train_encoded is not None:
            X_train, X_test, names, note = self._select_top_k(
                X_train, X_test, names, y_train_encoded, recipe.univariate_top_k
            )
            if note:
                notes.append(note)

        if recipe.drop_ad_outliers and len(X_train) > 10:
            mask = training_outlier_mask(X_train)
            if mask.any() and mask.sum() < len(X_train) // 4:
                dropped_ids = self.bundle.compound_ids(train_idx[mask])
                train_idx = train_idx[~mask]
                X_train = X_train[~mask]
                notes.append(
                    f"Dropped {int(mask.sum())} training compounds outside the k-NN domain."
                )
            elif mask.any():
                notes.append(
                    f"Kept all training compounds: {int(mask.sum())} flagged as domain outliers "
                    "would have removed too much of the training set."
                )

        if not names:
            raise ValueError(
                "Feature preprocessing removed every descriptor. Loosen the variance or "
                "correlation filters, or choose a different feature block."
            )

        return FeatureMatrices(
            X_train=np.asarray(X_train, dtype=float),
            X_test=np.asarray(X_test, dtype=float),
            feature_names=names,
            train_idx=train_idx,
            test_idx=test_idx,
            n_raw_features=n_raw,
            dropped_outlier_ids=dropped_ids,
            notes=notes,
        )

    @staticmethod
    def _drop_near_constant(
        X_train: np.ndarray, X_test: np.ndarray, names: list[str], threshold: float
    ) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        std = X_train.std(axis=0)
        keep = std > threshold
        if keep.all():
            return X_train, X_test, names, ""
        if not keep.any():
            raise ValueError(
                f"Every descriptor has standard deviation <= {threshold} on the training set."
            )
        dropped = int((~keep).sum())
        kept_names = [n for n, k in zip(names, keep) if k]
        return (
            X_train[:, keep],
            X_test[:, keep],
            kept_names,
            f"Dropped {dropped} near-constant descriptors (std <= {threshold}).",
        )

    @staticmethod
    def _drop_correlated(
        X_train: np.ndarray, X_test: np.ndarray, names: list[str], threshold: float
    ) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        if X_train.shape[1] < 2:
            return X_train, X_test, names, ""
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X_train, rowvar=False))
        corr = np.nan_to_num(corr)
        keep = np.ones(X_train.shape[1], dtype=bool)
        for i in range(X_train.shape[1]):
            if not keep[i]:
                continue
            duplicates = np.where(corr[i, i + 1 :] > threshold)[0] + i + 1
            keep[duplicates] = False
        if keep.all():
            return X_train, X_test, names, ""
        dropped = int((~keep).sum())
        kept_names = [n for n, k in zip(names, keep) if k]
        return (
            X_train[:, keep],
            X_test[:, keep],
            kept_names,
            f"Dropped {dropped} descriptors correlated above |r| > {threshold}.",
        )

    @staticmethod
    def _select_top_k(
        X_train: np.ndarray,
        X_test: np.ndarray,
        names: list[str],
        y_train: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str], str]:
        if top_k >= X_train.shape[1]:
            return X_train, X_test, names, ""
        score_func = f_classif if len(np.unique(y_train)) < len(y_train) / 2 else f_regression
        selector = SelectKBest(score_func=score_func, k=top_k)
        with np.errstate(invalid="ignore", divide="ignore"):
            X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        support = selector.get_support()
        kept_names = [n for n, k in zip(names, support) if k]
        return (
            X_train_sel,
            X_test_sel,
            kept_names,
            f"Kept the top {top_k} descriptors by univariate F-score "
            f"(from {X_train.shape[1]}).",
        )


class ModelingAgent:
    """Instantiates and fits the planned estimator, with cross-validation."""

    def __init__(self, task: str, random_seed: int = 42, cv_folds: int = 5, n_jobs: int = 1):
        self.task = task
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.primary_metric = primary_metric_name(task)

    def build(self, plan: IterationPlan) -> BaseEstimator:
        return build_estimator(
            task=self.task,
            estimator=plan.model.estimator,
            hyperparameters=plan.model.hyperparameters,
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
        )

    def effective_folds(self, y_train: np.ndarray) -> int:
        """Shrink the fold count when a class or the dataset cannot support it."""
        folds = min(self.cv_folds, len(y_train))
        if self.task == "classification":
            _, counts = np.unique(y_train, return_counts=True)
            folds = min(folds, int(counts.min()))
        return max(2, folds)

    def splitter(self, y_train: np.ndarray):
        folds = self.effective_folds(y_train)
        if self.task == "classification":
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_seed)
        return KFold(n_splits=folds, shuffle=True, random_state=self.random_seed)

    def cross_validate(
        self, estimator: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray
    ) -> CVSummary:
        """Score every fold on both its training and validation part.

        Both halves are needed because the train-minus-CV gap is the loop's
        overfitting signal, and ``cross_val_score`` alone would not give it.
        """
        splitter = self.splitter(y_train)
        scorer_name = CV_SCORER[self.task]
        train_scores: list[float] = []
        cv_scores: list[float] = []

        for fit_idx, score_idx in splitter.split(X_train, y_train):
            model = clone(estimator)
            model.fit(X_train[fit_idx], y_train[fit_idx])
            train_scores.append(self._score(model, X_train[fit_idx], y_train[fit_idx]))
            cv_scores.append(self._score(model, X_train[score_idx], y_train[score_idx]))

        finite_cv = [s for s in cv_scores if np.isfinite(s)]
        finite_train = [s for s in train_scores if np.isfinite(s)]
        mean_cv = float(np.mean(finite_cv)) if finite_cv else float("nan")
        mean_train = float(np.mean(finite_train)) if finite_train else float("nan")

        return CVSummary(
            folds=splitter.get_n_splits(),
            primary_metric=scorer_name,
            mean_train_score=mean_train,
            mean_cv_score=mean_cv,
            std_cv_score=float(np.std(finite_cv, ddof=0)) if finite_cv else float("nan"),
            train_cv_gap=mean_train - mean_cv,
            fold_scores=[float(s) for s in cv_scores],
        )

    def _score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        predictions = model.predict(X)
        scores = decision_scores(model, X) if self.task == "classification" else None
        metrics = compute_metrics(self.task, y, predictions, scores)
        return metrics.get(self.primary_metric, float("nan"))


class ValidationAgent:
    """Measures everything the acceptance criteria can be checked against."""

    def __init__(
        self,
        task: str,
        random_seed: int = 42,
        scramble_repeats: int = 10,
        ad_k: int = 5,
        ad_z_factor: float = 3.0,
        n_jobs: int = 1,
    ):
        self.task = task
        self.random_seed = random_seed
        self.scramble_repeats = scramble_repeats
        self.ad_k = ad_k
        self.ad_z_factor = ad_z_factor
        self.n_jobs = n_jobs
        self.primary_metric = primary_metric_name(task)

    def validate(
        self,
        modeling_agent: ModelingAgent,
        estimator: BaseEstimator,
        matrices: FeatureMatrices,
        y_train: np.ndarray,
        y_test: np.ndarray,
        test_ids: list[str],
    ) -> tuple[ValidationReport, BaseEstimator]:
        warnings: list[str] = list(matrices.notes)

        cv = modeling_agent.cross_validate(estimator, matrices.X_train, y_train)

        fitted = clone(estimator)
        fitted.fit(matrices.X_train, y_train)

        train_pred = fitted.predict(matrices.X_train)
        test_pred = fitted.predict(matrices.X_test)
        train_scores = (
            decision_scores(fitted, matrices.X_train) if self.task == "classification" else None
        )
        test_scores = (
            decision_scores(fitted, matrices.X_test) if self.task == "classification" else None
        )

        train_metrics = compute_metrics(self.task, y_train, train_pred, train_scores)
        test_metrics = compute_metrics(self.task, y_test, test_pred, test_scores)

        scrambling = None
        try:
            scrambling = run_y_scrambling(
                X=matrices.X_train,
                y=y_train,
                estimator=estimator,
                task=self.task,
                real_cv_score=cv.mean_cv_score,
                n_repeats=self.scramble_repeats,
                cv_folds=modeling_agent.effective_folds(y_train),
                random_seed=self.random_seed,
                n_jobs=self.n_jobs,
            )
        except ValueError as exc:
            warnings.append(f"y-scrambling skipped: {exc}")

        domain: ADSummary | None = None
        try:
            domain, _mask = evaluate_applicability_domain(
                X_train=matrices.X_train,
                X_test=matrices.X_test,
                test_ids=test_ids,
                k=self.ad_k,
                z_factor=self.ad_z_factor,
            )
        except ValueError as exc:
            warnings.append(f"Applicability domain skipped: {exc}")

        report = ValidationReport(
            task=self.task,
            primary_metric=self.primary_metric,
            n_train=int(len(y_train)),
            n_test=int(len(y_test)),
            n_features=len(matrices.feature_names),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            cv=cv,
            y_scrambling=scrambling,
            applicability_domain=domain,
            warnings=warnings,
        )
        return report, fitted


def encode_targets(
    task: str, activity: pd.Series
) -> tuple[np.ndarray, LabelEncoder | None]:
    """Return the model-ready target vector, with the encoder for classification."""
    if task == "classification":
        encoder = LabelEncoder()
        return encoder.fit_transform(activity.astype(str)), encoder
    return pd.to_numeric(activity).to_numpy(dtype=float), None
