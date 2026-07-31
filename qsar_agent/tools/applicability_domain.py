"""Williams plot and applicability domain analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qsar_agent.schemas.applicability_domain import (
    ApplicabilityDomainResult,
    ApplicabilityDomainSummary,
)
from qsar_agent.services.artifact_manager import save_json
from qsar_agent.services.plotting import plot_williams
from qsar_agent.tools.descriptor_calculation import META_COLUMNS


def calculate_applicability_domain(
    train_path: str | Path,
    test_path: str | Path,
    predictions_path: str | Path,
    run_dir: Path,
    selected_features: list[str],
) -> ApplicabilityDomainResult:
    """
    Compute Williams plot diagnostics using leverage and standardized residuals.

    Note: Williams plots are classical descriptor-space diagnostics and may have
    limitations for nonlinear models such as RandomForest.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(predictions_path)

    X_train = train_df[selected_features].values
    X_test = test_df[selected_features].values

    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])

    XtX_inv = np.linalg.pinv(X_train_aug.T @ X_train_aug)

    def leverage(X_aug):
        h = np.array([x @ XtX_inv @ x for x in X_aug])
        return h

    h_train = leverage(X_train_aug)
    h_test = leverage(X_test_aug)

    n_train = len(X_train)
    p = len(selected_features)
    h_star = 3 * (p + 1) / n_train

    residuals = pred_df["residual"].values
    mse = np.mean(residuals**2)
    if mse == 0:
        std_residuals = np.zeros_like(residuals)
    else:
        std_residuals = residuals / np.sqrt(mse)

    train_id_to_lev = dict(zip(train_df["compound_id"], h_train))
    test_id_to_lev = dict(zip(test_df["compound_id"], h_test))

    records = []
    for i, row in pred_df.iterrows():
        if row["split"] == "train":
            lev = train_id_to_lev[row["compound_id"]]
        else:
            lev = test_id_to_lev[row["compound_id"]]

        high_leverage = lev > h_star
        response_outlier = abs(std_residuals[i]) > 3
        if high_leverage and response_outlier:
            ad_class = "outside_both"
        elif high_leverage:
            ad_class = "high_leverage"
        elif response_outlier:
            ad_class = "response_outlier"
        else:
            ad_class = "in_domain"

        records.append(
            {
                "compound_id": row["compound_id"],
                "split": row["split"],
                "leverage": float(lev),
                "standardized_residual": float(std_residuals[i]),
                "h_star": h_star,
                "applicability_domain": ad_class,
                "in_domain": ad_class == "in_domain",
            }
        )

    ad_df = pd.DataFrame(records)
    ad_path = run_dir / "applicability_domain.csv"
    ad_df.to_csv(ad_path, index=False)

    train_in = ad_df[(ad_df["split"] == "train") & ad_df["in_domain"]]
    test_in = ad_df[(ad_df["split"] == "test") & ad_df["in_domain"]]
    n_train = len(ad_df[ad_df["split"] == "train"])
    n_test = len(ad_df[ad_df["split"] == "test"])

    high_lev_ids = ad_df[ad_df["leverage"] > h_star]["compound_id"].tolist()
    outlier_ids = ad_df[ad_df["standardized_residual"].abs() > 3]["compound_id"].tolist()

    summary = ApplicabilityDomainSummary(
        train_in_domain_count=len(train_in),
        train_in_domain_pct=100 * len(train_in) / n_train if n_train else 0,
        test_in_domain_count=len(test_in),
        test_in_domain_pct=100 * len(test_in) / n_test if n_test else 0,
        warning_leverage=float(h_star),
        high_leverage_ids=high_lev_ids,
        response_outlier_ids=outlier_ids,
    )

    report_path = run_dir / "applicability_domain_report.json"
    save_json(
        report_path,
        {
            **summary.model_dump(),
            "note": (
                "Williams plot is a classical descriptor-space diagnostic; "
                "interpret with caution for nonlinear models."
            ),
        },
    )

    png_path = run_dir / "williams_plot.png"
    svg_path = run_dir / "williams_plot.svg"
    plot_williams(
        ad_df["leverage"].values,
        ad_df["standardized_residual"].values,
        ad_df["split"].values,
        h_star,
        png_path,
        svg_path,
    )

    return ApplicabilityDomainResult(
        summary=summary,
        classifications_path=str(ad_path),
        report_path=str(report_path),
        williams_png_path=str(png_path),
        williams_svg_path=str(svg_path),
    )
