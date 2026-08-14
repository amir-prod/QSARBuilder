"""Tests for feature count selection."""

from qsar_agent.schemas.feature_selection import SFSResult, SFSResultRow
from qsar_agent.tools.feature_count_selection import select_feature_count_one_se_rule


def test_one_se_rule():
    results = [
        SFSResultRow(n_features=1, mean_train_r2=0.5, mean_cv_r2=0.4, std_cv_r2=0.05, selected_features=[]),
        SFSResultRow(n_features=2, mean_train_r2=0.6, mean_cv_r2=0.55, std_cv_r2=0.04, selected_features=[]),
        SFSResultRow(n_features=3, mean_train_r2=0.7, mean_cv_r2=0.58, std_cv_r2=0.03, selected_features=[]),
        SFSResultRow(n_features=4, mean_train_r2=0.75, mean_cv_r2=0.57, std_cv_r2=0.03, selected_features=[]),
    ]
    sfs = SFSResult(
        results=results,
        max_features_evaluated=4,
        results_csv_path="",
        selected_features_json_path="",
        plot_png_path="",
        plot_svg_path="",
    )
    sel = select_feature_count_one_se_rule(sfs)
    assert sel.best_feature_count == 3
    assert sel.selected_feature_count <= sel.best_feature_count


def test_one_se_rule_uses_combined_cv_and_val():
    results = [
        SFSResultRow(
            n_features=2,
            mean_train_r2=0.7,
            mean_cv_r2=0.60,
            std_cv_r2=0.02,
            selected_features=[],
            val_r2=0.20,
        ),
        SFSResultRow(
            n_features=3,
            mean_train_r2=0.72,
            mean_cv_r2=0.50,
            std_cv_r2=0.02,
            selected_features=[],
            val_r2=0.80,
        ),
    ]
    sfs = SFSResult(
        results=results,
        max_features_evaluated=2,
        results_csv_path="",
        selected_features_json_path="",
        plot_png_path="",
        plot_svg_path="",
    )
    sel = select_feature_count_one_se_rule(sfs)
    # combined: k=2 → 0.40; k=3 → 0.65
    assert sel.best_feature_count == 3
    assert sel.selected_feature_count == 3
    assert sel.selected_val_r2 == 0.80
