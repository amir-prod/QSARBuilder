"""Tests for combined CV + validation scoring."""

import pytest

from qsar_agent.tools.combined_score import combined_r2


def test_combined_r2_equal_weight():
    assert combined_r2(0.4, 0.8) == pytest.approx(0.6)


def test_combined_r2_falls_back_to_cv_when_val_missing():
    assert combined_r2(0.55, None) == 0.55
