"""Tests for cycle selection and LoRA promotion."""

import json

import pytest

from selection import select_and_promote


def _write_cycle_results(tmp_path, cycle, best_accuracy):
    """Write a minimal cycle_results.json and return its path."""
    p = tmp_path / "cycle_results.json"
    p.write_text(json.dumps({
        "cycle": cycle,
        "evaluation": {"best_accuracy": best_accuracy},
    }))
    return p


class TestSelectAndPromote:
    """All tests use dry_run=True to avoid importing MergeLLM / torch."""

    def test_first_cycle_always_promotes(self, tmp_path):
        path = _write_cycle_results(tmp_path, 1, 0.78)
        result = select_and_promote(path, "/fake/adapter", "/fake/merged", dry_run=True)
        assert result.promoted is True
        assert result.best_accuracy == 0.78
        assert result.delta is None

    def test_positive_delta_promotes(self, tmp_path):
        path = _write_cycle_results(tmp_path, 2, 0.85)
        result = select_and_promote(path, "/fake/adapter", "/fake/merged",
                                    baseline_accuracy=0.78, dry_run=True)
        assert result.promoted is True
        assert result.delta == pytest.approx(0.07)

    def test_negative_delta_rejects(self, tmp_path):
        path = _write_cycle_results(tmp_path, 3, 0.75)
        result = select_and_promote(path, "/fake/adapter", "/fake/merged",
                                    baseline_accuracy=0.85, dry_run=True)
        assert result.promoted is False
        assert result.delta == pytest.approx(-0.10)

    def test_equal_accuracy_rejects(self, tmp_path):
        path = _write_cycle_results(tmp_path, 4, 0.85)
        result = select_and_promote(path, "/fake/adapter", "/fake/merged",
                                    baseline_accuracy=0.85, dry_run=True)
        assert result.promoted is False
        assert result.delta == pytest.approx(0.0)

    def test_no_evaluation_rejects(self, tmp_path):
        path = _write_cycle_results(tmp_path, 5, None)
        result = select_and_promote(path, "/fake/adapter", "/fake/merged",
                                    baseline_accuracy=0.80, dry_run=True)
        assert result.promoted is False
        assert result.best_accuracy is None
        assert result.delta is None

    def test_history_file_written(self, tmp_path):
        path = _write_cycle_results(tmp_path, 1, 0.78)
        history_path = tmp_path / "history.json"
        select_and_promote(path, "/fake/adapter", "/fake/merged",
                           history_path=history_path, dry_run=True)
        history = json.loads(history_path.read_text())
        assert len(history) == 1
        assert history[0]["promoted"] is True


class TestMultiCycleSimulation:
    def test_five_cycle_pipeline(self, tmp_path):
        accuracies = [0.42, 0.51, 0.48, 0.55, 0.57]
        baseline = None
        promotions = []

        for i, acc in enumerate(accuracies):
            cycle_dir = tmp_path / f"cycle_{i + 1}"
            cycle_dir.mkdir()
            path = _write_cycle_results(cycle_dir, i + 1, acc)
            result = select_and_promote(path, f"/adapter/{i + 1}", f"/merged/{i + 1}",
                                        baseline_accuracy=baseline, dry_run=True)
            promotions.append(result.promoted)
            if result.promoted:
                baseline = result.best_accuracy

        assert promotions == [True, True, False, True, True]
        assert baseline == pytest.approx(0.57)
