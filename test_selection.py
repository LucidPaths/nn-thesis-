"""Tests for the cycle selection and LoRA promotion module."""

import json
import tempfile
from pathlib import Path

import pytest

from selection import (
    SelectionResult,
    load_cycle_results,
    get_best_accuracy,
    select_and_promote,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic cycle_results.json following CycleResults schema
# ---------------------------------------------------------------------------

def _make_cycle_results(cycle, best_accuracy, avg_accuracy=None, success_rate=1.0,
                        models_trained=150, total_generated=150, successful=142, novel=130):
    """Build a cycle_results dict matching the CycleResults.generate_cycle_results schema."""
    return {
        "cycle": cycle,
        "success": best_accuracy is not None,
        "training": {
            "data_dir": None,
            "total_examples": None,
            "new_examples_added": None,
            "training_time_minutes": None,
        },
        "generation": {
            "total_generated": total_generated,
            "successful": successful,
            "novel": novel,
        },
        "evaluation": {
            "models_trained": models_trained,
            "best_accuracy": best_accuracy,
            "avg_accuracy": avg_accuracy if avg_accuracy else (best_accuracy * 0.85 if best_accuracy else None),
            "success_rate": success_rate,
        },
        "cycle_time_minutes": 45.3,
    }


def _write_cycle_results(tmp_path, cycle_results):
    """Write a cycle_results dict to a temp JSON file and return its path."""
    p = tmp_path / "cycle_results.json"
    with open(p, "w") as f:
        json.dump(cycle_results, f)
    return p


# ---------------------------------------------------------------------------
# load_cycle_results / get_best_accuracy
# ---------------------------------------------------------------------------

class TestLoadCycleResults:
    def test_reads_valid_json(self, tmp_path):
        cr = _make_cycle_results(cycle=1, best_accuracy=0.82)
        path = _write_cycle_results(tmp_path, cr)
        loaded = load_cycle_results(path)
        assert loaded["cycle"] == 1
        assert loaded["evaluation"]["best_accuracy"] == 0.82

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cycle_results(tmp_path / "nonexistent.json")


class TestGetBestAccuracy:
    def test_extracts_accuracy(self):
        cr = _make_cycle_results(cycle=2, best_accuracy=0.91)
        assert get_best_accuracy(cr) == 0.91

    def test_returns_none_when_missing(self):
        assert get_best_accuracy({}) is None
        assert get_best_accuracy({"evaluation": {}}) is None

    def test_returns_none_when_null(self):
        cr = _make_cycle_results(cycle=3, best_accuracy=None)
        assert get_best_accuracy(cr) is None


# ---------------------------------------------------------------------------
# select_and_promote
# ---------------------------------------------------------------------------

class TestSelectAndPromote:
    """All tests use dry_run=True to avoid importing MergeLLM / torch."""

    def test_first_cycle_always_promotes(self, tmp_path):
        cr = _make_cycle_results(cycle=1, best_accuracy=0.78)
        path = _write_cycle_results(tmp_path, cr)

        result = select_and_promote(
            cycle_results_path=path,
            adapter_path="/fake/adapter",
            base_model_path="/fake/base",
            merged_output_path="/fake/merged",
            baseline_accuracy=None,
            dry_run=True,
        )
        assert result.promoted is True
        assert result.best_accuracy == 0.78
        assert result.baseline_accuracy is None
        assert result.delta is None  # no baseline to compare
        assert result.merged_path == "/fake/merged"

    def test_positive_delta_promotes(self, tmp_path):
        cr = _make_cycle_results(cycle=2, best_accuracy=0.85)
        path = _write_cycle_results(tmp_path, cr)

        result = select_and_promote(
            cycle_results_path=path,
            adapter_path="/fake/adapter",
            base_model_path="/fake/base",
            merged_output_path="/fake/merged",
            baseline_accuracy=0.78,
            dry_run=True,
        )
        assert result.promoted is True
        assert result.delta == pytest.approx(0.07)
        assert result.merged_path == "/fake/merged"

    def test_negative_delta_rejects(self, tmp_path):
        cr = _make_cycle_results(cycle=3, best_accuracy=0.75)
        path = _write_cycle_results(tmp_path, cr)

        result = select_and_promote(
            cycle_results_path=path,
            adapter_path="/fake/adapter",
            base_model_path="/fake/base",
            merged_output_path="/fake/merged",
            baseline_accuracy=0.85,
            dry_run=True,
        )
        assert result.promoted is False
        assert result.delta == pytest.approx(-0.10)
        assert result.merged_path is None

    def test_equal_accuracy_rejects(self, tmp_path):
        cr = _make_cycle_results(cycle=4, best_accuracy=0.85)
        path = _write_cycle_results(tmp_path, cr)

        result = select_and_promote(
            cycle_results_path=path,
            adapter_path="/fake/adapter",
            base_model_path="/fake/base",
            merged_output_path="/fake/merged",
            baseline_accuracy=0.85,
            dry_run=True,
        )
        assert result.promoted is False
        assert result.delta == pytest.approx(0.0)

    def test_no_evaluation_results_rejects(self, tmp_path):
        cr = _make_cycle_results(cycle=5, best_accuracy=None)
        path = _write_cycle_results(tmp_path, cr)

        result = select_and_promote(
            cycle_results_path=path,
            adapter_path="/fake/adapter",
            base_model_path="/fake/base",
            merged_output_path="/fake/merged",
            baseline_accuracy=0.80,
            dry_run=True,
        )
        assert result.promoted is False
        assert result.best_accuracy is None
        assert result.delta is None


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------

class TestSelectionHistory:
    def test_creates_history_file(self, tmp_path):
        cr = _make_cycle_results(cycle=1, best_accuracy=0.78)
        cr_path = _write_cycle_results(tmp_path, cr)
        history_path = tmp_path / "selection_history.json"

        select_and_promote(
            cycle_results_path=cr_path,
            adapter_path="/fake/adapter",
            base_model_path="/fake/base",
            merged_output_path="/fake/merged",
            baseline_accuracy=None,
            history_path=history_path,
            dry_run=True,
        )

        assert history_path.exists()
        history = json.loads(history_path.read_text())
        assert len(history) == 1
        assert history[0]["cycle"] == 1
        assert history[0]["promoted"] is True

    def test_appends_to_existing_history(self, tmp_path):
        (tmp_path / "c1").mkdir()
        cr_path_1 = _write_cycle_results(tmp_path / "c1", _make_cycle_results(1, 0.78))

        history_path = tmp_path / "selection_history.json"

        select_and_promote(cr_path_1, "/a", "/b", "/m", None, history_path, dry_run=True)

        (tmp_path / "c2").mkdir()
        cr_path_2 = _write_cycle_results(tmp_path / "c2", _make_cycle_results(2, 0.85))

        select_and_promote(cr_path_2, "/a2", "/b", "/m2", 0.78, history_path, dry_run=True)

        history = json.loads(history_path.read_text())
        assert len(history) == 2
        assert history[0]["cycle"] == 1
        assert history[1]["cycle"] == 2
        assert history[1]["promoted"] is True
        assert history[1]["delta"] == pytest.approx(0.07)

    def test_no_history_when_path_is_none(self, tmp_path):
        cr = _make_cycle_results(cycle=1, best_accuracy=0.78)
        cr_path = _write_cycle_results(tmp_path, cr)

        result = select_and_promote(
            cr_path, "/a", "/b", "/m", None, history_path=None, dry_run=True
        )
        assert result.promoted is True
        # No history file should exist
        assert not (tmp_path / "selection_history.json").exists()


# ---------------------------------------------------------------------------
# Multi-cycle simulation
# ---------------------------------------------------------------------------

class TestMultiCycleSimulation:
    """Simulate a realistic 5-cycle pipeline with mixed promotions/rejections."""

    def test_five_cycle_pipeline(self, tmp_path):
        cycle_accuracies = [0.42, 0.51, 0.48, 0.55, 0.57]
        #                    ^promote ^promote ^reject ^promote ^promote
        history_path = tmp_path / "history.json"
        baseline = None

        for i, acc in enumerate(cycle_accuracies):
            cycle = i + 1
            cr_path = tmp_path / f"cycle_{cycle}" / "cycle_results.json"
            cr_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cr_path, "w") as f:
                json.dump(_make_cycle_results(cycle, acc), f)

            result = select_and_promote(
                cr_path, f"/adapter/{cycle}", "/base", f"/merged/{cycle}",
                baseline_accuracy=baseline,
                history_path=history_path,
                dry_run=True,
            )

            if result.promoted:
                baseline = result.best_accuracy

        history = json.loads(history_path.read_text())
        assert len(history) == 5

        promotions = [h["promoted"] for h in history]
        assert promotions == [True, True, False, True, True]

        # Baseline should be the last promoted accuracy
        assert baseline == pytest.approx(0.57)
