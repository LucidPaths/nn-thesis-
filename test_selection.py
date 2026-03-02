"""Tests for cycle selection, LoRA promotion, and resolve_cycle decision logic."""

import json
import time

import pytest

from selection import select_and_promote
from resolve_cycle import decide, load_metrics, load_lineage, save_lineage, get_cycle_files
import resolve_cycle


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


class TestProgressiveBase:
    """Tests for _resolve_base and progressive merging across cycles."""

    def test_effective_base_is_original_when_no_merged_exists(self, tmp_path):
        """When merged_output_path doesn't exist, effective_base = base_model_path."""
        path = _write_cycle_results(tmp_path, 1, 0.78)
        result = select_and_promote(path, "/fake/adapter",
                                    tmp_path / "merged_out",
                                    base_model_path="/original/base",
                                    dry_run=True)
        assert result.effective_base == "/original/base"

    def test_effective_base_is_merged_when_prior_promotion_exists(self, tmp_path):
        """When merged_output_path already has content, effective_base = merged_output_path."""
        merged_dir = tmp_path / "merged_out"
        merged_dir.mkdir()
        (merged_dir / "model.safetensors").write_text("fake")

        path = _write_cycle_results(tmp_path, 2, 0.85)
        result = select_and_promote(path, "/fake/adapter",
                                    str(merged_dir),
                                    base_model_path="/original/base",
                                    baseline_accuracy=0.78,
                                    dry_run=True)
        assert result.effective_base == str(merged_dir)

    def test_effective_base_stays_original_when_merged_dir_empty(self, tmp_path):
        """An empty merged dir is not considered a prior promotion."""
        merged_dir = tmp_path / "merged_out"
        merged_dir.mkdir()

        path = _write_cycle_results(tmp_path, 1, 0.78)
        result = select_and_promote(path, "/fake/adapter",
                                    str(merged_dir),
                                    base_model_path="/original/base",
                                    dry_run=True)
        assert result.effective_base == "/original/base"

    def test_multi_cycle_progressive_base_tracking(self, tmp_path):
        """Simulate 3 cycles: promote, reject, promote — verify effective_base progression."""
        merged_dir = tmp_path / "merged_out"
        base = "/original/base"

        # Cycle 1: promoted (merged_dir doesn't exist yet → uses original base)
        (tmp_path / "c1").mkdir()
        path = _write_cycle_results(tmp_path / "c1", 1, 0.50)
        r1 = select_and_promote(path, "/adapter/1", str(merged_dir),
                                base_model_path=base, dry_run=True)
        assert r1.promoted is True
        assert r1.effective_base == base

        # Simulate what merge() would do: populate merged_dir
        merged_dir.mkdir()
        (merged_dir / "model.safetensors").write_text("fake_merged_1")

        # Cycle 2: rejected (merged_dir exists → effective_base points there)
        (tmp_path / "c2").mkdir()
        path = _write_cycle_results(tmp_path / "c2", 2, 0.45)
        r2 = select_and_promote(path, "/adapter/2", str(merged_dir),
                                base_model_path=base,
                                baseline_accuracy=r1.best_accuracy,
                                dry_run=True)
        assert r2.promoted is False
        assert r2.effective_base == str(merged_dir)

        # Cycle 3: promoted (merged_dir still has cycle 1's output → stacks on top)
        (tmp_path / "c3").mkdir()
        path = _write_cycle_results(tmp_path / "c3", 3, 0.60)
        r3 = select_and_promote(path, "/adapter/3", str(merged_dir),
                                base_model_path=base,
                                baseline_accuracy=r1.best_accuracy,
                                dry_run=True)
        assert r3.promoted is True
        assert r3.effective_base == str(merged_dir)


class TestCycleResultsArchival:
    """Tests for _archive_cycle_results."""

    def test_archive_file_created(self, tmp_path):
        """Each cycle's results are archived to cycle_{N}_results.json."""
        path = _write_cycle_results(tmp_path, 3, 0.82)
        select_and_promote(path, "/fake/adapter", "/fake/merged", dry_run=True)
        archive = tmp_path / "cycle_3_results.json"
        assert archive.exists()
        data = json.loads(archive.read_text())
        assert data["cycle"] == 3
        assert data["evaluation"]["best_accuracy"] == 0.82

    def test_archive_not_created_when_cycle_missing(self, tmp_path):
        """If cycle number is absent from results, no archive is created."""
        p = tmp_path / "cycle_results.json"
        p.write_text(json.dumps({"evaluation": {"best_accuracy": 0.50}}))
        select_and_promote(p, "/fake/adapter", "/fake/merged", dry_run=True)
        archives = list(tmp_path.glob("cycle_*_results.json"))
        assert archives == []

    def test_five_cycle_archives_all_preserved(self, tmp_path):
        """Run 5 cycles writing to the same path; all 5 archives survive."""
        results_path = tmp_path / "cycle_results.json"
        baseline = None
        for i in range(5):
            acc = 0.40 + i * 0.05
            results_path.write_text(json.dumps({
                "cycle": i,
                "evaluation": {"best_accuracy": acc},
            }))
            r = select_and_promote(results_path, f"/adapter/{i}", "/fake/merged",
                                   baseline_accuracy=baseline, dry_run=True)
            if r.promoted:
                baseline = r.best_accuracy
        for i in range(5):
            archive = tmp_path / f"cycle_{i}_results.json"
            assert archive.exists(), f"cycle_{i}_results.json missing"


class TestEnrichedHistory:
    """Tests for enriched selection_history entries."""

    def test_history_contains_effective_base_and_cycle(self, tmp_path):
        path = _write_cycle_results(tmp_path, 7, 0.90)
        history_path = tmp_path / "history.json"
        select_and_promote(path, "/fake/adapter", "/fake/merged",
                           base_model_path="/original/base",
                           history_path=history_path, dry_run=True)
        history = json.loads(history_path.read_text())
        assert len(history) == 1
        assert history[0]["cycle"] == 7
        assert history[0]["effective_base"] == "/original/base"
        assert history[0]["promoted"] is True

    def test_history_accumulates_across_cycles(self, tmp_path):
        history_path = tmp_path / "history.json"
        for i in range(3):
            (tmp_path / f"c{i}").mkdir()
            path = _write_cycle_results(tmp_path / f"c{i}", i, 0.50 + i * 0.05)
            select_and_promote(path, f"/adapter/{i}", "/fake/merged",
                               baseline_accuracy=0.48 if i > 0 else None,
                               history_path=history_path, dry_run=True)
        history = json.loads(history_path.read_text())
        assert len(history) == 3
        assert all("cycle" in h for h in history)
        assert all("effective_base" in h for h in history)


# =====================================================================
# Tests for resolve_cycle.py  (partner's decision engine)
# =====================================================================

class TestDecideFunction:
    """Tests for resolve_cycle.decide() — pure logic, no I/O."""

    def test_keep_on_clear_improvement(self):
        decision, reason = decide(0.50, 0.60, 1.0)
        assert decision == "KEEP"
        assert "Improved" in reason

    def test_revert_on_zero_success(self):
        """Zero success rate always reverts, even if accuracy improved."""
        decision, reason = decide(0.50, 0.90, 0)
        assert decision == "REVERT"
        assert "Zero success" in reason

    def test_revert_on_regression(self):
        decision, reason = decide(0.80, 0.70, 0.5)
        assert decision == "REVERT"
        assert "below threshold" in reason

    def test_revert_on_equal_accuracy(self):
        """Equal accuracy (delta=0) does NOT meet epsilon threshold."""
        decision, reason = decide(0.75, 0.75, 0.5)
        assert decision == "REVERT"

    def test_revert_on_tiny_improvement_below_epsilon(self):
        """Improvement smaller than IMPROVEMENT_EPS (1e-4) is noise."""
        decision, _ = decide(0.750000, 0.750005, 0.5)
        assert decision == "REVERT"

    def test_keep_on_improvement_just_above_epsilon(self):
        """Improvement just above IMPROVEMENT_EPS should keep."""
        decision, _ = decide(0.750000, 0.750200, 0.5)
        assert decision == "KEEP"


class TestLoadMetrics:
    """Tests for resolve_cycle.load_metrics() — JSON parsing."""

    def test_full_metrics(self, tmp_path):
        p = tmp_path / "results.json"
        p.write_text(json.dumps({
            "cycle": 5,
            "evaluation": {"best_accuracy": 0.82, "success_rate": 0.75},
        }))
        m = load_metrics(p)
        assert m["accuracy"] == 0.82
        assert m["success_rate"] == 0.75
        assert m["cycle"] == 5

    def test_missing_fields_default_to_zero(self, tmp_path):
        p = tmp_path / "results.json"
        p.write_text(json.dumps({"evaluation": {}}))
        m = load_metrics(p)
        assert m["accuracy"] == 0.0
        assert m["success_rate"] == 0.0
        assert m["cycle"] == 0

    def test_missing_evaluation_defaults(self, tmp_path):
        p = tmp_path / "results.json"
        p.write_text(json.dumps({"cycle": 3}))
        m = load_metrics(p)
        assert m["accuracy"] == 0.0
        assert m["success_rate"] == 0.0
        assert m["cycle"] == 3


class TestResolveCycleLineage:
    """Tests for resolve_cycle.load_lineage() / save_lineage() with monkeypatched paths."""

    def test_save_and_load_lineage(self, tmp_path, monkeypatch):
        lineage_file = tmp_path / "accepted_adapters.json"
        monkeypatch.setattr(resolve_cycle, "LINEAGE_FILE", lineage_file)

        # Create a fake adapter_config.json so infer_base_model works
        epoch_dir = tmp_path / "out" / "nngpt" / "llm" / "epoch" / "A0"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "deepseek-ai/deepseek-coder-1.3b-instruct"
        }))

        lineage = {"adapters": [{"epoch": 0, "path": str(epoch_dir)}],
                    "base_model": "deepseek-ai/deepseek-coder-1.3b-instruct"}
        save_lineage(lineage)
        assert lineage_file.exists()

        loaded = load_lineage()
        assert loaded["base_model"] == "deepseek-ai/deepseek-coder-1.3b-instruct"
        assert len(loaded["adapters"]) == 1
        assert loaded["adapters"][0]["epoch"] == 0

    def test_load_lineage_creates_new_when_missing(self, tmp_path, monkeypatch):
        """When no lineage file exists, load_lineage() creates a fresh one via infer_base_model."""
        lineage_file = tmp_path / "nonexistent" / "adapters.json"
        monkeypatch.setattr(resolve_cycle, "LINEAGE_FILE", lineage_file)

        # Need at least one adapter_config.json for infer_base_model
        epoch_dir = tmp_path / "out" / "nngpt" / "llm" / "epoch" / "A1"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "test-model/base"
        }))
        monkeypatch.chdir(tmp_path)

        result = load_lineage()
        assert result["adapters"] == []
        assert result["base_model"] == "test-model/base"

    def test_load_lineage_fills_missing_keys(self, tmp_path, monkeypatch):
        """Existing lineage file missing 'adapters' key gets it added."""
        lineage_file = tmp_path / "accepted_adapters.json"
        lineage_file.write_text(json.dumps({"base_model": "some-model"}))
        monkeypatch.setattr(resolve_cycle, "LINEAGE_FILE", lineage_file)

        result = load_lineage()
        assert result["adapters"] == []
        assert result["base_model"] == "some-model"


class TestGetCycleFiles:
    """Tests for resolve_cycle.get_cycle_files() with monkeypatched NNGPT_DIR."""

    def test_finds_current_and_previous(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resolve_cycle, "NNGPT_DIR", tmp_path)

        current = tmp_path / "cycle_results.json"
        current.write_text(json.dumps({"cycle": 3, "evaluation": {"best_accuracy": 0.55}}))

        backup = tmp_path / "cycle_results_002.json"
        backup.write_text(json.dumps({"cycle": 2, "evaluation": {"best_accuracy": 0.50}}))

        prev, curr = get_cycle_files()
        assert curr == current
        assert prev == backup

    def test_picks_newest_backup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resolve_cycle, "NNGPT_DIR", tmp_path)

        current = tmp_path / "cycle_results.json"
        current.write_text("{}")

        old = tmp_path / "cycle_results_001.json"
        old.write_text("{}")
        time.sleep(0.05)
        new = tmp_path / "cycle_results_002.json"
        new.write_text("{}")

        prev, _ = get_cycle_files()
        assert prev == new

    def test_raises_when_no_current(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resolve_cycle, "NNGPT_DIR", tmp_path)
        with pytest.raises(RuntimeError, match="cycle_results.json not found"):
            get_cycle_files()

    def test_raises_when_no_backups(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resolve_cycle, "NNGPT_DIR", tmp_path)
        (tmp_path / "cycle_results.json").write_text("{}")
        with pytest.raises(RuntimeError, match="No backup found"):
            get_cycle_files()


class TestDecisionPipeline:
    """End-to-end test: load_metrics → decide → verify outcome."""

    def test_three_cycle_scenario(self, tmp_path):
        """Simulate 3 cycles through the resolve_cycle decision logic."""
        cycles = [
            {"cycle": 1, "evaluation": {"best_accuracy": 0.42, "success_rate": 0.80}},
            {"cycle": 2, "evaluation": {"best_accuracy": 0.51, "success_rate": 0.85}},
            {"cycle": 3, "evaluation": {"best_accuracy": 0.50, "success_rate": 0.60}},
        ]
        decisions = []
        for i, data in enumerate(cycles):
            p = tmp_path / f"cycle_{i}.json"
            p.write_text(json.dumps(data))
            m = load_metrics(p)
            if i == 0:
                prev_acc = 0.0
            else:
                prev_p = tmp_path / f"cycle_{i - 1}.json"
                prev_m = load_metrics(prev_p)
                prev_acc = prev_m["accuracy"]
            decision, _ = decide(prev_acc, m["accuracy"], m["success_rate"])
            decisions.append(decision)

        assert decisions == ["KEEP", "KEEP", "REVERT"]

    def test_zero_success_always_reverts(self, tmp_path):
        """Even a huge accuracy jump is reverted when success_rate is 0."""
        p = tmp_path / "cycle.json"
        p.write_text(json.dumps({
            "cycle": 1,
            "evaluation": {"best_accuracy": 0.99, "success_rate": 0.0},
        }))
        m = load_metrics(p)
        decision, _ = decide(0.0, m["accuracy"], m["success_rate"])
        assert decision == "REVERT"
