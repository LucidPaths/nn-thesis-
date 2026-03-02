"""Check if fine-tuned model improved over baseline; merge adapter on positive delta."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace


def select_and_promote(cycle_results_path, adapter_path, merged_output_path,
                       base_model_path=None, baseline_accuracy=None,
                       history_path=None, dry_run=False):
    """
    Compare this cycle's best accuracy against the baseline.
    On positive delta (or first cycle), merge the LoRA adapter into base model.
    If base_model_path is None, reads it from adapter_config.json.

    Progressive merging: if merged_output_path already contains a model from
    a prior promotion, the adapter is merged against that (not the original
    base_model_path), so improvements accumulate across cycles.
    """
    cycle_results_path = Path(cycle_results_path)
    with open(cycle_results_path) as f:
        cycle_data = json.load(f)
    best_accuracy = cycle_data.get("evaluation", {}).get("best_accuracy")

    # Determine effective base: use previously merged model if it exists
    effective_base = _resolve_base(base_model_path, merged_output_path)

    if best_accuracy is None:
        result = SimpleNamespace(promoted=False, best_accuracy=None, delta=None,
                                 effective_base=effective_base)
        _append_history(result, history_path, cycle_data)
        _archive_cycle_results(cycle_results_path, cycle_data)
        return result

    if baseline_accuracy is None:
        promoted, delta = True, None
    else:
        delta = best_accuracy - baseline_accuracy
        promoted = delta > 0

    if promoted and not dry_run:
        from MergeLLM import merge
        if effective_base is None:
            with open(Path(adapter_path) / "adapter_config.json") as f:
                effective_base = json.load(f)["base_model_name_or_path"]
        merge(str(effective_base), str(adapter_path), str(merged_output_path))

    result = SimpleNamespace(promoted=promoted, best_accuracy=best_accuracy, delta=delta,
                             effective_base=effective_base)
    _append_history(result, history_path, cycle_data)
    _archive_cycle_results(cycle_results_path, cycle_data)
    return result


def _resolve_base(base_model_path, merged_output_path):
    """Return the most current base: the merged output if it already exists
    from a prior promotion, otherwise the original base_model_path."""
    if merged_output_path is not None:
        merged = Path(merged_output_path)
        if merged.is_dir() and any(merged.iterdir()):
            return str(merged)
    return str(base_model_path) if base_model_path else None


def _archive_cycle_results(cycle_results_path, cycle_data):
    """Save a per-cycle copy so results survive when the shared file is overwritten."""
    cycle_num = cycle_data.get("cycle")
    if cycle_num is None:
        return
    archive_path = cycle_results_path.parent / f"cycle_{cycle_num}_results.json"
    shutil.copy2(cycle_results_path, archive_path)


def _append_history(result, history_path, cycle_data=None):
    if history_path is None:
        return
    history_path = Path(history_path)
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    entry = {
        "promoted": result.promoted,
        "best_accuracy": result.best_accuracy,
        "delta": result.delta,
        "effective_base": getattr(result, "effective_base", None),
    }
    if cycle_data:
        entry["cycle"] = cycle_data.get("cycle")
    history.append(entry)
    history_path.write_text(json.dumps(history, indent=2))
