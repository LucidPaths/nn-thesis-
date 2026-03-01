"""Check if fine-tuned model improved over baseline; merge adapter on positive delta."""

import json
from pathlib import Path
from types import SimpleNamespace


def select_and_promote(cycle_results_path, adapter_path, merged_output_path,
                       base_model_path=None, baseline_accuracy=None,
                       history_path=None, dry_run=False):
    """
    Compare this cycle's best accuracy against the baseline.
    On positive delta (or first cycle), merge the LoRA adapter into base model.
    If base_model_path is None, reads it from adapter_config.json.
    """
    with open(cycle_results_path) as f:
        best_accuracy = json.load(f).get("evaluation", {}).get("best_accuracy")

    if best_accuracy is None:
        result = SimpleNamespace(promoted=False, best_accuracy=None, delta=None)
        _append_history(result, history_path)
        return result

    if baseline_accuracy is None:
        promoted, delta = True, None
    else:
        delta = best_accuracy - baseline_accuracy
        promoted = delta > 0

    if promoted and not dry_run:
        from MergeLLM import merge
        if base_model_path is None:
            with open(Path(adapter_path) / "adapter_config.json") as f:
                base_model_path = json.load(f)["base_model_name_or_path"]
        merge(str(base_model_path), str(adapter_path), str(merged_output_path))

    result = SimpleNamespace(promoted=promoted, best_accuracy=best_accuracy, delta=delta)
    _append_history(result, history_path)
    return result


def _append_history(result, history_path):
    if history_path is None:
        return
    history_path = Path(history_path)
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({"promoted": result.promoted, "best_accuracy": result.best_accuracy, "delta": result.delta})
    history_path.write_text(json.dumps(history, indent=2))
