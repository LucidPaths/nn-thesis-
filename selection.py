"""Check if fine-tuned model improved over baseline; merge adapter on positive delta."""

import json
from pathlib import Path


def select_and_promote(cycle_results_path, adapter_path, merged_output_path,
                       baseline_accuracy=None, dry_run=False):
    """
    Compare this cycle's best accuracy against the baseline.
    On positive delta (or first cycle), merge the LoRA adapter into base model.
    Base model path is read from adapter_config.json in the adapter directory.
    """
    with open(cycle_results_path) as f:
        best_accuracy = json.load(f).get("evaluation", {}).get("best_accuracy")

    if best_accuracy is None:
        return {"promoted": False, "best_accuracy": None, "delta": None}

    if baseline_accuracy is None:
        promoted, delta = True, None
    else:
        delta = best_accuracy - baseline_accuracy
        promoted = delta > 0

    if promoted and not dry_run:
        from MergeLLM import merge
        adapter_path = Path(adapter_path)
        with open(adapter_path / "adapter_config.json") as f:
            base_model_path = json.load(f)["base_model_name_or_path"]
        merge(base_model_path, str(adapter_path), str(merged_output_path))

    return {"promoted": promoted, "best_accuracy": best_accuracy, "delta": delta}
