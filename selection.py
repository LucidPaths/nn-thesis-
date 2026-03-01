"""
Cycle selection and LoRA promotion module.

After each fine-tuning cycle, compares the cycle's best NN accuracy against
the previous cycle's baseline. On a positive delta, permanently merges the
cycle's LoRA adapter into the base LLM weights via MergeLLM.merge().
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SelectionResult:
    """Outcome of a single cycle's promotion decision."""
    cycle: int
    best_accuracy: Optional[float]
    baseline_accuracy: Optional[float]
    promoted: bool
    delta: Optional[float]
    adapter_path: Optional[str]
    merged_path: Optional[str] = None


def load_cycle_results(cycle_results_path: Path) -> dict:
    """Read and return the cycle_results.json produced by CycleResults."""
    with open(cycle_results_path) as f:
        return json.load(f)


def get_best_accuracy(cycle_results: dict) -> Optional[float]:
    """Extract best_accuracy from the evaluation section of cycle results."""
    return cycle_results.get("evaluation", {}).get("best_accuracy")


def select_and_promote(
    cycle_results_path,
    adapter_path,
    base_model_path,
    merged_output_path,
    baseline_accuracy=None,
    history_path=None,
    dry_run=False,
) -> SelectionResult:
    """
    Core selection logic for one cycle.

    Reads cycle_results.json, compares best_accuracy against the baseline
    (previous cycle's best). If the delta is positive, permanently merges
    the LoRA adapter into the base model.

    Args:
        cycle_results_path: Path to this cycle's cycle_results.json.
        adapter_path: Path to the LoRA adapter produced by this cycle's training.
        base_model_path: Path to the current base LLM weights.
        merged_output_path: Where to save the merged model on promotion.
        baseline_accuracy: Previous cycle's best accuracy. None on the first cycle.
        history_path: Optional path to append selection history (JSON list).
        dry_run: If True, skip the actual merge (for testing).

    Returns:
        SelectionResult describing the promotion decision.
    """
    cycle_results_path = Path(cycle_results_path)
    cycle_results = load_cycle_results(cycle_results_path)
    cycle = cycle_results.get("cycle", -1)
    best_accuracy = get_best_accuracy(cycle_results)

    if best_accuracy is None:
        result = SelectionResult(
            cycle=cycle,
            best_accuracy=None,
            baseline_accuracy=baseline_accuracy,
            promoted=False,
            delta=None,
            adapter_path=str(adapter_path),
        )
        _append_history(result, history_path)
        return result

    if baseline_accuracy is None:
        # First cycle — always promote
        promoted = True
    else:
        promoted = best_accuracy > baseline_accuracy

    delta = (best_accuracy - baseline_accuracy) if baseline_accuracy is not None else None

    if promoted:
        if not dry_run:
            from MergeLLM import merge
            merge(str(base_model_path), str(adapter_path), str(merged_output_path))
        result = SelectionResult(
            cycle=cycle,
            best_accuracy=best_accuracy,
            baseline_accuracy=baseline_accuracy,
            promoted=True,
            delta=delta,
            adapter_path=str(adapter_path),
            merged_path=str(merged_output_path),
        )
    else:
        result = SelectionResult(
            cycle=cycle,
            best_accuracy=best_accuracy,
            baseline_accuracy=baseline_accuracy,
            promoted=False,
            delta=delta,
            adapter_path=str(adapter_path),
        )

    _append_history(result, history_path)
    return result


def _append_history(result: SelectionResult, history_path) -> None:
    """Append a selection result to the JSON history file."""
    if history_path is None:
        return
    history_path = Path(history_path)
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    history.append(asdict(result))
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
