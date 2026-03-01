from dataclasses import dataclass
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
