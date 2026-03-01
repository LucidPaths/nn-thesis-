import json
from dataclasses import asdict
from pathlib import Path

from selection.result import SelectionResult


def append_history(result: SelectionResult, history_path) -> None:
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
