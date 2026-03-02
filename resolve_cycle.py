# resolve_cycle.py - Complete version with auto-rebuild

import json
import subprocess
from pathlib import Path

NNGPT_DIR = Path("out/nngpt")
LINEAGE_FILE = NNGPT_DIR / "accepted_adapters.json"
IMPROVEMENT_EPS = 1e-4


def infer_base_model():
    """Read base model from any adapter_config.json."""
    for cfg_path in Path("out/nngpt/llm/epoch").rglob("adapter_config.json"):
        if "synth_nn" not in str(cfg_path):
            with open(cfg_path) as f:
                return json.load(f)["base_model_name_or_path"]
    raise RuntimeError("No adapters found to infer base model")


def load_lineage():
    """Load lineage or create new one."""
    if LINEAGE_FILE.exists():
        with open(LINEAGE_FILE) as f:
            data = json.load(f)
        if "adapters" not in data:
            data["adapters"] = []
        if "base_model" not in data:
            data["base_model"] = infer_base_model()
        return data
    return {"adapters": [], "base_model": infer_base_model()}


def save_lineage(lineage):
    """Save lineage to disk."""
    LINEAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LINEAGE_FILE, "w") as f:
        json.dump(lineage, f, indent=2)


def get_cycle_files():
    """Get previous (newest backup) and current cycle files."""
    current = NNGPT_DIR / "cycle_results.json"
    if not current.exists():
        raise RuntimeError("cycle_results.json not found")

    backups = list(NNGPT_DIR.glob("cycle_results_*.json"))
    if not backups:
        raise RuntimeError("No backup found")

    previous = max(backups, key=lambda p: p.stat().st_mtime)
    return previous, current


def load_metrics(path):
    """Extract metrics from cycle results."""
    with open(path) as f:
        data = json.load(f)
    eval_data = data.get("evaluation", {})
    return {
        "accuracy": eval_data.get("best_accuracy", 0.0),
        "success_rate": eval_data.get("success_rate", 0.0),
        "cycle": data.get("cycle", 0)
    }


def decide(prev_acc, curr_acc, curr_success):
    """Decide KEEP or REVERT."""
    if curr_success == 0:
        return "REVERT", "Zero success"
    delta = curr_acc - prev_acc
    if delta > IMPROVEMENT_EPS:
        return "KEEP", f"Improved by {delta:+.6f}"
    return "REVERT", f"Delta {delta:+.6f} below threshold"


def rebuild_model():
    """Trigger model rebuild from lineage."""
    print("\n" + "=" * 70)
    print("REBUILDING MODEL FROM UPDATED LINEAGE")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["python", "-m", "ab.gpt.util.MergeLLM"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print("✓ Rebuild complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Rebuild failed: {e}")
        print(e.stderr)
        return False


def main():
    """Compare cycles, update lineage, and rebuild if needed."""
    try:
        prev_file, curr_file = get_cycle_files()
        prev_m = load_metrics(prev_file)
        curr_m = load_metrics(curr_file)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    decision, reason = decide(prev_m["accuracy"], curr_m["accuracy"], curr_m["success_rate"])
    lineage = load_lineage()
    current_epoch = curr_m["cycle"] - 1

    print("\n" + "=" * 70)
    print("CYCLE COMPARISON")
    print("=" * 70)

    # --- Extract previous metrics safely ---
    prev_acc = prev_m.get("accuracy")
    prev_sr = prev_m.get("success_rate")

    if prev_acc is None:
        prev_acc = 0.0
    if prev_sr is None:
        prev_sr = 0.0

    # --- Extract current metrics safely ---
    curr_acc = curr_m.get("accuracy")
    curr_sr = curr_m.get("success_rate")

    if curr_acc is None:
        curr_acc = 0.0
    if curr_sr is None:
        curr_sr = 0.0

    print(f"Previous: {prev_file.name} → Acc: {prev_acc:.6f}, Success: {prev_sr:.2%}")
    print(f"Current : {curr_file.name} → Acc: {curr_acc:.6f}, Success: {curr_sr:.2%}")

    print("=" * 70)

    # --- Decision rule (unchanged logic) ---
    decision = "REVERT"

    if curr_sr == 0:
        decision = "REVERT"
    elif curr_acc > prev_acc:
        decision = "KEEP"
    else:
        decision = "REVERT"

    print(f"\nDecision: {decision}")
    # Update lineage
    accepted_epochs = [a["epoch"] for a in lineage["adapters"]]

    if decision == "KEEP":
        if current_epoch not in accepted_epochs:
            lineage["adapters"].append({"epoch": current_epoch, "path": f"out/nngpt/llm/epoch/A{current_epoch}"})
            save_lineage(lineage)
            accepted_epochs.append(current_epoch)
            print(f"✓ Adapter A{current_epoch} ACCEPTED and added to lineage")
        else:
            print(f"ℹ Adapter A{current_epoch} already accepted")
    else:
        print(f"✗ Adapter A{current_epoch} REJECTED (not added to lineage)")

    print(f"📋 Accepted lineage: {accepted_epochs}\n")

    # Save decision
    with open(NNGPT_DIR / "merge_decision.json", "w") as f:
        json.dump({
            "current_epoch": current_epoch,
            "delta": curr_m['accuracy'] - prev_m['accuracy'],
            "decision": decision,
            "reason": reason,
            "accepted_lineage": accepted_epochs
        }, f, indent=2)

    print(f"✓ Decision saved\n")

    # Rebuild model if KEEP
    if decision == "KEEP":
        print("Decision is KEEP → Triggering model rebuild...")
        if rebuild_model():
            print("\n✓ Model rebuilt with new adapter")
            print("✓ Ready for next evolution cycle\n")
        else:
            print("\n⚠ Rebuild failed - run MergeLLM manually\n")
    else:
        print("Decision is REVERT → No rebuild needed")
        print("✓ Ready for next evolution cycle (will skip rejected adapter)\n")


if __name__ == "__main__":
    main()
