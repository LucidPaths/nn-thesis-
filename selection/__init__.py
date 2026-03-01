"""
Cycle selection and LoRA promotion module.

After each fine-tuning cycle, compares the cycle's best NN accuracy against
the previous cycle's baseline. On a positive delta, permanently merges the
cycle's LoRA adapter into the base LLM weights via MergeLLM.merge().
"""

from selection.result import SelectionResult
from selection.promote import select_and_promote, load_cycle_results, get_best_accuracy
from selection.history import append_history
