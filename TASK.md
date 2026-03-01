# TASK: Selection & Tracking Mechanism for NNGPT Pipeline

## Context

This repo contains extracted working files from [ABrain-One/nn-gpt](https://github.com/ABrain-One/nn-gpt) and [ABrain-One/nn-dataset](https://github.com/ABrain-One/nn-dataset).

NNGPT is a system that uses LLMs to automatically generate neural network architectures. The LEMUR dataset (`nn-dataset`) provides the training data, model zoo, and evaluation API.

## Current Pipeline (How It Works Now)

The system runs this loop:

1. **Base LLM** (e.g., Code Llama / DeepSeek) — a pretrained code-generation model
2. **LoRA Fine-tuning** — the base LLM is fine-tuned on LEMUR dataset (neural network code + performance data) using LoRA adapters
3. **NN Generation** — the fine-tuned LLM generates new neural network architecture code
4. **Evaluation** — generated NNs are trained and evaluated (accuracy, efficiency metrics) via `check_nn()` from `ab.nn.api`

### Key Scripts (from `ab/gpt/`)

- `TuneNNGen*.py` — Performs LoRA fine-tuning of the LLM, then generates NNs and evaluates them
- `NNAlter*.py` — Generates modified/new neural network models using the LLM
- `NNEval.py` — Evaluates generated NN models by training them and measuring performance

### Key API (from `nn-dataset`)

- `ab.nn.api.data()` → Returns a DataFrame of historical experiment results (nn_code, hyperparameters, accuracy, duration)
- `ab.nn.api.check_nn()` → Submits new NN code for automated training/evaluation, returns (model_name, accuracy, accuracy_to_time, code_quality)

## The Problem

**The pipeline currently has no selection pressure.** It runs loops of fine-tuning → generation → evaluation, but:

- There is no tracking of which base model / LoRA produced which results
- There is no comparison against a baseline to determine if a loop iteration improved things
- There is no mechanism to promote a better-performing fine-tuned LLM as the new base
- There is no rollback when a loop iteration produces worse results
- The evaluation is essentially non-deterministic with no filtering for quality

**In short: it's evolution without natural selection.**

## The Goal

Build a **selection and tracking layer** that turns the blind loop into a directed improvement loop. This module should integrate with the existing codebase (not replace it).

### Required Capabilities

#### 1. Tracking / Logging
- Record every loop iteration with:
  - Loop ID / generation number
  - Base model identifier (which LLM + which LoRA adapter if any)
  - Fine-tuning configuration (hyperparameters, training data subset, epochs)
  - List of generated NN architectures with their eval scores (accuracy, accuracy_to_time, code_quality)
  - Aggregate metrics per loop (mean accuracy, best accuracy, % of valid/compilable NNs)
  - Timestamp and duration
- Store this as structured data (JSON/SQLite) so it can be queried and visualized

#### 2. Baseline Comparison
- Maintain a "current best" baseline score
- After each loop iteration, compare the generated NNs' aggregate performance against baseline
- Compute improvement delta (or regression)

#### 3. Selection / Promotion
- If a loop iteration produces measurably better results → promote that fine-tuned LLM (LoRA adapter) as the new base for the next iteration
- Define "better" clearly: e.g., higher mean accuracy of generated NNs, or higher best-accuracy, or a composite score
- The promotion should be automatic but configurable (thresholds, which metric to optimize)

#### 4. Rollback
- If a loop iteration produces worse results → do NOT promote; keep the previous best as the base
- Log the failed iteration for analysis (it's still useful data)
- Optionally: if N consecutive iterations fail to improve, flag for human review

#### 5. Forward-Proof Pipeline
- The end state is a self-improving loop:
  ```
  Base LLM₀ → LoRA fine-tune → Generate NNs → Evaluate
       ↓ (if better)
  Base LLM₁ → LoRA fine-tune → Generate NNs → Evaluate
       ↓ (if better)
  Base LLM₂ → ...
  ```
- Each generation should be identifiable and reproducible
- The system should support branching (try multiple LoRA configs per generation, pick the best)

## Design Constraints

- **Must integrate with existing code** — this plugs INTO the `ab/gpt/` pipeline, not replace it
- **Minimal dependencies** — use standard library + what's already in the project (PyTorch, transformers, etc.)
- **The nn-dataset API is the evaluation interface** — use `check_nn()` for scoring, `data()` for historical context
- **LoRA adapters are the unit of iteration** — the base model weights don't change; each "generation" is a new LoRA adapter
- **Storage should be lightweight** — JSON or SQLite, not a full database server

## Files in This Repo

```
src/           — Relevant source files copied from ab/gpt/ (the pipeline scripts)
reference/     — API signatures and relevant code from nn-dataset for context
TASK.md        — This file
```

## Instructions for AI Agent

1. **Start by reading all files in `src/`** to understand the current pipeline flow
2. **Read `reference/`** to understand the evaluation API
3. **Design the selection module** as a new file (e.g., `selection.py` or `tracker.py`) that can be imported by the existing scripts
4. **Show where and how it hooks into the existing pipeline** — provide the specific integration points (which functions to call, where in the loop)
5. **Write tests** that validate the tracking/comparison/promotion logic independently of GPU training

## Parent Repos (reference only, do not clone)
- https://github.com/ABrain-One/nn-gpt — full pipeline repo, these src files are extracted from `ab/gpt/`
- https://github.com/ABrain-One/nn-dataset — LEMUR dataset, the `ab.nn.api` module contains `data()` and `check_nn()`

Fetch specific files from these repos as needed for import signatures or API details.
