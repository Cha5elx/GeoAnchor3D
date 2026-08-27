# Reviewer Response Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add isolated, reproducible reviewer-response experiments for IGGA/GATH while preserving the repository's default training and evaluation behavior.

**Architecture:** Keep experiment launchers, analysis code, and generated outputs under `reviewer_response/`. Add only the configuration hooks that the existing model needs for controlled ablations and fair efficiency measurement; reuse current datasets, checkpoints, and evaluation outputs.

**Tech Stack:** Bash, Python 3.9, PyTorch, the repository's existing Chat-Scene/GeoAnchor3D training stack.

---

### Task 1: Make IGGA ablations configuration-driven

**Files:**
- Modify: `models/chat3d.py`
- Modify: `scripts/config.py`
- Test: `tests/test_reviewer_experiments.py`

1. Add explicit configuration for dynamic per-head, dynamic scalar, and fixed gating.
2. Correct the fixed-gate implementation so its documented value is actually used.
3. Add configurable gate and coordinate loss weights while preserving the intended full-model defaults.
4. Verify scalar gates broadcast over all attention heads and per-head gates remain independent.

### Task 2: Add isolated training launchers

**Files:**
- Create: `reviewer_response/run_dynamic_scalar.sh`
- Create: `reviewer_response/run_per_head_no_gate.sh`
- Create: `reviewer_response/lib/common.sh`

1. Reuse `tasks/train.py` and `scripts/config.py` through command-line overrides.
2. Require the local LLM path and optional initialization checkpoint through environment variables.
3. Write every run under `reviewer_response/results/<experiment>/`.

### Task 3: Add within-task gating analysis

**Files:**
- Create: `reviewer_response/analyze_within_task_gating.py`
- Create: `reviewer_response/run_within_task_gating.sh`
- Test: `tests/test_reviewer_experiments.py`

1. Read the existing ScanRefer prediction JSON containing prompts and gate values.
2. Count explicit spatial-relation expressions with a documented lexicon.
3. Report 0, 1, and 2+ relation groups with sample count, mean, standard deviation, and confidence interval.

### Task 4: Add layer-wise geometry probing

**Files:**
- Create: `reviewer_response/layerwise_geometry_probe.py`
- Create: `reviewer_response/run_layerwise_geometry_probe.sh`

1. Load baseline and GeoAnchor3D checkpoints one at a time.
2. Extract object-token hidden states at selected LLM layers from unique ScanRefer validation scenes.
3. Split by scene, fit lightweight linear probes, and report held-out RMSE and R-squared.
4. Keep all feature extraction and probe fitting outside the main model implementation.

### Task 5: Add fair efficiency comparison

**Files:**
- Modify: `models/chat3d.py`
- Create: `reviewer_response/benchmark_efficiency.py`
- Create: `reviewer_response/run_efficiency.sh`

1. Preserve normal generation defaults while allowing fixed-token, beam-one timing only when explicitly enabled.
2. Measure final parameter counts, synchronized batch-1 inference latency, throughput, and peak CUDA memory for baseline and GeoAnchor3D.
3. Store one JSON report per method in the isolated results directory.

### Task 6: Document experiment priority and commands

**Files:**
- Create: `reviewer_response/README.md`
- Create: `reviewer_response/.gitignore`
- Create: `reviewer_response/results/.gitkeep`

1. Mark each item as full-model retraining, evaluation only, or lightweight probe training.
2. Explain which reviewer requests are intentionally not implemented and why.
3. Document required paths, foreground commands, expected outputs, and comparison cautions.

### Task 7: Verify

1. Run focused unit tests for gating modes and within-task grouping.
2. Run the existing efficiency tests.
3. Compile every new or modified Python file.
4. Run shell syntax checks where Bash is available.
5. Review `git diff --check` and confirm generated results remain untracked.
