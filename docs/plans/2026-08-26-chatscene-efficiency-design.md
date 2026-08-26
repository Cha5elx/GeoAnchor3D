# Chat-Scene Efficiency Statistics Implementation Plan

> **For Claude:** Implement this plan task-by-task without creating a worktree or Git commit, as explicitly requested by the user.

**Goal:** Add reproducible trainable-parameter and batch-1 ScanRefer latency statistics for the original Chat-Scene model in `models/chat3d_origin.py` without changing existing training, validation, or ablation behavior.

**Architecture:** Keep the existing GeoAnchor3D training path untouched. Put parameter accounting, checkpoint loading, generation-option construction, and latency aggregation in a small shared utility; provide standalone audit and benchmark entry points; add only opt-in generation arguments to the original Chat-Scene model.

**Tech Stack:** Python, PyTorch, Transformers generation, existing Chat-Scene datasets/configuration.

---

### Task 1: Shared efficiency utilities

**Files:**
- Create: `utils/efficiency.py`
- Test: `tests/test_efficiency.py`

1. Add tests for `requires_grad` accounting, tied-parameter deduplication, module grouping, warm-up exclusion, batch-size rejection, completion, generation defaults, and JSON summary fields.
2. Implement parameter grouping and byte/count aggregation without importing PyTorch at module import time.
3. Implement an efficiency tracker using population standard deviation and linearly interpolated percentiles.
4. Implement generation kwargs that preserve the original beam/EOS behavior unless efficiency mode is explicit.

### Task 2: Original Chat-Scene opt-in generation metadata

**Files:**
- Modify: `models/chat3d_origin.py`

1. Build generation kwargs through the shared helper.
2. In normal mode retain `max_new_tokens=self.max_txt_len`, `num_beams=5`, and no `min_new_tokens`.
3. In efficiency mode set equal min/max new tokens and the requested beam count.
4. Return generated-token counts only when explicitly requested; retain the original list-of-text return otherwise.

### Task 3: Standalone parameter audit

**Files:**
- Create: `scripts/audit_model_parameters.py`

1. Load the existing Python configuration and command-line overrides.
2. Build only `models.chat3d_origin.Chat3D`, optionally load checkpoint weights, and never construct a dataset or run a forward pass.
3. Write model, checkpoint, dtype, and deduplicated parameter statistics to JSON.

### Task 4: Standalone ScanRefer latency benchmark

**Files:**
- Create: `scripts/benchmark_chatscene_efficiency.py`
- Modify: `scripts/config.py`

1. Add inert efficiency defaults to the existing configuration.
2. Enforce CUDA, one process, ScanRefer, batch size one, beam one, and positive warm-up/timed counts.
3. Reuse the existing validation dataset and collate function.
4. Move each batch to CUDA before synchronizing and starting `time.perf_counter()`.
5. Run the formal `model(**batch, is_eval=True)` call under `torch.inference_mode()`, synchronize, then record time.
6. Stop after warm-up plus the requested timed samples and write version-2 JSON including parameters, generation, hardware, memory, checkpoint, and latency statistics.

### Task 5: Verification and handoff

1. Run `python -m unittest tests.test_efficiency -v`.
2. Run `python -m compileall` on all changed Python files.
3. Review `git diff --check` and `git diff` to ensure the default path is unchanged.
4. Report that local GPU execution was not possible if PyTorch/CUDA/data/checkpoint are unavailable, and provide exact server foreground/background/log/result commands using only paths already present in the repository.
