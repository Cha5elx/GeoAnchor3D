"""Shared helpers for opt-in Chat-Scene efficiency measurements."""

from __future__ import annotations

import math
import os
import statistics
from collections import defaultdict


def parameter_group(name):
    """Map a real parameter name to a stable, human-readable module group."""
    lowered = name.lower()
    if any(token in lowered for token in ("lora_", ".adapter", "adapters.", "modules_to_save")):
        return "lora_adapter"
    if lowered.startswith("spatial_relation_attention."):
        return "igga"
    if lowered.startswith("coord_head.") or lowered == "geo_layer_weights":
        return "gath"
    if lowered.startswith("llama_model."):
        return "language_model"
    if lowered.startswith("object_img_proj."):
        return "image_projector"
    if lowered.startswith("object_proj."):
        return "object_projector"
    if lowered.startswith("pos_proj.") or lowered.startswith("pos_embedding."):
        return "position_projector"
    if lowered.startswith(("scene_", "relation_module.", "graph_", "spatial_")):
        return "scene_graph_projector"
    return "other"


def _storage_key(parameter):
    """Identify exact aliases while keeping distinct views separate."""
    try:
        storage = parameter.untyped_storage()
        data_ptr = storage.data_ptr()
        if data_ptr:
            return (
                "storage",
                data_ptr,
                parameter.storage_offset(),
                tuple(parameter.shape),
                tuple(parameter.stride()),
                str(parameter.dtype),
                str(parameter.device),
            )
    except (AttributeError, RuntimeError, TypeError):
        pass
    return ("object", id(parameter))


def _named_parameters_with_aliases(model):
    try:
        return model.named_parameters(remove_duplicate=False)
    except TypeError:
        return model.named_parameters()


def count_model_parameters(model):
    """Count final resident/trainable parameters, deduplicating tied weights."""
    seen = set()
    groups = defaultdict(int)
    trainable_groups = defaultdict(int)
    total = trainable = resident_bytes = trainable_bytes = 0

    for name, parameter in _named_parameters_with_aliases(model):
        key = _storage_key(parameter)
        if key in seen:
            continue
        seen.add(key)
        count = int(parameter.numel())
        byte_count = count * int(parameter.element_size())
        group = parameter_group(name)
        total += count
        resident_bytes += byte_count
        groups[group] += count
        if bool(parameter.requires_grad):
            trainable += count
            trainable_bytes += byte_count
            trainable_groups[group] += count

    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_fraction": trainable / total if total else 0.0,
        "resident_parameter_bytes": resident_bytes,
        "trainable_parameter_bytes": trainable_bytes,
        "groups": dict(sorted(groups.items())),
        "trainable_groups": dict(sorted(trainable_groups.items())),
    }


def build_generation_kwargs(
    original_max_new_tokens,
    *,
    efficiency_mode=False,
    fixed_new_tokens=0,
    num_beams=1,
    original_num_beams=5,
):
    """Return generation limits while preserving the original default behavior."""
    kwargs = {
        "max_new_tokens": int(original_max_new_tokens),
        "num_beams": int(original_num_beams),
    }
    if efficiency_mode:
        fixed_new_tokens = int(fixed_new_tokens)
        if fixed_new_tokens <= 0:
            raise ValueError("fixed_new_tokens must be positive in efficiency mode")
        kwargs.update(
            max_new_tokens=fixed_new_tokens,
            min_new_tokens=fixed_new_tokens,
            num_beams=int(num_beams),
        )
    return kwargs


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


class EfficiencyTracker:
    """Track batch-1 warm-up and timed inference samples."""

    def __init__(self, warmup_batches=20, max_timed_samples=200):
        if warmup_batches < 0:
            raise ValueError("warmup_batches must be non-negative")
        if max_timed_samples <= 0:
            raise ValueError("max_timed_samples must be positive")
        self.warmup_batches = int(warmup_batches)
        self.max_timed_samples = int(max_timed_samples)
        self.seen_batches = 0
        self.latencies = []
        self.generated_tokens = []

    @property
    def complete(self):
        return len(self.latencies) >= self.max_timed_samples

    def record(self, elapsed_seconds, generated_tokens, batch_size=1):
        if batch_size != 1:
            raise ValueError("efficiency measurement requires batch size 1")
        if self.complete:
            return False
        is_warmup = self.seen_batches < self.warmup_batches
        self.seen_batches += 1
        if is_warmup:
            return False
        self.latencies.append(float(elapsed_seconds))
        self.generated_tokens.append(int(generated_tokens))
        return True

    def summary(self):
        count = len(self.latencies)
        total = sum(self.latencies)
        mean = total / count if count else 0.0
        std = statistics.pstdev(self.latencies) if count else 0.0
        p50 = _percentile(self.latencies, 0.50)
        p95 = _percentile(self.latencies, 0.95)
        total_tokens = sum(self.generated_tokens)
        per_token = [
            latency / tokens
            for latency, tokens in zip(self.latencies, self.generated_tokens)
            if tokens > 0
        ]
        token_mean = total_tokens / count if count else 0.0
        return {
            "complete": self.complete,
            "timed_sample_count": count,
            "total_measured_model_seconds": total,
            "throughput_samples_per_second": count / total if total else 0.0,
            "latency_seconds_per_sample": {
                "mean": mean,
                "std": std,
                "p50": p50,
                "p95": p95,
            },
            "latency_milliseconds_per_sample": {
                "mean": mean * 1000.0,
                "std": std * 1000.0,
                "p50": p50 * 1000.0,
                "p95": p95 * 1000.0,
            },
            "generated_tokens_per_sample": {
                "mean": token_mean,
                "min": min(self.generated_tokens) if count else 0,
                "max": max(self.generated_tokens) if count else 0,
            },
            "latency_seconds_per_generated_token": {
                "mean": statistics.mean(per_token) if per_token else None,
                "total_ratio": total / total_tokens if total_tokens else None,
            },
        }


def validate_model_paths(llm_path, checkpoint_path="", require_checkpoint=False):
    """Resolve local model inputs and fail before Transformers sees bad paths."""
    if not llm_path or not str(llm_path).strip():
        raise ValueError(
            "LLM path is empty. Pass a literal path or define the shell variable "
            "used by --llm-path before launching the script."
        )
    llm_path = os.path.abspath(os.path.expanduser(str(llm_path)))
    if not os.path.isdir(llm_path):
        raise FileNotFoundError(f"LLM directory does not exist: {llm_path}")

    if require_checkpoint and (not checkpoint_path or not str(checkpoint_path).strip()):
        raise ValueError("checkpoint path is required for latency measurement")
    checkpoint_path = str(checkpoint_path).strip() if checkpoint_path else ""
    if checkpoint_path:
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")
    return llm_path, checkpoint_path


def checkpoint_metadata(path):
    path = os.path.abspath(path) if path else ""
    return {
        "path": path,
        "size_bytes": os.path.getsize(path) if path and os.path.isfile(path) else None,
    }


def build_efficiency_report(
    *,
    passed,
    summary,
    parameters,
    generation,
    cuda,
    method="Chat-Scene",
    split="scanrefer",
    batch_size=1,
    warmup_batches=20,
    **metadata,
):
    """Assemble the stable version-2 efficiency JSON schema."""
    report = {
        "version": 2,
        "status": "passed" if passed else "failed",
        "method": method,
        "split": split,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "generation": generation,
        "parameters": parameters,
        "cuda": cuda,
    }
    report.update(metadata)
    report.update(summary)
    return report


def load_checkpoint_weights(model, checkpoint_path):
    """Load the model portion of a Chat-Scene checkpoint without optimizer state."""
    if not checkpoint_path:
        return {"loaded": False, "missing_keys": [], "unexpected_keys": [], "shape_mismatches": []}
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    shape_mismatches = []
    compatible = {}
    for name, value in state_dict.items():
        if name in model_state and value.size() != model_state[name].size():
            shape_mismatches.append(name)
            continue
        compatible[name] = value
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "shape_mismatches": shape_mismatches,
    }
