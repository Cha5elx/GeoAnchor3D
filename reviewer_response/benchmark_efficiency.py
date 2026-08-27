#!/usr/bin/env python
"""Benchmark baseline/GeoAnchor3D inference or training-step efficiency."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.efficiency import (
    checkpoint_metadata,
    count_model_parameters,
    load_checkpoint_weights,
    validate_model_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "geoanchor3d"), required=True)
    parser.add_argument("--mode", choices=("inference", "training"), required=True)
    parser.add_argument("--config", default="scripts/config.py")
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--timed-batches", type=int, default=50)
    parser.add_argument("--fixed-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-obj-num", type=int, default=100)
    return parser.parse_args()


def load_config(args):
    from utils.config import Config

    config = Config.from_file(args.config)
    config.model.llama_model_path = args.llm_path
    config.model.attn_implementation = args.attn_implementation
    config.model.max_obj_num = args.max_obj_num
    config.model.add_scene_token = False
    config.lora.lora_r = args.lora_rank
    config.pretrained_path = args.checkpoint
    config.train_tag = "scanrefer"
    config.val_tag = "scanrefer"
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device
    config.evaluate = args.mode == "inference"
    config.wandb.enable = False
    config.auto_resume = False
    config.distributed = False
    config.gpu_num = 1
    if args.model == "geoanchor3d":
        config.model.gate_granularity = "per_head"
        config.model.alpha_ablation_mode = 0
    return config


def validate_args(args, torch):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("efficiency measurement requires one process")
    if args.warmup_batches < 0 or args.timed_batches <= 0:
        raise ValueError("warmup-batches must be non-negative and timed-batches positive")
    if args.mode == "inference" and args.batch_size != 1:
        raise ValueError("inference comparison requires --batch-size 1")
    if args.mode == "inference" and args.fixed_new_tokens <= 0:
        raise ValueError("fixed-new-tokens must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synchronized efficiency measurement")
    return device


def build_model(args, config, device):
    if args.model == "baseline":
        from models.chat3d_origin import Chat3D
    else:
        from models.chat3d import Chat3D
    model = Chat3D(config=config).to(device)
    checkpoint_load = load_checkpoint_weights(model, args.checkpoint)
    return model, checkpoint_load


def build_loader(args, config):
    from dataset import create_dataset, create_loader
    from dataset.dataset_train import train_collate_fn
    from dataset.dataset_val import val_collate_fn

    train_datasets, val_datasets = create_dataset(config)
    datasets = val_datasets if args.mode == "inference" else train_datasets
    if len(datasets) != 1:
        raise RuntimeError("efficiency measurement requires exactly the ScanRefer split")
    if args.mode == "inference" and datasets[0].datasets[0].dataset_name != "scanrefer":
        raise RuntimeError("inference efficiency measurement requires ScanRefer validation")
    collate_fn = val_collate_fn if args.mode == "inference" else train_collate_fn
    return create_loader(
        datasets,
        samplers=[None],
        batch_size=[args.batch_size],
        num_workers=[args.num_workers],
        is_trains=[args.mode == "training"],
        collate_fns=[collate_fn],
    )[0]


def move_batch(batch, device, torch):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(values, batch_size):
    total = sum(values)
    count = len(values)
    mean = total / count
    return {
        "timed_batch_count": count,
        "timed_sample_count": count * batch_size,
        "total_measured_seconds": total,
        "seconds_per_batch": {
            "mean": mean,
            "std": statistics.pstdev(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        },
        "milliseconds_per_sample": mean * 1000.0 / batch_size,
        "throughput_samples_per_second": count * batch_size / total,
    }


def run_inference(model, loader, args, device, torch):
    model.eval()
    model.llama_model.config.use_cache = True
    timings = []
    observed_tokens = []
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, device, torch)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = model(
                **batch,
                is_eval=True,
                efficiency_mode=True,
                efficiency_fixed_new_tokens=args.fixed_new_tokens,
                num_beams=1,
                return_generation_metadata=True,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        token_counts = result[-1]
        if batch_index >= args.warmup_batches:
            timings.append(elapsed)
            observed_tokens.extend(int(value) for value in token_counts)
        if len(timings) >= args.timed_batches:
            break
    return timings, {
        "num_beams": 1,
        "fixed_new_tokens": args.fixed_new_tokens,
        "observed_token_min": min(observed_tokens),
        "observed_token_max": max(observed_tokens),
    }


def run_training(model, loader, args, device, torch):
    model.train()
    model.llama_model.config.use_cache = False
    timings = []
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, device, torch)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss_dict = model(**batch)
        loss_dict["loss"].backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if batch_index >= args.warmup_batches:
            timings.append(elapsed)
        if len(timings) >= args.timed_batches:
            break
    return timings, {"scope": "forward + backward; excludes optimizer step and data transfer"}


def main():
    args = parse_args()
    args.llm_path, args.checkpoint = validate_model_paths(
        args.llm_path, args.checkpoint, require_checkpoint=True
    )
    import torch

    device = validate_args(args, torch)
    config = load_config(args)
    loader = build_loader(args, config)
    model, checkpoint_load = build_model(args, config, device)
    parameters = count_model_parameters(model)
    torch.cuda.reset_peak_memory_stats(device)

    if args.mode == "inference":
        timings, mode_details = run_inference(model, loader, args, device, torch)
    else:
        timings, mode_details = run_training(model, loader, args, device, torch)
    if len(timings) != args.timed_batches:
        raise RuntimeError(
            f"requested {args.timed_batches} timed batches but collected {len(timings)}"
        )

    report = {
        "version": 1,
        "status": "passed",
        "method": "Chat-Scene" if args.model == "baseline" else "GeoAnchor3D",
        "mode": args.mode,
        "model_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "attention_implementation": args.attn_implementation,
        "checkpoint": checkpoint_metadata(args.checkpoint),
        "checkpoint_load": checkpoint_load,
        "parameters": parameters,
        "cuda": {
            "device": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "mode_details": mode_details,
        **timing_summary(timings, args.batch_size),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
