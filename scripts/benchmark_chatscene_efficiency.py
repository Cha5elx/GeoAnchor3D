#!/usr/bin/env python
"""Measure original Chat-Scene batch-1 generation latency on ScanRefer val."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.efficiency import (
    EfficiencyTracker,
    build_efficiency_report,
    checkpoint_metadata,
    count_model_parameters,
    load_checkpoint_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/config.py")
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--timed-samples", type=int, default=200)
    parser.add_argument("--fixed-new-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--config-options",
        nargs=argparse.REMAINDER,
        default=[],
        metavar="KEY VALUE",
        help="Existing config key/value overrides; this option must be last.",
    )
    return parser.parse_args()


def load_config(args):
    from utils.config import Config, eval_dict_leaf

    config = Config.from_file(args.config)
    if args.config_options:
        config = Config.merge_list(config, args.config_options)
        config = eval_dict_leaf(config)
    config.evaluate = True
    config.wandb.enable = False
    config.train_tag = "scanrefer"
    config.val_tag = "scanrefer"
    config.batch_size = 1
    config.model.llama_model_path = args.llm_path
    config.pretrained_path = args.checkpoint
    config.device = args.device
    config.auto_resume = False
    config.distributed = False
    config.gpu_num = 1
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.eval_efficiency_output = args.output
    config.efficiency_warmup_batches = args.warmup_batches
    config.efficiency_max_timed_samples = args.timed_samples
    config.efficiency_fixed_new_tokens = args.fixed_new_tokens
    config.num_beams = args.num_beams
    return config


def validate_environment(args, torch):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    if world_size != 1:
        raise RuntimeError(f"efficiency measurement requires world size 1, got {world_size}")
    if not args.output:
        raise ValueError("--output is required to enable efficiency measurement")
    if args.num_beams != 1:
        raise ValueError("fair batch-1 efficiency measurement requires --num-beams 1")
    if args.fixed_new_tokens <= 0:
        raise ValueError("--fixed-new-tokens must be positive")
    if args.timed_samples <= 0:
        raise ValueError("--timed-samples must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synchronized GPU latency measurement")
    return device


def move_batch_to_device(batch, device, torch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def write_report(path, report):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    args = parse_args()

    import torch
    from dataset import create_dataset, create_loader
    from dataset.dataset_val import val_collate_fn
    from models.chat3d_origin import Chat3D

    device = validate_environment(args, torch)
    config = load_config(args)
    _, val_datasets = create_dataset(config)
    if len(val_datasets) != 1 or val_datasets[0].datasets[0].dataset_name != "scanrefer":
        raise RuntimeError("efficiency measurement requires exactly the ScanRefer validation split")
    val_loader = create_loader(
        val_datasets,
        samplers=[None],
        batch_size=[1],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[val_collate_fn],
    )[0]
    if val_loader.batch_size != 1:
        raise RuntimeError(f"efficiency measurement requires batch size 1, got {val_loader.batch_size}")

    model = Chat3D(config=config).to(device)
    checkpoint_load = load_checkpoint_weights(model, args.checkpoint)
    model.eval()
    model.llama_model.config.use_cache = True
    parameters = count_model_parameters(model)
    parameter_dtypes = sorted({str(parameter.dtype).replace("torch.", "") for parameter in model.parameters()})
    generation_dtype = "bfloat16_autocast"

    tracker = EfficiencyTracker(args.warmup_batches, args.timed_samples)
    torch.cuda.reset_peak_memory_stats(device)
    for batch in val_loader:
        batch = move_batch_to_device(batch, device, torch)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _, generated_token_counts = model(
                **batch,
                is_eval=True,
                efficiency_mode=True,
                efficiency_fixed_new_tokens=args.fixed_new_tokens,
                num_beams=args.num_beams,
                return_generation_metadata=True,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if len(generated_token_counts) != 1:
            raise RuntimeError("batch-1 generation must return exactly one token count")
        tracker.record(elapsed, generated_token_counts[0], batch_size=1)
        if tracker.complete:
            break

    summary = tracker.summary()
    fixed_length_observed = (
        summary["generated_tokens_per_sample"]["min"] == args.fixed_new_tokens
        and summary["generated_tokens_per_sample"]["max"] == args.fixed_new_tokens
    )
    passed = summary["complete"] and fixed_length_observed
    report = build_efficiency_report(
        passed=passed,
        summary=summary,
        model_class="models.chat3d_origin.Chat3D",
        measurement_scope=(
            "Synchronized wall-clock time around model(**batch, is_eval=True); "
            "excludes DataLoader iteration before batch receipt, CPU-to-GPU transfer, "
            "result persistence, and task metric computation."
        ),
        warmup_batches=args.warmup_batches,
        generation={
            "num_beams": args.num_beams,
            "max_new_tokens": args.fixed_new_tokens,
            "min_new_tokens": args.fixed_new_tokens,
            "fixed_length": True,
            "fixed_length_observed": fixed_length_observed,
        },
        dtype=generation_dtype,
        parameter_dtypes=parameter_dtypes,
        llm_path=os.path.abspath(args.llm_path),
        checkpoint=checkpoint_metadata(args.checkpoint),
        checkpoint_load=checkpoint_load,
        parameters=parameters,
        cuda={
            "device": torch.cuda.get_device_name(device),
            "device_index": device.index if device.index is not None else torch.cuda.current_device(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    )
    write_report(args.output, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError(
            "efficiency run did not collect the requested samples with the fixed token length; "
            f"partial results were written to {args.output}"
        )


if __name__ == "__main__":
    main()
