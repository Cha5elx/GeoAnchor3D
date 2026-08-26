#!/usr/bin/env python
"""Build the original Chat-Scene model and audit its final parameters."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument("--config", default="scripts/config.py")
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
        help="Parameter counts do not depend on the attention kernel; eager avoids FlashAttention dependencies.",
    )
    parser.add_argument("--output", required=True)
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
    config.model.llama_model_path = args.llm_path
    config.model.attn_implementation = args.attn_implementation
    config.device = args.device
    config.pretrained_path = args.checkpoint
    config.auto_resume = False
    config.distributed = False
    return config


def main():
    args = parse_args()
    args.llm_path, args.checkpoint = validate_model_paths(args.llm_path, args.checkpoint)
    config = load_config(args)

    import torch
    from models.chat3d_origin import Chat3D

    device = torch.device(config.device)
    model = Chat3D(config=config).to(device)
    checkpoint_load = load_checkpoint_weights(model, args.checkpoint)
    parameter_stats = count_model_parameters(model)
    dtypes = sorted({str(parameter.dtype).replace("torch.", "") for parameter in model.parameters()})

    result = {
        "version": 2,
        "status": "passed",
        "method": "Chat-Scene",
        "model_class": "models.chat3d_origin.Chat3D",
        "device": str(device),
        "dtype": dtypes,
        "attention_implementation": args.attn_implementation,
        "llm_path": os.path.abspath(args.llm_path),
        "checkpoint": checkpoint_metadata(args.checkpoint),
        "checkpoint_load": checkpoint_load,
        "parameters": parameter_stats,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
