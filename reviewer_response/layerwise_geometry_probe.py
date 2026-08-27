#!/usr/bin/env python
"""Fit layer-wise linear probes for object-centered 3D coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.efficiency import load_checkpoint_weights, validate_model_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "geoanchor3d"), required=True)
    parser.add_argument("--config", default="scripts/config.py")
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--layers", default="4,8,12,16,20,24,28,32")
    parser.add_argument("--max-scenes", type=int, default=100)
    parser.add_argument("--max-objects-per-scene", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--probe-epochs", type=int, default=25)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-obj-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="Describe the objects and their spatial arrangement in this 3D scene.",
    )
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
    config.batch_size = 1
    config.num_workers = 0
    config.device = args.device
    config.evaluate = True
    config.wandb.enable = False
    config.auto_resume = False
    config.distributed = False
    config.gpu_num = 1
    if args.model == "geoanchor3d":
        config.model.gate_granularity = "per_head"
        config.model.alpha_ablation_mode = 0
    return config


def parse_layers(value):
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers or min(layers) < 1:
        raise ValueError("--layers must contain positive transformer layer numbers")
    return layers


def stable_is_test(scene_id, test_fraction):
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test-fraction must be between 0 and 1")
    digest = hashlib.sha1(str(scene_id).encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return fraction < test_fraction


def build_model_and_dataset(args, config, device, torch):
    from dataset import create_dataset

    if args.model == "baseline":
        from models.chat3d_origin import Chat3D
    else:
        from models.chat3d import Chat3D

    _, val_datasets = create_dataset(config)
    if len(val_datasets) != 1 or val_datasets[0].datasets[0].dataset_name != "scanrefer":
        raise RuntimeError("geometry probing requires exactly ScanRefer validation")
    dataset = val_datasets[0].datasets[0]
    model = Chat3D(config=config).to(device)
    checkpoint_load = load_checkpoint_weights(model, args.checkpoint)
    model.eval()
    model.llama_model.config.use_cache = False
    return model, dataset, checkpoint_load


def unique_scene_indices(dataset, max_scenes):
    indices = []
    seen = set()
    for index, record in enumerate(dataset.anno):
        scene_id = record["scene_id"]
        if scene_id in seen:
            continue
        seen.add(scene_id)
        indices.append(index)
        if len(indices) >= max_scenes:
            break
    return indices


def extract_scene_features(model, sample, prompt, layers, max_objects, device, torch):
    (
        scene_feat,
        scene_img_feat,
        scene_mask,
        scene_locs,
        _obj_id,
        _assigned_ids,
        _sample_prompt,
        _ref_captions,
        scene_id,
        _qid,
        _pred_id,
        _type_info,
    ) = sample
    scene_feat = scene_feat.unsqueeze(0).to(device)
    scene_img_feat = scene_img_feat.unsqueeze(0).to(device)
    scene_mask = scene_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
    scene_locs = scene_locs.unsqueeze(0).to(device)
    object_count = min(
        int(scene_mask[0].sum().item()),
        int(scene_locs.shape[1]),
        int(max_objects),
    )
    if object_count <= 0:
        return scene_id, {}, torch.empty(0, 3)
    scene_mask[:, object_count:] = False
    assigned_ids = torch.arange(scene_mask.shape[1], device=device).unsqueeze(0)

    object_embed, object_img_embed = model.encode_object_feat(
        scene_feat, scene_img_feat, scene_locs
    )
    prompt_text = f" {prompt} {model.role[1]}: "
    text_result = model.get_text_emb([prompt_text], device=device)
    if isinstance(text_result, tuple):
        seq_embeds, prompt_mask = text_result
    else:
        seq_embeds = text_result
        prompt_mask = torch.ones(seq_embeds.shape[:2], dtype=torch.long, device=device)
    mask_expanded = prompt_mask.unsqueeze(-1).float()
    instr_embeds = (seq_embeds * mask_expanded).sum(1) / mask_expanded.sum(1).clamp_min(1e-9)
    if getattr(model, "use_spatial_attention", False):
        object_embed, _ = model.spatial_relation_attention(
            object_embed,
            scene_locs[:, :, :3],
            instr_embeds,
            alpha_ablation_mode=model.alpha_ablation_mode,
            task_type=None,
        )
    object_embed = torch.nn.functional.normalize(object_embed, dim=-1)
    projected_objects = model.object_proj(object_embed)
    projected_images = model.object_img_proj(object_img_embed)
    object_tokens = model.get_object_list_embed(
        projected_objects[0],
        projected_images[0] if model.add_img_token else None,
        None,
        scene_mask[0],
        torch.tensor(0, device=device),
        assigned_ids[0],
    )
    if object_tokens.shape[0] % object_count:
        raise RuntimeError("object token sequence is not divisible by the object count")
    stride = object_tokens.shape[0] // object_count
    valid_prompt_len = int(prompt_mask[0].sum().item())
    wrapped = torch.cat(
        [
            model.p_0_embed.to(device),
            object_tokens,
            model.p_1_embed.to(device),
            seq_embeds[0, :valid_prompt_len],
        ],
        dim=0,
    ).unsqueeze(0)
    attention_mask = torch.ones(wrapped.shape[:2], dtype=torch.long, device=device)
    with model.maybe_autocast():
        outputs = model.llama_model(
            inputs_embeds=wrapped,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
    if max(layers) >= len(outputs.hidden_states):
        raise ValueError(
            f"requested layer {max(layers)}, model exposes {len(outputs.hidden_states) - 1} layers"
        )
    object_start = model.p_0_embed.shape[0]
    layer_features = {}
    for layer in layers:
        hidden = outputs.hidden_states[layer][0]
        pooled = [
            hidden[object_start + index * stride:object_start + (index + 1) * stride].mean(0)
            for index in range(object_count)
        ]
        layer_features[layer] = torch.stack(pooled).detach().to("cpu", dtype=torch.float16)
    targets = scene_locs[0, :object_count, :3]
    targets = (targets - targets.mean(0, keepdim=True)).detach().cpu().float()
    return scene_id, layer_features, targets


def collect_features(model, dataset, args, layers, device, torch):
    buckets = {
        layer: {"train_x": [], "train_y": [], "test_x": [], "test_y": []}
        for layer in layers
    }
    scene_counts = {"train": 0, "test": 0}
    with torch.inference_mode():
        for index in unique_scene_indices(dataset, args.max_scenes):
            scene_id, features, targets = extract_scene_features(
                model,
                dataset[index],
                args.prompt,
                layers,
                args.max_objects_per_scene,
                device,
                torch,
            )
            if not features:
                continue
            split = "test" if stable_is_test(scene_id, args.test_fraction) else "train"
            scene_counts[split] += 1
            for layer in layers:
                buckets[layer][f"{split}_x"].append(features[layer])
                buckets[layer][f"{split}_y"].append(targets)
    if not scene_counts["train"] or not scene_counts["test"]:
        raise RuntimeError(f"scene split is empty: {scene_counts}")
    for layer in layers:
        for key in buckets[layer]:
            buckets[layer][key] = torch.cat(buckets[layer][key], dim=0)
    return buckets, scene_counts


def fit_probe(data, args, device, torch):
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]
    feature_mean = train_x.float().mean(0).to(device)
    feature_std = train_x.float().std(0, unbiased=False).clamp_min(1e-6).to(device)
    probe = torch.nn.Linear(train_x.shape[1], 3).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    probe.train()
    for _ in range(args.probe_epochs):
        permutation = torch.randperm(train_x.shape[0], generator=generator)
        for start in range(0, train_x.shape[0], args.probe_batch_size):
            indices = permutation[start:start + args.probe_batch_size]
            features = train_x[indices].to(device=device, dtype=torch.float32)
            targets = train_y[indices].to(device)
            features = (features - feature_mean) / feature_std
            loss = torch.nn.functional.mse_loss(probe(features), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    squared_error = torch.zeros(3, device=device)
    target_sum = torch.zeros(3, device=device)
    target_square_sum = torch.zeros(3, device=device)
    sample_count = 0
    with torch.inference_mode():
        for start in range(0, test_x.shape[0], args.probe_batch_size):
            features = test_x[start:start + args.probe_batch_size].to(
                device=device, dtype=torch.float32
            )
            targets = test_y[start:start + args.probe_batch_size].to(device)
            predictions = probe((features - feature_mean) / feature_std)
            squared_error += ((predictions - targets) ** 2).sum(0)
            target_sum += targets.sum(0)
            target_square_sum += (targets ** 2).sum(0)
            sample_count += targets.shape[0]
    mse_axis = squared_error / sample_count
    target_ss = target_square_sum - target_sum.square() / sample_count
    r2_axis = 1.0 - squared_error / target_ss.clamp_min(1e-12)
    return {
        "train_objects": int(train_x.shape[0]),
        "test_objects": int(test_x.shape[0]),
        "rmse": float(mse_axis.mean().sqrt().item()),
        "rmse_axis": [float(value) for value in mse_axis.sqrt().cpu()],
        "r2_mean": float(r2_axis.mean().item()),
        "r2_axis": [float(value) for value in r2_axis.cpu()],
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    args.llm_path, args.checkpoint = validate_model_paths(
        args.llm_path, args.checkpoint, require_checkpoint=True
    )
    layers = parse_layers(args.layers)
    import torch

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("CUDA is required for 7B hidden-state extraction")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = load_config(args)
    model, dataset, checkpoint_load = build_model_and_dataset(args, config, device, torch)
    buckets, scene_counts = collect_features(model, dataset, args, layers, device, torch)
    results = {str(layer): fit_probe(buckets[layer], args, device, torch) for layer in layers}
    report = {
        "version": 1,
        "status": "passed",
        "method": "Chat-Scene" if args.model == "baseline" else "GeoAnchor3D",
        "checkpoint_load": checkpoint_load,
        "layers": layers,
        "hidden_state_indexing": "index l is the output after transformer layer l; index 0 is excluded",
        "target": "scene-centered object coordinates (x, y, z)",
        "split": "deterministic SHA-1 scene-level split",
        "scene_counts": scene_counts,
        "probe": {
            "type": "linear",
            "epochs": args.probe_epochs,
            "learning_rate": args.probe_lr,
            "weight_decay": args.probe_weight_decay,
            "seed": args.seed,
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
