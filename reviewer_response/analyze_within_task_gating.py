#!/usr/bin/env python
"""Analyze ScanRefer gate values by within-task spatial complexity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewer_response.experiment_utils import (
    count_spatial_relations,
    spatial_complexity_group,
    summarize_values,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Merged ScanRefer prediction JSON")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def analyze(records):
    grouped = {"0": [], "1": [], "2+": []}
    head_values = {"0": [], "1": [], "2+": []}
    for index, record in enumerate(records):
        if "prompt" not in record:
            raise ValueError(f"record {index} has no prompt")
        if "gate_values" not in record and "gate_value_mean" not in record:
            raise ValueError(f"record {index} has no gate value")
        relation_count = count_spatial_relations(record["prompt"])
        group = spatial_complexity_group(relation_count)
        gates = record.get("gate_values")
        if gates is None:
            gates = [record["gate_value_mean"]]
        gates = [float(value) for value in gates]
        if not gates:
            raise ValueError(f"record {index} has an empty gate_values list")
        grouped[group].append(sum(gates) / len(gates))
        head_values[group].append(gates)

    summary = {group: summarize_values(grouped[group]) for group in ("0", "1", "2+")}
    summary["metadata"] = {
        "total_samples": len(records),
        "group_definition": {"0": "no explicit relation", "1": "one", "2+": "two or more"},
        "gate_statistic": "per-sample mean across available gate heads",
        "relation_lexicon_version": 1,
    }
    return summary


def write_csv(path, summary):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("group", "count", "mean", "std", "ci95_low", "ci95_high"),
        )
        writer.writeheader()
        for group in ("0", "1", "2+"):
            writer.writerow({"group": group, **summary[group]})


def main():
    args = parse_args()
    records = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("prediction JSON must contain a list")
    summary = analyze(records)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_csv, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
