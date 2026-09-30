"""Summarize recorded retrieval, selection, and generation latencies."""

from __future__ import annotations

import argparse
import statistics

from biomrag.utils.jsonl import read_jsonl, write_json

COMPONENT_KEYS = ("retrieval_latency", "selection_latency", "generation_latency", "total_latency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    summary = {}
    for key in COMPONENT_KEYS:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
