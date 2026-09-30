"""Compute LLaVA-Med style open-ended generation metrics."""

from __future__ import annotations

import argparse
import json

from biomrag.utils.jsonl import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions = read_jsonl(args.predictions)
    references = _rows_by_id(read_jsonl(args.references), label="reference")
    if not predictions or not references:
        raise ValueError("Open-ended evaluation inputs must be non-empty.")
    prediction_ids = [str(row.get("id")) for row in predictions]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("Prediction IDs must be unique.")
    if set(prediction_ids) != set(references):
        raise ValueError("Prediction and reference ID sets must match exactly.")
    if args.expected_count and len(predictions) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} open-ended examples, "
            f"got {len(predictions)}."
        )
    hyps = []
    refs = []
    for pred in predictions:
        pred_id = str(pred.get("id"))
        ref = references.get(pred_id)
        if ref is None:
            raise KeyError(f"Missing reference for prediction id={pred_id}")
        hyp = str(pred.get("response") or pred.get("prediction") or "")
        if not hyp.strip():
            raise ValueError(f"Empty prediction for id={pred_id}")
        reference = reference_text(ref)
        if not reference:
            raise ValueError(f"Empty reference answer for id={pred_id}")
        hyps.append(hyp)
        refs.append(reference)

    result = compute_caption_metrics(hyps, refs)
    result = {key: round(value * 100, 4) for key, value in result.items()}
    print(result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")


def reference_text(row: dict) -> str:
    if row.get("answer") or row.get("reference") or row.get("reference_answer"):
        return str(
            row.get("answer")
            or row.get("reference")
            or row.get("reference_answer")
        )
    for key in ("messages", "conversations"):
        turns = row.get(key)
        if isinstance(turns, list) and turns:
            last = turns[-1]
            if isinstance(last, dict):
                return str(last.get("content") or last.get("value") or "")
    return ""


def _rows_by_id(rows: list[dict], *, label: str) -> dict[str, dict]:
    output = {}
    for row in rows:
        if "id" not in row:
            raise ValueError(f"Every {label} row must contain an id.")
        key = str(row["id"])
        if key in output:
            raise ValueError(f"Duplicate {label} id: {key}")
        output[key] = row
    return output


def compute_caption_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise RuntimeError(
            "Open-ended metrics require pycocoevalcap==1.2."
        ) from exc

    tokenized_truth, tokenized_results = _tokenized_coco_inputs(
        hypotheses,
        references,
    )

    metrics: dict[str, float] = {}
    scorers = (
        (Bleu(4), ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4")),
        (Meteor(), ("METEOR",)),
        (Rouge(), ("ROUGE_L",)),
        (Cider(), ("CIDEr",)),
    )
    for scorer, names in scorers:
        score, _per_example = scorer.compute_score(
            tokenized_truth,
            tokenized_results,
        )
        values = score if isinstance(score, (list, tuple)) else [score]
        if len(values) != len(names):
            raise RuntimeError(
                f"{type(scorer).__name__} returned {len(values)} aggregate scores; "
                f"expected {len(names)}."
            )
        metrics.update(
            {
                name: float(value)
                for name, value in zip(names, values)
            }
        )
    return {
        "BLEU-4": metrics["Bleu_4"],
        "ROUGE-L": metrics["ROUGE_L"],
        "METEOR": metrics["METEOR"],
        "CIDEr": metrics["CIDEr"],
    }


def _tokenized_coco_inputs(
    hypotheses: list[str],
    references: list[str],
) -> tuple[dict, dict]:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError(
            "Caption metrics require non-empty matched hypotheses and references."
        )
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Open-ended metrics require pycocoevalcap==1.2."
        ) from exc

    ground_truth = {
        index: [{"caption": reference}]
        for index, reference in enumerate(references)
    }
    results = {
        index: [{"caption": hypothesis}]
        for index, hypothesis in enumerate(hypotheses)
    }
    tokenizer = PTBTokenizer()
    return tokenizer.tokenize(ground_truth), tokenizer.tokenize(results)


if __name__ == "__main__":
    main()
