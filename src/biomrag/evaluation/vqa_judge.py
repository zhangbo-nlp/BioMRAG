"""Semantic-equivalence VQA judge using an OpenAI-compatible model."""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from biomrag.utils.jsonl import read_jsonl, write_jsonl

SYSTEM_PROMPT = """You are an impartial biomedical VQA evaluator. Compare the predicted answer with the reference answer for the given question. Ignore minor wording differences and accept medically equivalent synonyms. Penalize answers that contradict the reference, omit the key finding, or introduce unsupported clinically important information.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "JUDGE_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
    )
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("JUDGE_API_BASE_URL")
        or os.environ.get("API_BASE_URL")
        or "http://localhost:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("JUDGE_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    references = _rows_by_id(read_jsonl(args.references), label="reference")
    predictions = read_jsonl(args.predictions)
    prediction_map = _rows_by_id(predictions, label="prediction")
    if set(prediction_map) != set(references):
        only_predictions = sorted(set(prediction_map) - set(references))
        only_references = sorted(set(references) - set(prediction_map))
        raise ValueError(
            "Full-test VQA evaluation requires identical prediction and reference "
            f"ID sets. Only predictions={only_predictions[:5]}, "
            f"only references={only_references[:5]}."
        )
    if not predictions:
        raise ValueError("VQA prediction file is empty.")
    if args.expected_count and len(predictions) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} VQA examples, got {len(predictions)}."
        )

    outputs = []
    yes = no = 0
    for pred in tqdm(predictions, desc="Judging VQA"):
        pred_id = str(pred["id"])
        ref = references[pred_id]
        question = str(ref.get("question") or pred.get("question") or "").strip()
        reference = reference_text(ref).strip()
        if not question:
            raise ValueError(f"Empty VQA question for id={pred_id}")
        if not reference:
            raise ValueError(f"Empty VQA reference answer for id={pred_id}")
        prediction = str(
            pred.get("response") or pred.get("prediction") or ""
        ).strip()
        if not prediction:
            raise ValueError(f"Empty VQA prediction for id={pred_id}")
        prompt = (
            f"Question: {question}\n"
            f"Reference answer: {reference}\n\n"
            f"Predicted answer: {prediction}\n"
            "Does the predicted answer correctly satisfy or contain the "
            "reference answer? Consider only the key concepts, not the exact "
            "wording.\n"
            "Return only valid JSON:\n"
            "{\n"
            '  "correct": <true or false>,\n'
            '  "reason": "<brief explanation>"\n'
            "}"
        )
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        parsed = parse_judge_json(content)
        parsed["id"] = pred.get("id")
        outputs.append(parsed)
        if parsed.get("correct"):
            yes += 1
        else:
            no += 1
    total = yes + no
    outputs.append({"summary": {"correct": yes, "incorrect": no, "accuracy": yes / total}})
    write_jsonl(args.output, outputs)
    print(f"Accuracy: {yes / total * 100:.2f}% ({yes}/{total})")


def parse_judge_json(text: str) -> dict:
    try:
        parsed = json.loads(_extract_json(text))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed VQA judge response: {text[:240]!r}") from exc
    if not isinstance(parsed.get("correct"), bool):
        raise ValueError(f"VQA judge returned a non-boolean decision: {parsed!r}")
    parsed["reason"] = str(parsed.get("reason", "")).strip()
    if not parsed["reason"]:
        raise ValueError("VQA judge returned an empty reason.")
    return parsed


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else text


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


if __name__ == "__main__":
    main()
