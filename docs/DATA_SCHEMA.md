# Data Schemas

All project manifests use JSONL with globally unique, source-prefixed
identifiers. Relative image paths are allowed when commands run from the
project root; absolute local paths and HTTP(S) URLs are also supported at
runtime.

## Knowledge Entry

```json
{
  "id": "pmcoa_000001",
  "text": "caption, report, or approximately 200-token passage",
  "image_path": "data/images/pmcoa_000001.jpg",
  "source": "PMC-OA",
  "modality": "image-text",
  "metadata": {}
}
```

Allowed modalities are `text`, `image`, and `image-text`. Their declared value
must agree with the presence of `text` and `image_path`. The primary index uses
text and image-text entries.

## Retriever Training Entry

```json
{
  "id": "pmcoa_000001",
  "image_path": "data/images/pmcoa_000001.jpg",
  "caption": "paired biomedical caption or report"
}
```

## Query Entry

```json
{
  "id": "vqa_rad_test_0001",
  "question": "What abnormality is shown?",
  "image_path": "data/images/query_0001.jpg",
  "answer": "reference answer"
}
```

`messages` and `conversations` are also accepted. The last user turn is the
current question, prior turns form dialogue context, and the target assistant
turn is excluded from model inputs.

## Retrieved Entry

Retrieval preserves the query and adds:

```json
{
  "retrieved_docs": [
    {
      "id": "pmcoa_000001",
      "text": "...",
      "image_path": "...",
      "source": "PMC-OA",
      "modality": "image-text",
      "retrieval_rank": 1,
      "retrieval_score": 0.67
    }
  ],
  "retrieval_latency": 0.0
}
```

When candidate vectors are included, `embedding` is stored with each retrieved
item and becomes `metadata.embedding` after schema conversion.

## Distillation Entry

Subset sampling adds `parent_id`, a unique `subset_id`, `subset_type`, and the
complete proposal in both `retrieved_docs` and `selected_docs`. Paired response
generation adds `response` and `response_without_knowledge`. Teacher annotation
adds:

```json
{
  "teacher_utility": {
    "rationale": "The subset improves the answer by ...",
    "score": 8
  }
}
```

The proxy SFT representation is:

```json
{
  "messages": [
    {"role": "system", "content": "ordinal subset-utility rubric"},
    {"role": "user", "content": "<image>\nquery\n<image>\nevidence"},
    {"role": "assistant", "content": "Rationale: ...\nScore: 8"}
  ],
  "images": [
    "data/images/query.jpg",
    "data/images/evidence.jpg"
  ],
  "score": 8,
  "rationale": "..."
}
```

`<image>` tokens and `images` paths must match exactly in number and order.
Reference answers, paired responses, and teacher outputs do not appear in the
proxy's user message.

## Selected Entry

Selection adds:

```json
{
  "selected_docs": [{"id": "pmcoa_000001", "text": "..."}],
  "selection_score": 8,
  "selection_rationale": "...",
  "selection_trace": [
    {
      "candidate_id": "pmcoa_000001",
      "proposal_ids": ["pmcoa_000001"],
      "evicted_id": null,
      "score": 8,
      "rationale": "...",
      "accepted": true
    }
  ],
  "proxy_calls": 30,
  "selection_latency": 0.0
}
```

## Prediction and Evaluation Entries

Generation adds `response`, `generation_latency`, and `total_latency`.
VQA judging writes one row per ID with boolean `correct` and non-empty `reason`,
followed by a summary row. Open-ended evaluation writes aggregate BLEU-4,
ROUGE-L, METEOR, and CIDEr scores.
