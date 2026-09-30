# BioMRAG

Implementation of **BioMRAG: subset-level utility deliberation for
biomedical multimodal retrieval-augmented generation**.

## Pipeline

1. Train the BiomedBERT and ViT-B/16 multimodal retriever.
2. Assemble and decontaminate the biomedical knowledge base.
3. Build the IVF-PQ index and retrieve 30 candidates per query.
4. Generate paired no-evidence and evidence-conditioned responses.
5. Annotate complete evidence subsets with GPT-4o.
6. Keep twelve of the fifteen annotated subsets per query for release.
7. Fine-tune the Qwen2.5-VL-7B-Instruct proxy with QLoRA and ms-swift.
8. Select up to five evidence items with streaming subset deliberation.
9. Generate the final answer with Qwen2.5-VL-72B-Instruct.
10. Evaluate VQA accuracy, open-ended generation, and component latency.

## Environment

Python 3.10 or newer is required.

```bash
pip install -e ".[train,eval,qwen,swift]"
```

Core runtime versions:

| Package | Version |
|---|---|
| PyTorch | 2.4.x |
| Transformers | 4.49.x |
| FAISS | 1.8.x |
| ms-swift | 3.12.6 |
| CUDA | 12.4 |

METEOR evaluation requires Java.

Source checks:

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests scripts
```

Expected runtime directories are `data/`, `checkpoints/`, and `outputs/`.

## Configuration

| Component | Setting |
|---|---|
| Retriever | BiomedBERT-base + ViT-B/16 ImageNet-21k |
| Embedding dimension | 768 |
| Retriever inputs | 512 text tokens, 224 x 224 images |
| Retriever objective | symmetric contrastive + intra-modal reconstruction + bidirectional cross-modal reconstruction |
| Retriever optimization | AdamW, 30 epochs, LR \(10^{-4}\), 10% linear warmup |
| Retriever batch | 256 effective: 4 GPUs x 32 per GPU x 2 accumulation |
| Reconstruction weights | \(\lambda_1=\lambda_2=0.6\) |
| Index | normalized inner product, IVF-PQ, nlist 4096, m 64, 8 bits, nprobe 32 |
| Retrieval and selection | \(M=30\), \(n=5\), \(\alpha_z=0.6\) |
| Teacher pool | 118,650 annotated subsets (7,910 queries x 15), 94,920 released |
| Proxy | Qwen2.5-VL-7B-Instruct, 4-bit QLoRA, rank 8, alpha 16 |
| Proxy optimization | 3 epochs, LR \(10^{-4}\), global batch 16 on 2 GPUs, seed 21 |
| Generator | Qwen2.5-VL-72B-Instruct, temperature 0.01, top-p 0.001 |

## Data Preparation

Canonical JSONL formats are defined in `docs/DATA_SCHEMA.md`.

Prepare approximately 200-token text passages:

```bash
INPUT_JSONL=data/source/pubmed.jsonl \
OUTPUT_JSONL=data/kb/pubmed_passages.jsonl \
AUDIT_JSON=outputs/audit/pubmed_passages.json \
SOURCE=PubMed \
bash scripts/prepare_text_passages.sh
```

Assemble the knowledge base:

```bash
KB_INPUTS="data/kb/mimic_cxr.jsonl data/kb/pmc_oa.jsonl data/kb/roco.jsonl data/kb/llava_med_align.jsonl data/kb/pubmed_passages.jsonl data/kb/statpearls_passages.jsonl data/kb/textbook_passages.jsonl data/kb/benchmark_train.jsonl" \
bash scripts/assemble_kb.sh
```

Apply held-out partition, image, text, QA-overlap, and manual-exclusion filters:

```bash
bash scripts/decontaminate_kb.sh
```

## Retriever and Index

```bash
bash scripts/run_retriever_train.sh
bash scripts/export_inference_retriever.sh
bash scripts/build_index.sh
bash scripts/retrieve_candidates.sh
```

Index metadata records the embedding dimension, normalization, index
parameters, and retriever checkpoint fingerprint.

## Proxy Distillation

Each query produces five top-ranked prefixes of sizes 1 to 5 and ten unique
five-item subsets, so 7,910 queries yield 118,650 annotated subsets. The
released corpus keeps twelve of the fifteen per query, namely all five
prefixes and seven of the ten five-item subsets, preferring the globally
rarest utility scores, giving 94,920 instances.

```bash
export RESPONSE_API_KEY=<7b-response-api-key>
export RESPONSE_API_BASE_URL=<7b-response-api-base-url>
export TEACHER_API_KEY=<teacher-api-key>
export TEACHER_API_BASE_URL=<teacher-api-base-url>

bash scripts/sample_proxy_subsets.sh
bash scripts/generate_proxy_responses.sh
bash scripts/annotate_proxy_teacher.sh
bash scripts/filter_proxy_annotations.sh
bash scripts/select_release_subsets.sh
bash scripts/build_proxy_data.sh
bash scripts/prepare_proxy_swift_data.sh
```

Train the proxy on two GPUs:

```bash
bash scripts/train_proxy_qlora_swift.sh
```

Serve the QLoRA adapter:

```bash
PROXY_ADAPTER=outputs/proxy/qwen2_5_vl_7b_qlora_seed21/checkpoint-<step> \
  bash scripts/deploy_proxy_swift.sh
```

## Selection and Generation

```bash
export PROXY_API_KEY=EMPTY
export PROXY_API_BASE_URL=http://127.0.0.1:8000/v1
bash scripts/run_selection.sh

export GENERATOR_API_KEY=<generator-api-key>
export GENERATOR_API_BASE_URL=<generator-api-base-url>
bash scripts/run_generation.sh
```

The selector performs one proxy call per candidate, 30 per query. The empty
subset is never scored; the running score starts at the neutral point of the
1-10 ordinal scale, so the first candidate is retained only when its proposal
is already net beneficial, and every later candidate only when it improves on
the best proposal so far. An empty subset is a legal outcome, in which case
the generator answers from parametric knowledge.

## Evaluation

VQA semantic-equivalence evaluation:

```bash
python -m biomrag.evaluation.vqa_judge \
  --predictions outputs/predictions/vqa_rad_test.jsonl \
  --references data/eval/vqa_rad_test.jsonl \
  --output outputs/eval/vqa_rad_judge.jsonl \
  --expected-count 451
```

LLaVA-Med open-ended evaluation:

```bash
python -m biomrag.evaluation.open_ended_metrics \
  --predictions outputs/predictions/llava_med_inst_test.jsonl \
  --references data/eval/llava_med_inst_test.jsonl \
  --output outputs/eval/llava_med_metrics.json \
  --expected-count 5665
```

Latency summary:

```bash
bash scripts/summarize_latency.sh
```

## API Variables

Component-specific clients read `RESPONSE_*`, `PROXY_*`, `GENERATOR_*`,
`TEACHER_*`, and `JUDGE_*`. `API_KEY` and `API_BASE_URL` are shared endpoint
variables.
