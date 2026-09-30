# Reproducibility

## Partitioning and Decontamination

External corpora and official benchmark **training** partitions may contribute
to retriever training, proxy distillation, and the primary index. Validation
and test partitions are reserved for selection/evaluation and must be removed
from all training and index manifests.

After filtering, no test image, question or reference answer, and no
near-duplicate of them, remains in the index or in any training pool. Removal
counts are dataset- and release-dependent, so they are reported by the audit
file rather than fixed in advance.

The primary index applies these ordered filters:

| Criterion | Rule |
|---|---|
| Held-out partition | exact sample identifier |
| Exact image | SHA-256 equality |
| Near-duplicate image | 64-bit dHash Hamming distance <= 6 |
| Caption/report | precomputed cosine similarity >= 0.92 |
| Question-answer overlap | normalized 5-gram or semantic match |
| Manual audit | adjudicated exclusion |

`scripts/decontaminate_kb.sh` requires separate caption/report semantic-match,
QA semantic-match, and manual-exclusion manifests, writes the per-criterion
removal counts to the audit JSON, and fails when the decontaminated index drops
below `--minimum-kept`. Unreadable images fail the audit by default.

## Retriever

- text tower: BiomedBERT-base;
- image tower: `google/vit-base-patch16-224-in21k`;
- projection dimension: 768;
- text length: 512;
- image size: 224 x 224;
- initial learnable temperature: 0.07;
- objective: symmetric contrastive loss plus text/image intra-modal
  reconstruction and bidirectional cross-modal reconstruction;
- intra-modal and cross-modal weights: 0.6 and 0.6;
- optimizer: AdamW, LR \(10^{-4}\);
- schedule: 10% linear warmup followed by linear decay;
- 30 epochs;
- effective training batch: 256 = 4 workers x 32 examples x 2 accumulation.

Distributed contrastive logits use all workers' current microbatches with
autograd-preserving gathers. Equal local batch sizes are enforced.
Training-only MLM, image decoder, cross-attention, and visual mask parameters
are removed by the inference export.

## Index and Retrieval

The primary index uses normalized 768-dimensional vectors and inner-product
FAISS IVF-PQ:

- `nlist = 4096`;
- `m = 64`;
- `nbits = 8`;
- `nprobe = 32`;
- `M = 30`;
- image-text fusion weight `alpha_z = 0.6`.

Text-only items use normalized text vectors. Image-text items normalize the
weighted sum of normalized text and image vectors.

## Proxy Supervision

Each proxy-distillation query has 30 retrieved candidates. Exactly fifteen
teacher requests are formed:

- top-ranked prefixes of sizes 1, 2, 3, 4, and 5;
- ten unique random five-item subsets.

The same frozen Qwen2.5-VL-7B-Instruct response backbone produces
no-evidence and evidence-conditioned answers. GPT-4o receives the query,
reference, both responses, and complete evidence subset at temperature 0.
There are 118,650 teacher requests over 7,910 distillation queries; subset
sampling validates both totals and requires contiguous top-30 retrieval ranks.

The released corpus keeps twelve of the fifteen annotated subsets per query,
namely the five prefixes and seven of the ten five-item subsets, preferring
the globally rarest utility scores so that scarce low- and high-utility
supervision survives the reduction. This gives 7,910 x 12 = 94,920 instances.
The selection is deterministic and requires no seed.

Filtering rejects malformed scores, scores outside the integer 1-10 scale,
empty rationales, rationales without response-change language, and reference
restatements. Query groups cannot cross the proxy train/validation split.

## Proxy QLoRA

- runtime: ms-swift 3.12.6 and Transformers 4.49.x;
- Qwen2.5-VL-7B-Instruct;
- 4-bit bitsandbytes NF4 with bfloat16 compute and double quantization;
- rank 8, alpha 16, dropout 0.05, all LLM linear targets;
- frozen visual tower and aligner;
- three epochs;
- LR \(10^{-4}\);
- AdamW with cosine decay and no warmup;
- maximum length 4096;
- global batch 16 = 2 workers x 1 example x 8 accumulation.

The optimization and data seed is 21.
The adapter is served by `swift deploy` with `infer_backend=pt`, temperature
zero, and the same base model and image-resolution limit used during training.

## Streaming Selection

The subset starts empty and the running score starts at the neutral point of
the ordinal 1-10 utility scale, which is 5. The empty subset is never scored,
because the proxy has no proposal to judge. Candidates are processed in
retrieval order starting with `k_1`, and each produces one complete proposal.
Before the budget is full the candidate is appended; once full, the proposal
removes the retained item with maximum mean positive cosine similarity to the
other retained items.

The proxy emits a non-empty rationale and one integer score in 1-10. A
proposal is accepted only when its score is strictly greater than the current
score, so equal scores keep the current subset. Each query therefore makes
exactly `M` proxy calls and retains at most `n=5` items. When no proposal
outscores the neutral baseline, the subset stays empty and generation falls
back to the no-evidence prompt.

## Generation

The final generator is API-hosted Qwen2.5-VL-72B-Instruct with temperature
0.01 and top-p 0.001. It receives the current query, prior dialogue, and only
the selected text and visual evidence. Proxy scores, rationales, reference
answers, and teacher responses are excluded. Each evaluated query uses one
generator call.

## Evaluation

- VQA-RAD, SLAKE, and PathVQA use full test-set Qwen2.5-7B-Instruct
  semantic-equivalence judgments.
- Predictions and references must have identical unique ID sets.
- LLaVA-Med instruction following uses corpus BLEU-4, ROUGE-L, METEOR, and
  CIDEr.

Latency is measured serially at batch size one after model/API initialization.
It includes request construction, image preprocessing, retrieval, proxy
decoding, API response time, and final generation. Index construction is
excluded.

## Required Artifacts

Numerical reproduction requires the exact source manifests, split registries,
7,910-query distillation manifest, semantic-match manifests, manual exclusion
manifest, trained retriever/proxy checkpoints, and API model revisions.
Runtime paths use `data/`, `checkpoints/`, and environment variables.
