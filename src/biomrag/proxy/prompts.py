"""BioMRAG prompt templates."""

from __future__ import annotations

from collections.abc import Sequence

from biomrag.schemas import EvidenceItem, QueryExample

TEACHER_SYSTEM_PROMPT = """You are an expert biomedical evaluator. Assess the marginal answer utility of the complete retrieved knowledge subset. Determine how the evidence-conditioned answer changes the no-evidence answer for the same query and response model, using the reference answer as the correctness anchor. Reward evidence that improves the answer and penalize evidence that causes errors, unsupported detail, or regression. Explain the contribution of the subset as a whole. Do not assign scores to individual evidence items.

Use the following ordinal utility scale. Scores rank subsets within the same query; numerical gaps between scores need not represent equal utility increments:
1 = the subset substantially degrades the no-evidence response
2-3 = the subset introduces errors or distracting information
4-5 = the subset produces little or no meaningful improvement
6-7 = the subset yields a modest but incomplete improvement
8-9 = the subset substantially improves correctness or completeness
10 = the subset decisively enables a precise, correct response
"""

PROXY_SYSTEM_PROMPT = """You are an expert biomedical evaluator. Assess the expected answer utility of the complete retrieved knowledge subset for the biomedical visual question. Judge the subset as a whole, considering whether its relevance, sufficiency, complementarity, redundancy, or misleading content would improve or degrade the answer. Do not assign scores to individual items.

Use the following ordinal utility scale. Scores rank subsets for the same query; numerical gaps need not represent equal utility increments:
1 = the subset is expected to substantially degrade the answer
2-3 = the subset is expected to introduce errors or distraction
4-5 = the subset is expected to yield little meaningful improvement
6-7 = the subset is expected to yield a modest but incomplete improvement
8-9 = the subset is expected to substantially improve correctness or completeness
10 = the subset is expected to enable a precise, correct answer
"""

RAG_GENERATION_SYSTEM_PROMPT = """You are a helpful biomedical visual question-answering assistant. Answer the user's current question using the image and retrieved biomedical knowledge. Use retrieved knowledge only when it is relevant to the current question. If some retrieved items are noisy or unrelated, ignore them. Be concise and medically accurate."""

NO_RETRIEVAL_SYSTEM_PROMPT = """You are a helpful biomedical visual question-answering assistant. Answer the user's current question using the provided image. Be concise, medically accurate, and consistent with the previous turns. Do not introduce unsupported findings."""


def format_knowledge_items(
    evidence: Sequence[EvidenceItem],
    *,
    include_image_tokens: bool = False,
) -> str:
    blocks = []
    for item in evidence:
        source = f" source={item.source}" if item.source else ""
        modality = f" modality={item.modality}" if item.modality else ""
        rank = (
            f" rank={item.retrieval_rank}" if item.retrieval_rank is not None else ""
        )
        header = f"[{item.id}{source}{modality}{rank}]"
        visual = "<image>\n" if include_image_tokens and item.image_path else ""
        blocks.append(f"{header}\n{visual}{item.text}".strip())
    return "\n\n".join(blocks)


def build_teacher_user_prompt(
    *,
    query: QueryExample,
    evidence: Sequence[EvidenceItem],
    generated_answer: str,
    reference_answer: str,
    no_evidence_answer: str = "",
    include_image_tokens: bool = False,
) -> str:
    no_evidence = no_evidence_answer or "(not provided)"
    query_image = "<image>\n" if include_image_tokens and query.image_path else ""
    return f"""User Query or Dialogue Context:
{query_image}{format_query_context(query)}

No-evidence answer:
{no_evidence}

Candidate knowledge subset:
{format_knowledge_items(evidence, include_image_tokens=include_image_tokens)}

Evidence-conditioned answer:
{generated_answer or "(None)"}

Reference answer:
{reference_answer or query.answer or "(None)"}

Return only valid JSON in the following format:
{{
  "rationale": "<brief assessment of response improvement or degradation>",
  "score": <integer from 1 to 10>
}}
"""


def build_proxy_user_prompt(
    query: QueryExample,
    evidence: Sequence[EvidenceItem],
    *,
    include_image_tokens: bool = False,
) -> str:
    query_image = "<image>\n" if include_image_tokens and query.image_path else ""
    return f"""User Query or Dialogue Context:
{query_image}{format_query_context(query)}

Knowledge set:
{format_knowledge_items(evidence, include_image_tokens=include_image_tokens)}
"""


def build_rag_generation_prompt(evidence: Sequence[EvidenceItem]) -> str:
    knowledge = format_knowledge_items(evidence)
    return f"{RAG_GENERATION_SYSTEM_PROMPT}\n\nRetrieved biomedical knowledge:\n{knowledge}"


def format_query_context(query: QueryExample) -> str:
    """Serialize prior dialogue through the current user turn, excluding targets."""
    if not query.history:
        return query.question

    normalized_question = _normalize_text(query.question)
    current_index = -1
    for index, message in enumerate(query.history):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"user", "human"} and _normalize_text(
            _message_text(message)
        ) == normalized_question:
            current_index = index

    cutoff = current_index + 1 if current_index >= 0 else len(query.history)
    turns = []
    for message in query.history[:cutoff]:
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"user", "human"}:
            label = "User"
        elif role in {"assistant", "gpt", "model"}:
            label = "Assistant"
        else:
            continue
        text = _message_text(message)
        if text:
            turns.append(f"{label}: {text}")
    if current_index < 0:
        turns.append(f"User: {query.question}")
    return "\n".join(turns) or query.question


def _message_text(message: dict) -> str:
    content = message.get("content", message.get("value", ""))
    if isinstance(content, str):
        return content.replace("<image>", "").strip()
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(item.get("text", "")).strip()
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and item.get("text")
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()
