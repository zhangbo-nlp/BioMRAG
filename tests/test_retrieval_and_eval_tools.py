from argparse import Namespace

from biomrag.data.decontaminate_index import Blocklist, leakage_reason
from biomrag.data.prepare_text_passages import segment_text


def test_decontamination_semantic_filters():
    args = Namespace(phash_threshold=6, ngram_size=5, min_ngram_hits=1)
    blocked = Blocklist()

    caption_match = leakage_reason(
        {"id": "k1"},
        blocked,
        args,
        caption_semantic_ids={"k1"},
    )
    qa_match = leakage_reason(
        {"id": "k2"},
        blocked,
        args,
        qa_semantic_ids={"k2"},
    )

    assert caption_match == "caption_or_report_similarity"
    assert qa_match == "question_answer_overlap"


def test_text_passages_follow_token_boundaries():
    class WhitespaceTokenizer:
        def __call__(self, text, **_kwargs):
            offsets = []
            cursor = 0
            for token in text.split():
                start = text.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end
            return {"offset_mapping": offsets}

    chunks = segment_text(
        "one two three four five",
        tokenizer=WhitespaceTokenizer(),
        target_tokens=2,
        overlap_tokens=0,
    )
    assert chunks == ["one two", "three four", "five"]
