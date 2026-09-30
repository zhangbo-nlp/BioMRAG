import json
from pathlib import Path

import pytest
import yaml

from biomrag.generation.run_generation import _history_before_current_query
from biomrag.proxy.prompts import format_query_context
from biomrag.proxy.validate_swift_config import validate_proxy_config
from biomrag.retrieval.checkpoint_fingerprint import checkpoint_fingerprint
from biomrag.schemas import EvidenceItem, QueryExample
from biomrag.utils.multimodal import content_from_image_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_retriever_config_has_effective_global_batch_256():
    config = json.loads(
        (PROJECT_ROOT / "configs/retriever_train.example.json").read_text(
            encoding="utf-8"
        )
    )
    effective = (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
        * config["expected_world_size"]
    )
    assert effective == config["expected_effective_batch_size"] == 256
    assert config["vision_model_name_or_path"].endswith("-in21k")
    assert config["image_processor_name"].endswith("-in21k")


def test_proxy_swift_config_has_effective_global_batch_16():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/proxy_qlora_swift.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_proxy_config(config, world_size=2)
    assert config["train_type"] == "lora"
    assert "tuner_type" not in config


def test_qwen_runtime_versions_are_pinned():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert pyproject.count('"transformers>=4.49,<4.50"') == 2
    assert '"ms-swift==3.12.6"' in pyproject
    assert '"pycocoevalcap==1.2"' in pyproject


def test_biomrag_configs_are_fixed():
    retriever = json.loads(
        (PROJECT_ROOT / "configs/retriever_train.example.json").read_text(
            encoding="utf-8"
        )
    )
    assert retriever["projection_dim"] == 768
    assert retriever["learning_rate"] == 1.0e-4
    assert retriever["warmup_ratio"] == 0.1
    assert retriever["num_train_epochs"] == 30
    assert retriever["initial_temperature"] == 0.07
    assert retriever["lambda_intra"] == retriever["lambda_cross"] == 0.6
    assert retriever["text_reconstruction"] is True
    assert retriever["image_reconstruction"] is True
    assert retriever["cross_reconstruction"] is True

    indexing = yaml.safe_load(
        (PROJECT_ROOT / "configs/indexing.yaml").read_text(encoding="utf-8")
    )
    assert indexing["batch_size"] == 256
    assert {
        key: indexing[key]
        for key in ("index_type", "nlist", "m", "nbits", "nprobe")
    } == {
        "index_type": "ivfpq",
        "nlist": 4096,
        "m": 64,
        "nbits": 8,
        "nprobe": 32,
    }

    selection = yaml.safe_load(
        (PROJECT_ROOT / "configs/selection.yaml").read_text(encoding="utf-8")
    )
    assert selection["retrieved_candidates"] == 30
    assert selection["evidence_budget"] == 5

    generation = yaml.safe_load(
        (PROJECT_ROOT / "configs/generation.yaml").read_text(encoding="utf-8")
    )
    assert generation == {
        "generator_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "temperature": 0.01,
        "top_p": 0.001,
        "max_tokens": 256,
    }


def test_dataset_and_decontamination_counts_are_fixed():
    datasets = yaml.safe_load(
        (PROJECT_ROOT / "configs/datasets.yaml").read_text(encoding="utf-8")
    )["datasets"]
    assert {
        name: datasets[name]["test"]
        for name in ("vqa_rad", "slake", "pathvqa", "llava_med_inst")
    } == {
        "vqa_rad": 451,
        "slake": 1061,
        "pathvqa": 6719,
        "llava_med_inst": 5665,
    }
    assert datasets["llava_med_inst"]["validation"] == 5666


def test_schema_extracts_dialogue_query_image_and_evidence_aliases():
    query = QueryExample.from_dict(
        {
            "id": "q1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "query.png"},
                        {"type": "text", "text": "What is shown?"},
                    ],
                }
            ],
            "reference": "answer",
        }
    )
    evidence = EvidenceItem.from_dict(
        {
            "id": "k1",
            "caption": "caption",
            "image": "evidence.png",
        }
    )

    assert query.question == "What is shown?"
    assert query.answer == "answer"
    assert query.image_path == "query.png"
    assert evidence.text == "caption"
    assert evidence.modality == "image-text"


def test_schema_extracts_target_assistant_answer_without_leaking_it():
    query = QueryExample.from_dict(
        {
            "id": "q2",
            "conversations": [
                {"from": "human", "value": "Prior question"},
                {"from": "gpt", "value": "Prior answer"},
                {"from": "human", "value": "<image>Current question"},
                {"from": "gpt", "value": "Target reference"},
            ],
            "image": "query.png",
        }
    )
    assert query.question == "Current question"
    assert query.answer == "Target reference"
    assert "Target reference" not in format_query_context(query)


def test_multimodal_helper_replaces_tokens_in_order(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    content = content_from_image_tokens(
        "A<image>B<image>C",
        [str(first), str(second)],
    )
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"] != image_parts[1]["image_url"]["url"]
    with pytest.raises(ValueError, match="contains 2"):
        content_from_image_tokens("A<image>B<image>C", [str(first)])


def test_generation_history_excludes_current_turn_and_target_answer():
    history = [
        {"from": "human", "value": "Earlier question"},
        {"from": "gpt", "value": "Earlier answer"},
        {"from": "human", "value": "<image>\nCurrent question"},
        {"from": "gpt", "value": "Reference target"},
    ]
    prior = _history_before_current_query(
        history,
        current_question="Current question",
    )
    assert prior == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]

    query = QueryExample(
        id="q",
        question="Current question",
        history=history,
    )
    context = format_query_context(query)
    assert "Earlier question" in context
    assert "Earlier answer" in context
    assert "Current question" in context
    assert "Reference target" not in context


def test_checkpoint_fingerprint_changes_with_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"first")
    first = checkpoint_fingerprint(tmp_path)
    weights.write_bytes(b"second")
    second = checkpoint_fingerprint(tmp_path)
    assert first != second
