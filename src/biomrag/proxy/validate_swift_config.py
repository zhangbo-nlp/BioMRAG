"""Validate the BioMRAG ms-swift QLoRA configuration."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version

import yaml

EXPECTED_MS_SWIFT_VERSION = "3.12.6"
EXPECTED_TRANSFORMERS_PREFIX = "4.49."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_proxy_config(config, world_size=args.world_size)
    if args.check_runtime:
        validate_runtime_versions()


def validate_proxy_config(config: dict, *, world_size: int) -> None:
    expected = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_type": "qwen2_5_vl",
        "use_hf": False,
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "train_type": "lora",
        "target_modules": "all-linear",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "quant_bits": 4,
        "quant_method": "bnb",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "freeze_vit": True,
        "freeze_aligner": True,
        "freeze_llm": False,
        "gradient_checkpointing": True,
        "num_train_epochs": 3,
        "learning_rate": 1.0e-4,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.0,
        "weight_decay": 0.1,
        "max_length": 4096,
        "deepspeed": "zero2",
        "split_dataset_ratio": 0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Proxy QLoRA configuration mismatch: {mismatches}")
    if world_size != 2:
        raise ValueError(f"Proxy QLoRA requires two workers; got {world_size}.")
    effective_batch = (
        int(config.get("per_device_train_batch_size", 0))
        * int(config.get("gradient_accumulation_steps", 0))
        * world_size
    )
    if effective_batch != 16:
        raise ValueError(
            f"Proxy effective global batch is {effective_batch}; expected 16."
        )
    if not config.get("dataset") or not config.get("val_dataset"):
        raise ValueError("Proxy QLoRA requires explicit train and validation data.")


def validate_runtime_versions() -> None:
    actual = {
        package: _installed_version(package)
        for package in ("ms-swift", "transformers")
    }
    if actual["ms-swift"] != EXPECTED_MS_SWIFT_VERSION:
        raise RuntimeError(
            "ms-swift runtime mismatch: "
            f"{actual['ms-swift']!r} installed, "
            f"{EXPECTED_MS_SWIFT_VERSION!r} required."
        )
    if not actual["transformers"].startswith(EXPECTED_TRANSFORMERS_PREFIX):
        raise RuntimeError(
            "Transformers runtime mismatch: "
            f"{actual['transformers']!r} installed, "
            f"{EXPECTED_TRANSFORMERS_PREFIX}x required."
        )


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Required package {package!r} is not installed.") from exc


if __name__ == "__main__":
    main()
