#!/usr/bin/env python
"""Train the BioMRAG dual-encoder retriever."""

from __future__ import annotations

import copy
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from biomrag.models import BioMRAGVisionTextDualEncoderModel

logger = logging.getLogger(__name__)

TEXT_MODEL = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
VISION_MODEL = "google/vit-base-patch16-224-in21k"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Existing BioMRAG retriever checkpoint to continue from."},
    )
    vision_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Vision backbone, used when initializing a new retriever."},
    )
    text_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Text backbone, used when initializing a new retriever."},
    )
    tokenizer_name: Optional[str] = None
    image_processor_name: Optional[str] = None
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    use_fast_tokenizer: bool = True
    token: Optional[str] = None
    trust_remote_code: bool = False
    freeze_vision_model: bool = False
    freeze_text_model: bool = False
    projection_dim: int = 768
    initial_temperature: float = 0.07
    lambda_intra: float = 0.6
    lambda_cross: float = 0.6
    fusion_text_weight: float = 0.6
    text_reconstruction: bool = True
    image_reconstruction: bool = True
    cross_reconstruction: bool = True
    expected_world_size: int = 4
    expected_effective_batch_size: int = 256


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    data_dir: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    image_column: str = "image_path"
    caption_column: str = "caption"
    mlm_probability: float = 0.15
    mask_patch_size: int = 32
    mask_ratio: float = 0.6
    max_seq_length: int = 512
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Provide --dataset_name or --train_file.")
        for attr in ("train_file", "validation_file"):
            value = getattr(self, attr)
            if value is not None:
                extension = value.split(".")[-1]
                if extension not in {"csv", "json", "jsonl"}:
                    raise ValueError(f"{attr} must be csv, json, or jsonl.")


class MaskGenerator:
    """Generate ViT boolean masks for masked image reconstruction."""

    def __init__(
        self,
        input_size: int = 224,
        mask_patch_size: int = 32,
        model_patch_size: int = 16,
        mask_ratio: float = 0.6,
    ):
        if input_size % mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size.")
        if mask_patch_size % model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size.")
        self.rand_size = input_size // mask_patch_size
        self.scale = mask_patch_size // model_patch_size
        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * mask_ratio))

    def __call__(self) -> torch.Tensor:
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return torch.tensor(mask.flatten(), dtype=torch.bool)


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": torch.tensor([example["input_ids"] for example in examples]),
        "masked_input_ids": torch.tensor(
            [example["masked_input_ids"] for example in examples]
        ),
        "attention_mask": torch.tensor(
            [example["attention_mask"] for example in examples]
        ),
        "mlm_labels": torch.tensor([example["mlm_labels"] for example in examples]),
        "bool_masked_pos": torch.stack([example["mask"] for example in examples]),
        "return_loss": True,
    }


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _setup_logging(training_args)
    _validate_training_config(model_args, data_args, training_args)
    set_seed(training_args.seed)
    last_checkpoint = _detect_checkpoint(training_args)
    dataset = _load_dataset(data_args, model_args)

    tokenizer_source = (
        model_args.tokenizer_name
        or model_args.text_model_name_or_path
        or model_args.model_name_or_path
    )
    image_processor_source = (
        model_args.image_processor_name
        or model_args.vision_model_name_or_path
        or model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        image_processor_source,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = _load_model(model_args)

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)
    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    column_names = _column_names(dataset, training_args)
    _validate_columns(column_names, data_args)

    train_dataset, eval_dataset = _prepare_datasets(
        dataset=dataset,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        image_processor=image_processor,
        vision_config=model.vision_model.config,
        column_names=column_names,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _setup_logging(training_args: TrainingArguments) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        "rank=%s device=%s n_gpu=%s distributed=%s fp16=%s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        training_args.fp16,
    )


def _validate_training_config(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> None:
    if data_args.max_seq_length != 512:
        raise ValueError("BioMRAG retriever requires max_seq_length=512.")
    expected_model_values = {
        "vision_model_name_or_path": VISION_MODEL,
        "text_model_name_or_path": TEXT_MODEL,
        "tokenizer_name": TEXT_MODEL,
        "image_processor_name": VISION_MODEL,
        "projection_dim": 768,
        "freeze_vision_model": False,
        "freeze_text_model": False,
        "text_reconstruction": True,
        "image_reconstruction": True,
        "cross_reconstruction": True,
        "expected_world_size": 4,
        "expected_effective_batch_size": 256,
    }
    model_mismatches = {
        key: {"actual": getattr(model_args, key), "expected": expected}
        for key, expected in expected_model_values.items()
        if getattr(model_args, key) != expected
    }
    expected_float_values = {
        "initial_temperature": 0.07,
        "lambda_intra": 0.6,
        "lambda_cross": 0.6,
        "fusion_text_weight": 0.6,
    }
    model_mismatches.update(
        {
            key: {"actual": getattr(model_args, key), "expected": expected}
            for key, expected in expected_float_values.items()
            if not math.isclose(
                float(getattr(model_args, key)),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }
    )
    if model_mismatches:
        raise ValueError(
            f"BioMRAG retriever model mismatch: {model_mismatches}"
        )
    if not training_args.do_train:
        return
    training_mismatches = {}
    exact_training_values = {
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "dataloader_drop_last": True,
    }
    for key, expected in exact_training_values.items():
        actual = getattr(training_args, key)
        if actual != expected:
            training_mismatches[key] = {
                "actual": actual,
                "expected": expected,
            }
    float_training_values = {
        "learning_rate": 1.0e-4,
        "warmup_ratio": 0.1,
        "num_train_epochs": 30.0,
    }
    for key, expected in float_training_values.items():
        actual = float(getattr(training_args, key))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            training_mismatches[key] = {
                "actual": actual,
                "expected": expected,
            }
    enum_training_values = {
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
    }
    for key, expected in enum_training_values.items():
        raw = getattr(training_args, key)
        actual = str(getattr(raw, "value", raw))
        if actual != expected:
            training_mismatches[key] = {
                "actual": actual,
                "expected": expected,
            }
    if training_mismatches:
        raise ValueError(
            f"BioMRAG retriever optimization mismatch: {training_mismatches}"
        )
    if (
        model_args.expected_world_size
        and training_args.world_size != model_args.expected_world_size
    ):
        raise ValueError(
            f"Retriever training requires {model_args.expected_world_size} workers; "
            f"got {training_args.world_size}."
        )
    effective_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if (
        model_args.expected_effective_batch_size
        and effective_batch != model_args.expected_effective_batch_size
    ):
        raise ValueError(
            "Retriever effective batch mismatch: "
            f"{training_args.per_device_train_batch_size} per device x "
            f"{training_args.world_size} workers x "
            f"{training_args.gradient_accumulation_steps} accumulation = "
            f"{effective_batch}, expected "
            f"{model_args.expected_effective_batch_size}."
        )


def _detect_checkpoint(training_args: TrainingArguments) -> str | None:
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory {training_args.output_dir} exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        return checkpoint
    return None


def _load_dataset(data_args: DataTrainingArguments, model_args: ModelArguments):
    if data_args.dataset_name is not None:
        return load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    data_files = {"train": data_args.train_file}
    extension = data_args.train_file.split(".")[-1]
    if extension == "jsonl":
        extension = "json"
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    return load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)


def _load_model(model_args: ModelArguments) -> BioMRAGVisionTextDualEncoderModel:
    if model_args.model_name_or_path:
        model = BioMRAGVisionTextDualEncoderModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation="sdpa",
        )
    else:
        model = BioMRAGVisionTextDualEncoderModel.from_vision_text_pretrained(
            vision_model_name_or_path=model_args.vision_model_name_or_path,
            text_model_name_or_path=model_args.text_model_name_or_path,
            projection_dim=model_args.projection_dim,
            logit_scale_init_value=math.log(1.0 / model_args.initial_temperature),
            text_reconstruction=model_args.text_reconstruction,
            image_reconstruction=model_args.image_reconstruction,
            cross_reconstruction=model_args.cross_reconstruction,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    model.config.lambda_intra = model_args.lambda_intra
    model.config.lambda_cross = model_args.lambda_cross
    model.config.fusion_text_weight = model_args.fusion_text_weight
    model.config.text_reconstruction = model_args.text_reconstruction
    model.config.image_reconstruction = model_args.image_reconstruction
    model.config.cross_reconstruction = model_args.cross_reconstruction
    model.config.mlm_pretrain = model_args.text_reconstruction
    model.config.mim_pretrain = model_args.image_reconstruction
    model.lambda_intra = model_args.lambda_intra
    model.lambda_cross = model_args.lambda_cross
    model.fusion_text_weight = model_args.fusion_text_weight
    model.text_reconstruction = model_args.text_reconstruction
    model.image_reconstruction = model_args.image_reconstruction
    model.cross_reconstruction = model_args.cross_reconstruction
    model.mlm_pretrain = model_args.text_reconstruction
    model.mim_pretrain = model_args.image_reconstruction
    return model


def _freeze_params(module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _column_names(dataset, training_args: TrainingArguments) -> list[str]:
    if training_args.do_train:
        return dataset["train"].column_names
    if training_args.do_eval:
        return dataset["validation"].column_names
    raise ValueError("Pass --do_train and/or --do_eval.")


def _validate_columns(column_names: list[str], data_args: DataTrainingArguments) -> None:
    for column in (data_args.image_column, data_args.caption_column):
        if column not in column_names:
            raise ValueError(f"Column {column!r} must be one of {column_names}.")


def _prepare_datasets(
    *,
    dataset,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer,
    image_processor,
    vision_config,
    column_names: list[str],
):
    size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize]
    )
    val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])
    mask_generator = MaskGenerator(
        input_size=vision_config.image_size,
        mask_patch_size=data_args.mask_patch_size,
        model_patch_size=vision_config.patch_size,
        mask_ratio=data_args.mask_ratio,
    )

    def tokenize_captions(examples):
        captions = list(examples[data_args.caption_column])
        text_inputs = tokenizer(
            captions,
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
        )
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask

        inputs = np.array(copy.deepcopy(text_inputs.input_ids))
        labels = np.copy(inputs)
        special_mask = np.array(text_inputs.special_tokens_mask, dtype=bool)
        probability = np.full(labels.shape, data_args.mlm_probability)
        probability[special_mask] = 0
        masked = np.random.binomial(1, probability, size=probability.shape).astype(bool)
        labels[~masked] = -100
        replace_mask = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked
        inputs[replace_mask] = tokenizer.mask_token_id
        random_mask = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool)
            & masked
            & ~replace_mask
        )
        random_words = np.random.randint(
            low=0, high=len(tokenizer), size=np.count_nonzero(random_mask)
        )
        inputs[random_mask] = random_words
        examples["masked_input_ids"] = inputs
        examples["mlm_labels"] = labels
        return examples

    def transform_images(examples, transforms):
        pixel_values = []
        for image_file in examples[data_args.image_column]:
            with Image.open(image_file) as image:
                pixel_values.append(transforms(image.convert("RGB")))
        examples["pixel_values"] = pixel_values
        examples["mask"] = [mask_generator() for _ in examples[data_args.image_column]]
        return examples

    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != data_args.image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing train captions",
        )
        train_dataset.set_transform(lambda examples: transform_images(examples, train_transforms))

    if training_args.do_eval:
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != data_args.image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing validation captions",
        )
        eval_dataset.set_transform(lambda examples: transform_images(examples, val_transforms))

    return train_dataset, eval_dataset


if __name__ == "__main__":
    main()
