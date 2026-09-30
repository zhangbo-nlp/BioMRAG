"""Export a retriever checkpoint without training-only reconstruction modules."""

from __future__ import annotations

import argparse

from transformers import AutoImageProcessor, AutoTokenizer

from biomrag.models import BioMRAGVisionTextDualEncoderModel

TRAINING_ONLY_MODULES = (
    "mlm_head",
    "vit_decoder",
    "visual_cross_block",
    "text_cross_block",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Trained retriever checkpoint.")
    parser.add_argument("--output", required=True, help="Inference checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = BioMRAGVisionTextDualEncoderModel.from_pretrained(args.input)
    strip_reconstruction_modules(model)
    model.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.input).save_pretrained(args.output)
    AutoImageProcessor.from_pretrained(args.input).save_pretrained(args.output)


def strip_reconstruction_modules(model: BioMRAGVisionTextDualEncoderModel) -> None:
    for name in TRAINING_ONLY_MODULES:
        if hasattr(model, name):
            delattr(model, name)
    model.text_reconstruction = False
    model.image_reconstruction = False
    model.cross_reconstruction = False
    model.mlm_pretrain = False
    model.mim_pretrain = False
    model.config.text_reconstruction = False
    model.config.image_reconstruction = False
    model.config.cross_reconstruction = False
    model.config.mlm_pretrain = False
    model.config.mim_pretrain = False
    embeddings = getattr(model.vision_model, "embeddings", None)
    if embeddings is not None and hasattr(embeddings, "mask_token"):
        embeddings.mask_token = None


if __name__ == "__main__":
    main()
