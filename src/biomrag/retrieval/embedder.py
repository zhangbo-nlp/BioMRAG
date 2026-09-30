"""Embedding helpers for the BioMRAG retriever."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from biomrag.models import BioMRAGVisionTextDualEncoderModel


@dataclass
class BioMRAGEmbedder:
    model_name_or_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512

    def __post_init__(self) -> None:
        self.model = BioMRAGVisionTextDualEncoderModel.from_pretrained(
            self.model_name_or_path
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)

    @torch.inference_mode()
    def encode(
        self,
        *,
        text: str = "",
        image_path: str = "",
    ) -> np.ndarray:
        matrix = self.encode_batch(
            [{"text": text, "image_path": image_path}],
        )
        return matrix[0]

    @torch.inference_mode()
    def encode_batch(self, rows: list[dict], text_key: str = "text") -> np.ndarray:
        if not rows:
            return np.empty((0, self.model.projection_dim), dtype="float32")

        grouped: dict[tuple[bool, bool], list[tuple[int, str, str]]] = {}
        for index, row in enumerate(rows):
            text = str(row.get(text_key, "") or "")
            image_path = str(row.get("image_path", "") or "")
            key = (bool(text), bool(image_path))
            if key == (False, False):
                raise ValueError(f"Cannot embed empty item at batch index {index}.")
            grouped.setdefault(key, []).append((index, text, image_path))

        output = np.empty(
            (len(rows), self.model.projection_dim),
            dtype="float32",
        )
        for (has_text, has_image), entries in grouped.items():
            text_inputs = None
            if has_text:
                text_inputs = self.tokenizer(
                    [entry[1] for entry in entries],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                text_inputs = {
                    key: value.to(self.device)
                    for key, value in text_inputs.items()
                }

            pixel_values = None
            if has_image:
                images = []
                for _index, _text, image_path in entries:
                    with Image.open(Path(image_path)) as image:
                        images.append(image.convert("RGB"))
                pixel_values = self.image_processor(
                    images,
                    return_tensors="pt",
                )["pixel_values"].to(self.device)

            features = self.model.get_features(
                input_ids=(
                    None if text_inputs is None else text_inputs["input_ids"]
                ),
                attention_mask=(
                    None
                    if text_inputs is None
                    else text_inputs.get("attention_mask")
                ),
                pixel_values=pixel_values,
                normalize=True,
            )
            vectors = features.detach().cpu().float().numpy()
            for local_index, (row_index, _text, _image_path) in enumerate(entries):
                output[row_index] = vectors[local_index]
        return output
