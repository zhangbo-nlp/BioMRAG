# coding=utf-8
"""BioMRAG vision-text dual encoder.

It keeps efficient dual-encoder inference while adding training-time masked
reconstruction losses used by BioMRAG.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Union

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    CLIPVisionConfig,
    CLIPVisionModel,
    VisionTextDualEncoderConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class CrossAttentionBlock(nn.Module):
    """Small cross-attention block used only for reconstruction objectives."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ):
        attended, _ = self.attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
        )
        hidden = self.norm1(query + self.dropout(attended))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return hidden


class BioMRAGVisionTextDualEncoderPreTrainedModel(PreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    base_model_prefix = "vision_text_dual_encoder"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _tied_weights_keys = [
        "mlm_head.predictions.decoder.bias",
        "mlm_head.predictions.bias",
    ]

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BioMRAGVisionTextDualEncoderModel(
    BioMRAGVisionTextDualEncoderPreTrainedModel
):
    """Dual encoder with auxiliary reconstruction losses.

    Configuration fields:
        text_reconstruction: enable intra-modal text masked reconstruction.
        image_reconstruction: enable intra-modal image masked reconstruction.
        cross_reconstruction: enable cross-modal masked reconstruction.
        lambda_intra: weight for intra-modal MLM/MIM.
        lambda_cross: weight for cross-modal reconstruction.
        fusion_text_weight: text weight when both modalities are embedded.
    """

    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
        mlm_head: Optional[BertOnlyMLMHead] = None,
    ):
        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Provide either a config or both vision and text models.")

        if config is None:
            config = VisionTextDualEncoderConfig.from_vision_text_configs(
                vision_model.config, text_model.config
            )
        elif not isinstance(config, self.config_class):
            raise ValueError(f"config must be {self.config_class}, got {type(config)}")

        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)
        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        for backbone, shared_config in (
            (vision_model, config.vision_config),
            (text_model, config.text_config),
        ):
            attn_implementation = getattr(
                backbone.config, "_attn_implementation", None
            )
            if attn_implementation is not None:
                shared_config._attn_implementation = attn_implementation

        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        self.text_reconstruction = bool(
            getattr(config, "text_reconstruction", getattr(config, "mlm_pretrain", True))
        )
        self.image_reconstruction = bool(
            getattr(config, "image_reconstruction", getattr(config, "mim_pretrain", True))
        )
        self.cross_reconstruction = bool(getattr(config, "cross_reconstruction", True))
        self.mlm_pretrain = self.text_reconstruction
        self.mim_pretrain = self.image_reconstruction
        self.lambda_intra = float(getattr(config, "lambda_intra", 0.6))
        self.lambda_cross = float(getattr(config, "lambda_cross", 0.6))
        self.fusion_text_weight = float(getattr(config, "fusion_text_weight", 0.6))

        if self.text_reconstruction or self.cross_reconstruction:
            if mlm_head is None:
                text_config = copy.deepcopy(config.text_config)
                text_config.hidden_size = self.projection_dim
                mlm_head = BertOnlyMLMHead(text_config)
            self.mlm_head = mlm_head

        if self.image_reconstruction or self.cross_reconstruction:
            self._init_visual_mask_token()
            patch_stride = getattr(config.vision_config, "encoder_stride", None)
            if patch_stride is None:
                patch_stride = getattr(config.vision_config, "patch_size", 16)
            self.vit_decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.projection_dim,
                    out_channels=patch_stride**2 * config.vision_config.num_channels,
                    kernel_size=1,
                ),
                nn.PixelShuffle(patch_stride),
            )
        if self.cross_reconstruction:
            self.visual_cross_block = CrossAttentionBlock(self.projection_dim, 4)
            self.text_cross_block = CrossAttentionBlock(self.projection_dim, 4)

    def _init_visual_mask_token(self) -> None:
        embeddings = getattr(self.vision_model, "embeddings", None)
        if embeddings is None:
            return
        if getattr(embeddings, "mask_token", None) is not None:
            return
        mask_token = torch.zeros(1, 1, self.vision_embed_dim)
        embeddings.mask_token = nn.Parameter(nn.init.normal_(mask_token, std=0.02))

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> torch.Tensor:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = text_outputs[1]
        return self.text_projection(pooled_output)

    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> torch.Tensor:
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = vision_outputs[1]
        return self.visual_projection(pooled_output)

    def get_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        text_features = None
        image_features = None

        if input_ids is not None:
            text_features = self.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if text_features is not None and image_features is not None:
            text_features = nn.functional.normalize(text_features, dim=-1)
            image_features = nn.functional.normalize(image_features, dim=-1)
            weight = self.fusion_text_weight
            features = weight * text_features + (1.0 - weight) * image_features
        elif text_features is not None:
            features = text_features
        elif image_features is not None:
            features = image_features
        else:
            raise ValueError("At least one of input_ids or pixel_values is required.")

        if normalize:
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        masked_input_ids: Optional[torch.LongTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], CLIPOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = self.visual_projection(vision_outputs[1])
        text_embeds = self.text_projection(text_outputs[1])
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        logit_scale = self.logit_scale.exp()
        logits_per_text, logits_per_image, contrastive_labels = (
            _distributed_contrastive_logits(
                text_embeds,
                image_embeds,
                logit_scale,
            )
        )

        loss = None
        if return_loss:
            caption_loss = nn.functional.cross_entropy(
                logits_per_text,
                contrastive_labels,
            )
            image_loss = nn.functional.cross_entropy(
                logits_per_image,
                contrastive_labels,
            )
            loss = (caption_loss + image_loss) / 2.0
            loss = self._add_text_reconstruction_loss(
                loss,
                masked_input_ids=masked_input_ids,
                mlm_labels=mlm_labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vision_outputs=vision_outputs,
            )
            loss = self._add_image_reconstruction_loss(
                loss,
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
                text_outputs=text_outputs,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                vision_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def _add_text_reconstruction_loss(
        self,
        loss: torch.Tensor,
        *,
        masked_input_ids,
        mlm_labels,
        attention_mask,
        token_type_ids,
        position_ids,
        output_attentions,
        output_hidden_states,
        return_dict,
        vision_outputs,
    ) -> torch.Tensor:
        if (
            (not self.text_reconstruction and not self.cross_reconstruction)
            or mlm_labels is None
            or masked_input_ids is None
        ):
            return loss

        mlm_outputs = self.text_model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        mlm_embeds = self.text_projection(mlm_outputs[0])
        if self.text_reconstruction:
            prediction_scores = self.mlm_head(mlm_embeds)
            mlm_loss = nn.functional.cross_entropy(
                prediction_scores.view(-1, self.config.text_config.vocab_size),
                mlm_labels.view(-1),
            )
            loss = loss + self.lambda_intra * mlm_loss

        if not self.cross_reconstruction:
            return loss

        vision_features = self.visual_projection(vision_outputs[0])
        cross_embeds = self.text_cross_block(mlm_embeds, vision_features, vision_features)
        cross_scores = self.mlm_head(cross_embeds)
        cross_loss = nn.functional.cross_entropy(
            cross_scores.view(-1, self.config.text_config.vocab_size),
            mlm_labels.view(-1),
        )
        return loss + self.lambda_cross * cross_loss

    def _add_image_reconstruction_loss(
        self,
        loss: torch.Tensor,
        *,
        pixel_values,
        bool_masked_pos,
        text_outputs,
        attention_mask,
        return_dict,
    ) -> torch.Tensor:
        if (
            (not self.image_reconstruction and not self.cross_reconstruction)
            or bool_masked_pos is None
            or pixel_values is None
        ):
            return loss

        mim_outputs = self.vision_model(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            return_dict=return_dict,
        )
        sequence_output = self.visual_projection(mim_outputs[0])
        if self.image_reconstruction:
            intra_loss = self._pixel_reconstruction_loss(
                sequence_output, pixel_values, bool_masked_pos
            )
            loss = loss + self.lambda_intra * intra_loss

        if not self.cross_reconstruction:
            return loss

        text_features = self.text_projection(text_outputs[0])
        key_padding_mask = (
            attention_mask.eq(0) if attention_mask is not None else None
        )
        cross_output = self.visual_cross_block(
            sequence_output,
            text_features,
            text_features,
            key_padding_mask=key_padding_mask,
        )
        cross_loss = self._pixel_reconstruction_loss(
            cross_output, pixel_values, bool_masked_pos
        )
        return loss + self.lambda_cross * cross_loss

    def _pixel_reconstruction_loss(
        self,
        sequence_output: torch.Tensor,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor,
    ) -> torch.Tensor:
        patch_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = patch_output.shape
        height = width = math.floor(sequence_length**0.5)
        patch_output = patch_output.permute(0, 2, 1).reshape(
            batch_size, num_channels, height, width
        )
        reconstructed = self.vit_decoder(patch_output)

        grid = self.config.vision_config.image_size // self.config.vision_config.patch_size
        mask = bool_masked_pos.reshape(-1, grid, grid)
        mask = (
            mask.repeat_interleave(self.config.vision_config.patch_size, 1)
            .repeat_interleave(self.config.vision_config.patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )
        reconstruction_loss = nn.functional.mse_loss(
            reconstructed, pixel_values, reduction="none"
        )
        return (
            (reconstruction_loss * mask).sum()
            / (mask.sum() + 1e-5)
            / self.config.vision_config.num_channels
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str | None = None,
        text_model_name_or_path: str | None = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        shared_loader_kwargs = {}
        if "model_revision" in kwargs:
            shared_loader_kwargs["revision"] = kwargs.pop("model_revision")
        for key in (
            "cache_dir",
            "revision",
            "token",
            "trust_remote_code",
            "local_files_only",
        ):
            if key in kwargs:
                shared_loader_kwargs[key] = kwargs.pop(key)

        kwargs_vision = {
            argument[len("vision_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vision_")
        }
        kwargs_text = {
            argument[len("text_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_")
        }
        for key in list(kwargs_vision):
            del kwargs["vision_" + key]
        for key in list(kwargs_text):
            del kwargs["text_" + key]
        for key, value in shared_loader_kwargs.items():
            kwargs_vision.setdefault(key, value)
            kwargs_text.setdefault(key, value)

        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError("vision_model_name_or_path is required.")
            vision_config = kwargs_vision.get("config")
            if vision_config is None:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)
            if vision_config.model_type == "clip":
                kwargs_vision["config"] = vision_config.vision_config
                vision_model = CLIPVisionModel.from_pretrained(
                    vision_model_name_or_path, *model_args, **kwargs_vision
                )
            else:
                kwargs_vision["config"] = vision_config
                vision_model = AutoModel.from_pretrained(
                    vision_model_name_or_path, *model_args, **kwargs_vision
                )

        text_model = kwargs_text.pop("model", None)
        mlm_head = None
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError("text_model_name_or_path is required.")
            text_config = kwargs_text.get("config")
            if text_config is None:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config
            text_model = AutoModel.from_pretrained(
                text_model_name_or_path, *model_args, **kwargs_text
            )
            mlm_model = AutoModelForMaskedLM.from_pretrained(
                text_model_name_or_path, *model_args, **kwargs_text
            )
            mlm_head = getattr(mlm_model, "cls", None)

        config = VisionTextDualEncoderConfig.from_vision_text_configs(
            vision_model.config, text_model.config, **kwargs
        )
        model = cls(
            config=config,
            vision_model=vision_model,
            text_model=text_model,
            mlm_head=mlm_head,
        )
        logger.warning(
            "Projection layers and logit_scale are newly initialized; train the "
            "retriever before using it for retrieval."
        )
        return model

    def save_pretrained(self, save_directory, safe_serialization=False, **kwargs):
        kwargs["safe_serialization"] = False
        if safe_serialization:
            logger.warning("Disabled safe_serialization for shared MLM-head tensors.")
        return super().save_pretrained(save_directory, **kwargs)


def _distributed_contrastive_logits(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build local-anchor/global-candidate logits with autograd-safe gathering."""
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        labels = torch.arange(text_embeds.shape[0], device=text_embeds.device)
        return logits_per_text, logits_per_text.t(), labels

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_size = torch.tensor(
        [text_embeds.shape[0]],
        device=text_embeds.device,
        dtype=torch.long,
    )
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    if len(set(sizes)) != 1:
        raise ValueError(
            "Distributed contrastive training requires equal local batch sizes; "
            f"got {sizes}."
        )

    from torch.distributed.nn.functional import all_gather

    all_text = torch.cat(all_gather(text_embeds), dim=0)
    all_images = torch.cat(all_gather(image_embeds), dim=0)
    labels = (
        torch.arange(text_embeds.shape[0], device=text_embeds.device)
        + rank * text_embeds.shape[0]
    )
    logits_per_text = torch.matmul(text_embeds, all_images.t()) * logit_scale
    logits_per_image = torch.matmul(image_embeds, all_text.t()) * logit_scale
    return logits_per_text, logits_per_image, labels
