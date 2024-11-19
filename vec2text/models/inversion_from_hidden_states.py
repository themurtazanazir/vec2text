import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel

from vec2text.models.model_utils import load_embedder_and_tokenizer


class InversionFromHiddenStatesModel(InversionModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        # hacky way of checking if model is a pre-trained HF decoder
        # assert ("CausalLM" in str(type(self.embedder))) or (
        #     "LMHead" in str(type(self.embedder))
        # )
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embedder_is_decoder = True
        bottleneck_dim = self.bottleneck_dim


        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )

        self._emb_top_p = None
        self._emb_top_k = None
        self._emb_temp = None
        self._softmax_in_log_space = True

    def load_embedder_and_tokenizer(self, config):
        return load_embedder_and_tokenizer(
            name=config.embedder_model_name,
            torch_dtype=config.embedder_torch_dtype,
            use_hidden_states=True,
            max_length=config.max_seq_length,
            max_new_tokens=config.max_new_tokens,
            extra_tokens=config.extra_tokens,
        )

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedder = self.embedder

        inputs_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        model_output = embedder(inputs_str)
        return model_output
    
    def embed_and_project(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 3  # batch by d
        elif self.embedder_no_grad:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )


        # if self.training:
        #     # Update unigram.
        #     unigram_batch = embeddings.mean(dim=0, keepdim=True)
        #     self.unigram.data = self.unigram.data * (
        #         1 - self.unigram_beta
        #     ) + unigram_batch * (self.unigram_beta)
        # embeddings -= self.unigram

        # if self._zero_except_topk is not None:
        #     embeddings = zero_embedding_except_topk(
        #         embeddings,
        #         vocab_size=self.embedder.config.vocab_size,
        #         k=self._zero_except_topk,
        #         default_val=-30.0,
        #     )

        # embeddings = embeddings.to(self.sequence_weights.dtype)
        # embeddings = embeddings.reshape(
        #     (embeddings.shape[0], self.num_repeat_tokens, self.encoder_hidden_dim)
        # )
        # embeddings = torch.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)
        # embeddings = embeddings.to(next(self.embedding_transform.parameters()).dtype)
        embeddings = self.embedding_transform(embeddings)
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )

        assert embeddings.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.encoder_hidden_dim,
        )
        return embeddings, attention_mask

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return outputs

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask

        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )


class InversionFromRandomTransformedHiddenStatesModel(InversionFromHiddenStatesModel):


    def load_embedder_and_tokenizer(self, config):
        return load_embedder_and_tokenizer(
            name=f"{config.embedder_model_name}-random-transformed",
            torch_dtype=config.embedder_torch_dtype,
            use_hidden_states=True,
            max_length=config.max_seq_length,
            max_new_tokens=config.max_new_tokens,
        )