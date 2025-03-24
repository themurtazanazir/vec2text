import copy
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel

from vec2text.models.model_utils import load_embedder_and_tokenizer


class AttentionBlock(nn.Module):
    """Single attention block with residual connections"""

    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x):
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        attn_output, _ = self.self_attention(normed_x, normed_x, normed_x)
        x = x + attn_output

        # Feed-forward with residual connection
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + ff_output

        return x


class TokensLogProbEncoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        hidden_dim,
        max_bytes,
        num_heads,
        num_layers,
    ):
        super(TokensLogProbEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.max_bytes = max_bytes
        self.byte_embedder = nn.Embedding(256, hidden_dim)

        self.pos_embedder = nn.Embedding(max_bytes, hidden_dim)

        self.attention_layers = nn.ModuleList(
            [
                AttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=4 * hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for the logprob
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.output_projection = nn.Linear(1, 1, bias=False)

    def forward(self, topk_toks, topk_logprobs):

        max_bytes_size = 0
        bytes_batch = []
        for sample in topk_toks:
            bytes_sample = []
            for timestep in sample:
                texts = self.tokenizer.batch_decode(timestep)
                bytes_timestep = [
                    list(i.encode("utf-8"))[: self.max_bytes] for i in texts
                ]
                tmp_max_size = max(len(i) for i in bytes_timestep)
                if max_bytes_size < tmp_max_size:
                    max_bytes_size = tmp_max_size
                bytes_sample.append(bytes_timestep)
            bytes_batch.append(bytes_sample)
        for sample_idx in range(len(bytes_batch)):
            for timestep_idx in range(len(bytes_batch[sample_idx])):
                for tok_idx in range(len(bytes_batch[sample_idx][timestep_idx])):
                    bytes_batch[sample_idx][timestep_idx][tok_idx] = bytes_batch[
                        sample_idx
                    ][timestep_idx][tok_idx] + [0] * (
                        max_bytes_size
                        - len(bytes_batch[sample_idx][timestep_idx][tok_idx])
                    )

        bytes_batch = torch.LongTensor(bytes_batch).to(
            device=next(self.parameters()).device
        )

        byte_embeddings = self.byte_embedder(bytes_batch)

        B, max_steps, top_k, max_bytes, _ = byte_embeddings.shape

        # Reshape for processing
        byte_data = byte_embeddings.view(B * max_steps * top_k, max_bytes, -1)
        logprobs = topk_logprobs.view(B * max_steps * top_k)

        pos = (
            torch.arange(byte_embeddings.shape[1])
            .unsqueeze(0)
            .repeat((byte_embeddings.shape[0], 1))
        ).to(next(self.parameters()).device)
        pos_emb = self.pos_embedder(pos)

        byte_data = byte_data + pos_emb

        # Process through attention layers
        for layer in self.attention_layers:
            byte_data = layer(byte_data)

        # Final layer norm
        byte_data = self.final_norm(byte_data)

        # Global average pooling
        # Shape: [B*max_steps*top_k, hidden_dim]
        token_encodings = byte_data.mean(dim=1)

        # Shape: [B*max_steps*top_k, hidden_dim]
        token_encodings = self.pooling(token_encodings)
        logprobs = logprobs.unsqueeze(-1)  # Shape: [B*max_steps*top_k, 1]
        # Shape: [B*max_steps*top_k, hidden_dim+1]
        combined = torch.cat([token_encodings, logprobs], dim=-1)

        scalars = self.combiner(combined)
        # Shape: [B*max_steps*top_k, 1]
        scalars = self.output_projection(scalars)

        scalars = scalars.view(B, max_steps, top_k)
        return scalars


class InversionFromToksProbs(InversionModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embedder_is_decoder = True
        bottleneck_dim = self.bottleneck_dim

        self.token_embedder = TokensLogProbEncoder(
            tokenizer=self.embedder.tokenizer,
            hidden_dim=128,
            max_bytes=20,
            num_heads=8,
            num_layers=3,
        )

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.token_embedder.hidden_dim, bottleneck_dim),
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
            use_toks_probs=True,
            max_length=config.max_seq_length,
            max_new_tokens=config.max_new_tokens,
            extra_tokens=config.extra_tokens,
        )

    def call_embedding_model(
        self,
        embedder_input_ids: Union[torch.Tensor, list],
        embedder_attention_mask: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        embedder = self.embedder

        model_output = embedder(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
        )
        return model_output

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frozen_embeddings is not None:
            embedder_output = frozen_embeddings
        elif self.embedder_no_grad:
            with torch.no_grad():
                embedder_output = self.call_embedding_model(
                    embedder_input_ids=embedder_input_ids,
                    embedder_attention_mask=embedder_attention_mask,
                )
        else:
            embedder_output = self.call_embedding_model(
                embedder_input_ids=embedder_input_ids,
                embedder_attention_mask=embedder_attention_mask,
            )

        topk_logprobs, topk_ids = embedder_output
        embeddings = self.token_embedder(
            topk_toks=topk_ids, topk_logprobs=topk_logprobs
        )
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
        # make a copy so we can edit
        generation_kwargs = copy.copy(generation_kwargs)
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
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
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask

        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )
