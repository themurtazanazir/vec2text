import copy
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel

from vec2text.models.model_utils import load_embedder_and_tokenizer
from transformers.modeling_outputs import BaseModelOutput


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


class TokenEncoder(nn.Module):
    def __init__(self, hidden_dim, max_bytes, num_heads, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_bytes = max_bytes
        self.byte_embedder = nn.Embedding(256, hidden_dim)

        self.pos_embedder = nn.Embedding(max_bytes, hidden_dim)

        self.attention_layers = nn.ModuleList(
            [
                AttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=2 * hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
        )

    def _process_chunk(self, chunk_bytes):
        """Process a chunk of bytes and logprobs to produce scalar scores."""
        # Embed bytes
        chunk_byte_embeddings = self.byte_embedder(
            chunk_bytes
        )  # B, gens, max_steps, topk, max_bytes, dim
        *shapes, chunk_max_bytes, dim = chunk_byte_embeddings.shape

        # Reshape for processing
        chunk_byte_data = chunk_byte_embeddings.reshape(
            -1, chunk_max_bytes, dim
        )  # B', max_bytes, dim

        # Add positional embeddings
        pos = (
            torch.arange(chunk_byte_data.shape[1])
            .unsqueeze(0)
            .repeat((chunk_byte_data.shape[0], 1))
            .to(chunk_byte_data.device)
        )
        pos_emb = self.pos_embedder(pos)
        chunk_byte_data = chunk_byte_data + pos_emb

        # Process through attention layers
        for layer in self.attention_layers:
            chunk_byte_data = layer(chunk_byte_data)

        # Final processing
        chunk_byte_data = self.final_norm(
            chunk_byte_data)  # B', max_bytes, dim
        chunk_token_encodings = chunk_byte_data.mean(dim=1)  # B', dim
        chunk_token_encodings = self.pooling(chunk_token_encodings)  # B', dim

        return chunk_token_encodings.reshape(
            *shapes,
            self.hidden_dim,
        )

    def forward(
        self,
        bytes_batch,  # B, gens, T, Topk, max_bytes
    ):

        *shapes, top_k, _ = bytes_batch.shape

        all_encodings = torch.zeros(
            (*shapes, top_k, self.hidden_dim),
            device=next(self.parameters()).device,
        )

        chunk_size = 868  # Adjust based on GPU memory
        for chunk_start in range(0, top_k, chunk_size):
            chunk_end = min(chunk_start + chunk_size, top_k)

            # Process this chunk and store results
            chunk_encodings = self._process_chunk(
                bytes_batch[..., chunk_start:chunk_end, :],
            )

            all_encodings[..., chunk_start:chunk_end, :] = chunk_encodings

        return all_encodings  # B, gens, max_steps, topk, dim


class TokensLogProbEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        max_bytes,
        num_heads,
        num_layers,
    ):
        super(TokensLogProbEncoder, self).__init__()
        self.token_encoder = TokenEncoder(
            hidden_dim=hidden_dim,
            max_bytes=max_bytes,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for the logprob
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        bytes_batch,  # B, T, Topk, max_bytes
        topk_logprobs,  # B, T, Topk
    ):

        B, max_steps, top_k = bytes_batch.shape[:3]
        token_encodings = self.token_encoder(
            bytes_batch)  # B, T, Topk, self.hidden_dim
        # B, T, Topk, self.hidden_dim + 1
        chunk_combined = torch.cat(
            [token_encodings, topk_logprobs.unsqueeze(-1)], dim=-1
        )
        hidden_states = self.combiner(chunk_combined)  # B, T, Topk, 1

        return hidden_states.view(B, max_steps, top_k)  # remove last dim


class TokensLogProbChosenEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        max_bytes,
        num_heads,
        num_layers,
        n_chosen,
    ):
        super(TokensLogProbChosenEncoder, self).__init__()
        self.token_encoder = TokenEncoder(
            hidden_dim=hidden_dim,
            max_bytes=max_bytes,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for the logprob
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.chosen_transform = nn.Linear(hidden_dim, n_chosen)

    def forward(
        self,
        bytes_batch,  # B, T, Topk+1, max_bytes
        topk_logprobs,  # B, T, Topk
    ):
        # this bytes batch is different, the first element in dim=2 (0 indexed) is chosen token

        B, max_steps, top_k = bytes_batch.shape[:3]
        top_k = top_k - 1  # remove the first k as that is the chosen one
        token_encodings = self.token_encoder(
            bytes_batch
        )  # B, T, Topk+1, self.hidden_dim
        chosen_encodings = token_encodings[:, :, 0, :]  # B, T, self.hidden_dim
        # B, T, Topk, self.hidden_dim
        token_encodings = token_encodings[:, :, 1:, :]
        # B, T, Topk, self.hidden_dim + 1
        chunk_combined = torch.cat(
            [token_encodings, topk_logprobs.unsqueeze(-1)], dim=-1
        )
        hidden_states = self.combiner(chunk_combined)  # B, T, Topk, 1
        hidden_states = hidden_states.view(B, max_steps, top_k)  # B, T, Topk
        chosen_transformed = self.chosen_transform(
            chosen_encodings)  # B, T, Topk

        return hidden_states + chosen_transformed


class InversionFromToksProbs(InversionModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embedder_is_decoder = True
        bottleneck_dim = self.bottleneck_dim

        self.token_embedder = TokensLogProbEncoder(
            hidden_dim=64,
            max_bytes=20,
            num_heads=4,
            num_layers=2,
        )

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )
        self.register_buffer("token2bytes", self.create_byte_embedding())

    def create_byte_embedding(self, max_bytes=20):
        tokenizer = self.embedder_tokenizer
        vocab_size = tokenizer.vocab_size

        # Initialize embedding tensor
        byte_embedding = torch.zeros(
            (vocab_size, max_bytes), dtype=torch.int32)

        # Fill embedding with byte values for each token
        for token_id in range(vocab_size):
            try:
                token_str = tokenizer.decode([token_id])
                token_bytes = token_str.encode("utf-8")

                # Convert bytes to tensor and store in embedding
                byte_len = min(len(token_bytes), max_bytes)
                byte_values = torch.tensor(
                    [int(b) for b in token_bytes[:byte_len]], dtype=torch.int32
                )

                byte_embedding[token_id, :byte_len] = byte_values
            except Exception as e:
                print(f"Error processing token ID {token_id}: {e}")

        return byte_embedding

    def load_embedder_and_tokenizer(self, config):
        return load_embedder_and_tokenizer(
            name=config.embedder_model_name,
            torch_dtype=config.embedder_torch_dtype,
            use_toks_probs=True,
            max_length=config.max_seq_length,
            max_new_tokens=config.max_new_tokens,
            extra_tokens=config.extra_tokens,
            hidden_size=config.hidden_size,
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
        frozen_topk_ids: Optional[torch.Tensor] = None,
        frozen_topk_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frozen_topk_ids is not None and frozen_topk_logprobs is not None:
            embedder_output = {
                "topk_ids": frozen_topk_ids,
                "topk_logprobs": frozen_topk_logprobs,
            }
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

        topk_ids = embedder_output["topk_ids"]  # B, T, topk
        shape = topk_ids.shape
        flattened_ids = topk_ids.view(-1)
        byte_ids = self.token2bytes[flattened_ids]
        byte_ids = byte_ids.view(*shape, -1)

        embeddings = self.token_embedder(
            bytes_batch=byte_ids,
            topk_logprobs=embedder_output["topk_logprobs"],
        )
        embeddings = self.embedding_transform(embeddings)
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]
             ), device=embeddings.device
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
            frozen_topk_ids=inputs.get("frozen_topk_ids"),
            frozen_topk_logprobs=inputs.get("frozen_topk_logprobs"),
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
        frozen_topk_ids: Optional[torch.Tensor] = None,
        frozen_topk_logprobs: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask

        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_topk_ids=frozen_topk_ids,
            frozen_topk_logprobs=frozen_topk_logprobs,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )


class InversionFromToksProbsChosen(InversionFromToksProbs):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embedder_is_decoder = True
        bottleneck_dim = self.bottleneck_dim

        self.token_embedder = TokensLogProbChosenEncoder(
            hidden_dim=64,
            max_bytes=20,
            num_heads=4,
            num_layers=2,
            n_chosen=self.embedder_dim,
        )

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )
        self.register_buffer("token2bytes", self.create_byte_embedding())

    def load_embedder_and_tokenizer(self, config):
        return load_embedder_and_tokenizer(
            name=config.embedder_model_name,
            torch_dtype=config.embedder_torch_dtype,
            use_toks_probs=True,
            add_chosen_ids=True,
            max_length=config.max_seq_length,
            max_new_tokens=config.max_new_tokens,
            extra_tokens=config.extra_tokens,
            hidden_size=config.hidden_size,
            num_gens=config.num_gens,
        )

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_topk_ids: Optional[torch.Tensor] = None,
        frozen_topk_logprobs: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask

        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_topk_ids=frozen_topk_ids,
            frozen_topk_logprobs=frozen_topk_logprobs,
        )

        B, num_gens, max_new_tokens, dim = inputs_embeds.shape
        # send each generations separately to encoder
        inputs_embeds = inputs_embeds.view(B*num_gens, max_new_tokens, dim)

        encoder_outputs = self.encoder_decoder.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        # B, num_gens, max_new_tokens, dim
        last_hidden_states = encoder_outputs[0]
        last_hidden_states = last_hidden_states.reshape(
            B, num_gens, max_new_tokens, -1).view(B, num_gens*max_new_tokens, -1)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=last_hidden_states, hidden_states=None, attentions=None,)

        return self.encoder_decoder(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )
