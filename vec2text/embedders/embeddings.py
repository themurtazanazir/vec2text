from types import SimpleNamespace
from typing import List, Literal
import warnings

import torch
from torch import nn
import transformers
from vec2text.lms.gpt2 import GPT2WithHidden, GPT2RandomCLRTransform
from vec2text.lms.llama import LlamaRandomCLRTransform
from transformers import AutoTokenizer


class Embedder(nn.Module):
    def __init__(self, max_length: int, max_new_tokens: int):
        super(Embedder, self).__init__()

        self.model, self.tokenizer = self.load_model_and_tokenizer()
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def train(self, mode):
        warnings.warn("Tried to set a mode. This model is permanently set in eval mode")
        return super().train(mode=False)

    def get_hidden_states(self, input_strings: List[str]):
        emb_input_ids = self.tokenizer(
            input_strings,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        output_states = []

        B, T = emb_input_ids.input_ids.shape
        emb_input_ids["input_ids"] = torch.cat(
            (
                emb_input_ids.input_ids,
                torch.zeros(
                    B,
                    self.max_new_tokens
                    + 1,  # i dont know why I am adding 1, but should not do any harm, right? right??
                    dtype=emb_input_ids.input_ids.dtype,
                    device=emb_input_ids.input_ids.device,
                ),
            ),
            dim=1,
        )
        emb_input_ids["attention_mask"] = torch.cat(
            (
                emb_input_ids.attention_mask,
                torch.zeros(
                    B,
                    self.max_new_tokens + 1,
                    dtype=emb_input_ids.attention_mask.dtype,
                    device=emb_input_ids.attention_mask.device,
                ),
            ),
            dim=1,
        )

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                model_output, hidden_state = self.model(**emb_input_ids)

                logits = model_output.logits

                p, i = torch.max(logits, dim=-1)
                next_token = i[
                    torch.arange(B), emb_input_ids.attention_mask.sum(-1) - 1
                ]
                emb_input_ids.input_ids[
                    torch.arange(B), emb_input_ids.attention_mask.sum(-1)
                ] = next_token
                output_states.append(
                    hidden_state[
                        torch.arange(B), emb_input_ids.attention_mask.sum(-1) - 1, :
                    ].unsqueeze(1)
                )
                emb_input_ids.attention_mask[
                    torch.arange(B), emb_input_ids.attention_mask.sum(-1)
                ] = 1
        return torch.cat(output_states, dim=1)

    def __call__(self, *args, **kwargs):
        return self.get_hidden_states(*args, **kwargs)


class GPT2Embedder(Embedder):  # converting to module so device stuff is handled
    def __init__(self, max_length: int, max_new_tokens: int):
        super(GPT2Embedder, self).__init__(
            max_length=max_length, max_new_tokens=max_new_tokens
        )
        self.config = SimpleNamespace(hidden_size=self.model.config.n_embd)

    def load_model_and_tokenizer(self):
        model = GPT2WithHidden.from_pretrained("gpt2")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


class GPT2RandomTransformEmbedder(GPT2Embedder):

    def load_model_and_tokenizer(self):
        model = GPT2RandomCLRTransform.from_pretrained("gpt2")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


class Llama2_7BRandomTransformEmbedder(Embedder):
    def __init__(
        self,
        max_length: int,
        max_new_tokens: int,
        torch_dtype: Literal["float32", "float16", "bfloat16"],
    ):
        self.torch_dtype = torch_dtype
        super(Llama2_7BRandomTransformEmbedder, self).__init__(
            max_length=max_length, max_new_tokens=max_new_tokens
        )
        self.config = SimpleNamespace(hidden_size=self.model.config.hidden_size)

    def load_model_and_tokenizer(self):

        if self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16

        model = LlamaRandomCLRTransform.from_pretrained(
            "meta-llama/Llama-2-7b-hf", torch_dtype=self.torch_dtype
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
