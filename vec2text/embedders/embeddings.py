from abc import ABC
from types import SimpleNamespace
from typing import List, Literal
import warnings

import numpy as np
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

    def get_hidden_states(self, embedder_input_ids, embedder_attention_mask):
        device = next(self.model.parameters()).device
        embedder_input_ids = embedder_input_ids.to(device)
        embedder_attention_mask = embedder_attention_mask.to(device)

        output_states = []

        B, T = embedder_input_ids.shape
        embedder_input_ids = torch.cat(
            (
                embedder_input_ids,
                torch.zeros(
                    B,
                    self.max_new_tokens
                    + 1,  # i dont know why I am adding 1, but should not do any harm, right? right??
                    dtype=embedder_input_ids.dtype,
                    device=embedder_input_ids.device,
                ),
            ),
            dim=1,
        )
        embedder_attention_mask = torch.cat(
            (
                embedder_attention_mask,
                torch.zeros(
                    B,
                    self.max_new_tokens + 1,
                    dtype=embedder_attention_mask.dtype,
                    device=embedder_attention_mask.device,
                ),
            ),
            dim=1,
        )

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                model_output, hidden_state = self.model(
                        input_ids=embedder_input_ids,
                        attention_mask=embedder_attention_mask,
                        )

                logits = model_output.logits

                p, i = torch.max(logits, dim=-1)
                next_token = i[
                    torch.arange(B), embedder_attention_mask.sum(-1) - 1
                ]
                embedder_input_ids[
                    torch.arange(B), embedder_attention_mask.sum(-1)
                ] = next_token
                output_states.append(
                    hidden_state[
                        torch.arange(B), embedder_attention_mask.sum(-1) - 1, :
                    ].unsqueeze(1)
                )
                embedder_attention_mask[
                    torch.arange(B), embedder_attention_mask.sum(-1)
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


class TransformedHiddenStateEmbedder(Embedder, ABC):

    def extract_hidden_state_from_logprobs(self, logprobs):
        raise NotImplementedError

    def get_hidden_states(self, embedder_input_ids, embedder_attention_mask):
        logprobs = self.get_logprobs(embedder_input_ids=embedder_input_ids,
                                     embedder_attention_mask=embedder_attention_mask)
        return self.extract_hidden_state_from_logprobs(logprobs)

    def get_logprobs(self, embedder_input_ids, embedder_attention_mask):
        device = next(self.model.parameters()).device 
        embedder_input_ids = embedder_input_ids.to(device)
        embedder_attention_mask = embedder_attention_mask.to(device)
        output = self.model.generate(
            input_ids=embedder_input_ids,
            attention_mask=embedder_attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True
        )

        ##!!  this part is usually in lms and not in embedder.
        logits = torch.cat([i.unsqueeze(1) for i in output.scores], dim=1)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        return logprobs


class RandomTransformCLREmbedder(TransformedHiddenStateEmbedder, ABC):

    def extract_hidden_state_from_logprobs(self, logprobs):
        clr = logprobs - torch.mean(logprobs, dim=-1, keepdims=True)  # B, T, V
        if not hasattr(self, "transform"):
            g = torch.Generator()
            g.manual_seed(666)
            self.transform = torch.randn(
                self.model.config.vocab_size,
                self.config.hidden_size,
                generator=g,
            ).to(logprobs.device)
        hidden_states = clr @ self.transform  # B, T, D

        return hidden_states


class RandomTransformALREmbedder(TransformedHiddenStateEmbedder, ABC):

    def extract_hidden_state_from_logprobs(self, logprobs):
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        if not hasattr(self, "transform"):
            g = torch.Generator()
            g.manual_seed(666)
            self.transform = torch.randn(
                self.model.config.vocab_size - 1,
                self.config.hidden_size,
                generator=g,
            ).to(logprobs.device)
        hidden_states = alr @ self.transform  # B, T, D

        return hidden_states


class GPT2RandomTransformCLREmbedder(RandomTransformCLREmbedder):

    def __init__(
        self,
        max_length: int,
        max_new_tokens: int,
    ):
        super(GPT2RandomTransformCLREmbedder, self).__init__(
            max_length=max_length, max_new_tokens=max_new_tokens
        )
        self.config = SimpleNamespace(hidden_size=self.model.config.n_embd)

    def load_model_and_tokenizer(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2",
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer


class GPT2RandomTransformALREmbedder(RandomTransformALREmbedder):

    def __init__(
        self,
        max_length: int,
        max_new_tokens: int,
    ):
        super(GPT2RandomTransformALREmbedder, self).__init__(
            max_length=max_length, max_new_tokens=max_new_tokens
        )
        self.config = SimpleNamespace(hidden_size=self.model.config.n_embd)

    def load_model_and_tokenizer(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2",
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer


class GPT2KTokensEmbedder(TransformedHiddenStateEmbedder, ABC):
    def __init__(self, max_length: int, max_new_tokens: int, extra_tokens: int):
        super(GPT2KTokensEmbedder, self).__init__(
            max_length=max_length, max_new_tokens=max_new_tokens
        )
        assert extra_tokens >= 0
        self.extra_tokens = extra_tokens

        self.config = SimpleNamespace(
            hidden_size=self.model.config.n_embd + extra_tokens
        )

    def load_model_and_tokenizer(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2",
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer


class Llama2KTokensEmbedder(TransformedHiddenStateEmbedder, ABC):
    def __init__(
        self,
        max_length: int,
        max_new_tokens: int,
        extra_tokens: int,
        torch_dtype: Literal["float32", "float16", "bfloat16"],
    ):
        self.torch_dtype = torch_dtype
        super(Llama2KTokensEmbedder, self).__init__(
            max_length=max_length, max_new_tokens=max_new_tokens
        )
        assert extra_tokens >= 0
        self.extra_tokens = extra_tokens

        self.config = SimpleNamespace(
            hidden_size=self.model.config.hidden_size + extra_tokens
        )

    def load_model_and_tokenizer(self):

        if self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16

        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=self.torch_dtype,
            # quantization_config=bnb_config,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer


class GPT2RandomKALREmbedder(GPT2KTokensEmbedder):

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[
                : self.config.hidden_size + 1
            ]  # alr will remove one

        logprobs = logprobs[:, :, self.chosen_tokens]
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        return alr


class Llama2RandomKALREmbedder(Llama2KTokensEmbedder):

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[
                : self.config.hidden_size + 1
            ]  # alr will remove one

        logprobs = logprobs[:, :, self.chosen_tokens]
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        return alr


class Llama2ChatRandomKALREmbedder(Llama2KTokensEmbedder):

    def load_model_and_tokenizer(self):

        if self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16

        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=self.torch_dtype,
            # quantization_config=bnb_config,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[
                : self.config.hidden_size + 1
            ]  # alr will remove one

        logprobs = logprobs[:, :, self.chosen_tokens]
        alr = logprobs[:, :, 1:] - logprobs[:, :, 0:1]  # B, T, V
        return alr


class GPT2RandomKCLREmbedder(GPT2KTokensEmbedder):

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[: self.config.hidden_size]

        clr = logprobs - torch.mean(logprobs, dim=-1, keepdims=True)  # B, T, V
        clr = clr[:, :, self.chosen_tokens]
        return clr


class Llama2RandomKCLREmbedder(Llama2KTokensEmbedder):

    def extract_hidden_state_from_logprobs(self, logprobs):

        if not hasattr(self, "chosen_tokens"):
            g = torch.Generator()
            g.manual_seed(666)
            self.chosen_tokens = torch.randperm(
                self.model.config.vocab_size,
                generator=g,
            )[: self.config.hidden_size]

        clr = logprobs - torch.mean(logprobs, dim=-1, keepdims=True)  # B, T, V
        clr = clr[:, :, self.chosen_tokens]
        return clr


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
        self.config = SimpleNamespace(hidden_size=self.model.config.hidden_size + 100)

    def load_model_and_tokenizer(self):

        if self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16

        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=self.torch_dtype,
            # quantization_config=bnb_config,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    def get_hidden_states(self, input_strings: List[str]):
        emb_input_ids = self.tokenizer(
            input_strings,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        output = self.model.generate(
            **emb_input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True
        )

        ##!!  this part is usually in lms and not in embedder.
        logits = torch.cat([i.unsqueeze(1) for i in output.scores], dim=1)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        clr = logprobs - torch.mean(logprobs, dim=-1, keepdims=True)  # B, T, V
        hidden_states = clr[:, :, : (self.config.hidden_size)]  # adding 100 to be safe
        return hidden_states
