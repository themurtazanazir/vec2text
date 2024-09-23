from types import SimpleNamespace
from typing import List

import torch
from torch import nn
from vec2text.lms.gpt2 import GPT2WithHidden
from transformers import AutoTokenizer


class GPT2Embedder(nn.Module):  # converting to module so device stuff is handled
    def __init__(self, max_length: int, max_new_tokens: int):
        super(GPT2Embedder, self).__init__()

        self.model = GPT2WithHidden.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        self.config = SimpleNamespace(hidden_size=self.model.config.n_embd)

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
        emb_input_ids['input_ids'] = torch.cat(
            (
                emb_input_ids.input_ids,
                torch.zeros(
                    B,
                    self.max_new_tokens + 1, # i dont know why I am adding 1, but should not do any harm, right? right??
                    dtype=emb_input_ids.input_ids.dtype,
                    device=emb_input_ids.input_ids.device,
                ),
            ),
            dim=1,
        )
        emb_input_ids['attention_mask'] = torch.cat(
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
                next_token = i[torch.arange(B), emb_input_ids.attention_mask.sum(-1) - 1]
                emb_input_ids.input_ids[torch.arange(B), emb_input_ids.attention_mask.sum(-1)] = (
                    next_token
                )
                output_states.append(
                    hidden_state[torch.arange(B), emb_input_ids.attention_mask.sum(-1) - 1, :].unsqueeze(1)
                )
                emb_input_ids.attention_mask[
                    torch.arange(B), emb_input_ids.attention_mask.sum(-1)
                ] = 1
        return torch.cat(output_states, dim=1)


    def __call__(self, *args, **kwargs):
        return self.get_hidden_states(*args, **kwargs)