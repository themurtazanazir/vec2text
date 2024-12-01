from transformers.models.llama.modeling_llama import LlamaForCausalLM, Cache, CausalLMOutputWithPast
import torch
from typing import Tuple, Union, List


##!! this model is not being used anywhere rn
class LlamaRandomCLRTransform(LlamaForCausalLM):

    def forward(
        self,
        *args, **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output = super().forward(
            *args, **kwargs
        )
        logprobs = torch.nn.functional.log_softmax(output.logits, dim=-1)  # B, T, V
        clr = logprobs -  torch.mean(logprobs, dim=-1, keepdims=True) # B, T, V
        if not hasattr(self, "transform"):
            device = next(self.parameters()).device
            g = torch.Generator(device=device)
            g.manual_seed(666)
            self.transform = torch.randn(
                self.config.vocab_size, self.config.hidden_size, generator=g, device=device
            )
        hidden_state = clr @ self.transform  # B, T, D
        return output, hidden_state
  