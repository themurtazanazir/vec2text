import os
os.environ["TMPDIR"]="/workspace/mnazir/vec2text/temp/"
os.environ["HF_HUB_CACHE"]="/workspace/mnazir/vec2text/huggingface/hub/"
os.environ["HF_HOME"]="/workspace/mnazir/vec2text/huggingface/"
os.environ["VEC2TEXT_CACHE"]="/workspace/mnazir/vec2text/vec2text/"
os.environ["WANDB_DIR"]="/workspace/mnazir/vec2text/"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments
import copy

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cmd = "--per_device_train_batch_size 250 --per_device_eval_batch_size 250 --max_seq_length 64 --num_train_epochs 100 --max_eval_samples 1000 --eval_steps 25000 --warmup_steps 25000 --learning_rate 0.0002 --dataset_name one_million_instructions --model_name_or_path t5-base --use_wandb=0 --experiment inversion_from_hidden_states --bf16=1 --embedder_torch_dtype bfloat16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --embedder_model_name llama2_chat-random_k-alr --max_new_tokens 16 --output_dir /workspace/llama2_chat-random_k-alr-16-toks-bugfix-4-nodes/ --exp_group_name llama2-chat --extra_tokens 100"


parser = transformers.HfArgumentParser(
(ModelArguments, DataArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses(cmd.split())
experiment = experiment_from_args(model_args, data_args, training_args)

model = experiment.load_model()


def format(system_message, instruction, chat_format):
    if chat_format:
        return f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
    else:
        return system_message + "\n\n" + instruction


ckpt = experiment._get_checkpoint()
print("CKPT", ckpt)
trainer = experiment.trainer_cls(model=model)
trainer._load_from_checkpoint(ckpt)
trainer.model.eval()

def invert(sys, ins, chat_format):
    strings = [format(sys, ins, chat_format).strip()]
    print(f"{strings=}", flush=True)
    t = trainer.embedder_tokenizer
    print(f"{t.padding_side=}")
    inputs = t(strings, return_tensors='pt', padding='max_length',
            # max_length=trainer.model.embedder.max_length, 
            max_length=64,
            truncation=True)
    inputs = {f"embedder_{k}": v for k,v in inputs.items()}
    gen_kwargs = copy.copy(trainer.gen_kwargs)
    max_length = trainer.model.config.max_seq_length
    gen_kwargs["max_length"] = max_length
    outputs = trainer.generate(inputs, generation_kwargs={'max_new_tokens': 64})
    output_strings = trainer.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_strings[0]



def get_overlap_toks(llama2_tokenizer, other_tokenizer, llama2_chosen_toks):
    llama_vocab = llama2_tokenizer.get_vocab()
    other_vocab = other_tokenizer.get_vocab()

    overlap = set(llama_vocab).intersection(set(other_vocab)) # in string formats
    chosen_strings = {k for k,v in llama_vocab.items() if v in llama2_chosen_toks}

    overlap_with_chosen = chosen_strings.intersection(overlap)

    remaining = list(overlap - overlap_with_chosen)
    import random
    random.seed(4673)
    random.shuffle(remaining)
    
    total_overlap_chosen = list(overlap_with_chosen) + remaining[:4200-len(overlap_with_chosen)]

    llama_overlap_toks = [llama_vocab[k] for k in total_overlap_chosen]
    other_overlap_toks = [other_vocab[k] for k in total_overlap_chosen]

    return llama_overlap_toks, other_overlap_toks


def get_logprobs(model, tokenizer, strings):
    inputs = tokenizer(strings, return_tensors='pt', padding='max_length',
            # max_length=trainer.model.embedder.max_length, 
            max_length=64,
            truncation=True)
    embedder_input_ids = inputs.input_ids
    embedder_attention_mask = inputs.attention_mask
    device = next(model.parameters()).device
    embedder_input_ids = embedder_input_ids.to(device)
    embedder_attention_mask = embedder_attention_mask.to(device)
    output = model.generate(
        input_ids=embedder_input_ids,
        attention_mask=embedder_attention_mask,
        max_new_tokens=16,
        do_sample=False,
        temperature=1,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True
    )

    ##!!  this part is usually in lms and not in embedder.
    logits = torch.cat([i.unsqueeze(1) for i in output.scores], dim=1)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    return logprobs


other_llm_name = "Qwen/Qwen2.5-7B-Instruct"
other_llm = AutoModelForCausalLM.from_pretrained(other_llm_name, torch_dtype=torch.bfloat16)
other_llm.eval()
other_llm.to(device)
other_tokenizer = AutoTokenizer.from_pretrained(other_llm_name)
other_tokenizer.padding_side = "left"

## invert from llama once to set its chosen_tokens
output = invert("", "reverse", True)
print(output)
llama_overlap_toks, other_overlap_toks = get_overlap_toks(trainer.embedder_tokenizer, other_tokenizer, model.embedder.chosen_tokens)
prompt = "reverse the string"
messages = [{"role":"system", "content":""},{"role":"user", "content":prompt}]
text = other_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
other_logprobs = get_logprobs(other_llm, other_tokenizer, [text])
other_logprobs = other_logprobs[0]
import numpy as np

llama_unembed = model.embedder.model.lm_head.weight.data
llama_unembed = llama_unembed
llama_unembed_alr = (llama_unembed-llama_unembed[[llama_overlap_toks[0]]]).cpu().float().numpy()
other_logprobs_alr = (other_logprobs - other_logprobs[:, [other_overlap_toks[0]]]).cpu().float().numpy()

llama2_hidden_state, *_ = np.linalg.lstsq(
        llama_unembed_alr[llama_overlap_toks[1:]],
        other_logprobs_alr.T[other_overlap_toks[1:]]
        )

llama2_logits = torch.from_numpy(llama_unembed_alr @ llama2_hidden_state).to(device).T.unsqueeze(0)
llama2_logprobs = torch.nn.functional.log_softmax(llama2_logits, dim=-1)
llama2_logprobs = llama2_logprobs[:, :, model.embedder.chosen_tokens]
alr = llama2_logprobs[:, :, 1:] - llama2_logprobs[:, :, 0:1]  
embeddings = model.embedding_transform(alr)
attention_mask = torch.ones(
        (embeddings.shape[0], embeddings.shape[1]),
        device=embeddings.device
)
output = model.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                max_new_tokens=64,
                #**generation_kwargs,
            )
