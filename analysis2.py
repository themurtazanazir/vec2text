from pprint import pprint
import json
import nltk
import torch
import tqdm
from typing import Dict, Tuple, List, Optional, Union
import copy
from vec2text.utils import dataset_map_multi_worker, get_num_proc
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments
from vec2text.experiments import experiment_from_args
from vec2text.data_helpers import load_standard_val_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os


os.environ["TMPDIR"] = "/home/mnazir/vec2text/data/test/temp/"
os.environ["HF_HOME"] = "/home/mnazir/vec2text/data/test/huggingface/"
os.environ["HF_HUB_CACHE"] = "/home/mnazir/vec2text/data/test/huggingface/hub/"
os.environ["VEC2TEXT_CACHE"] = "/home/mnazir/vec2text/data/test/vec2text/"
os.environ["WANDB_DIR"] = "/home/mnazir/vec2text/data/test/"

nltk.download("punkt_tab")

import torch
from vec2text.models.inversion_from_hidden_states import InversionFromHiddenStatesModel
from vec2text.models.config import InversionConfig
from transformers import AutoTokenizer

# Load the model from a local directory or from the Hub
model = InversionFromHiddenStatesModel.from_pretrained(
    "/home/mnazir/vec2text/data/test/experiments/llama2_chat-random_k-alr-16-toks-bugfix-4-nodes"  # or "username/model-name" from Hub
    # "/home/mnazir/vec2text/data/test/experiments/llama2-random_k-alr-16_toks/" 
    # "/home/mnazir/vec2text/data/test/experiments/llama2-random_k-alr/"
)

# Move to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def format(system_message, instruction, chat_format):
    if chat_format:
        return f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
    else:
        return system_message + "\n\n" + instruction


# def invert(sys, ins, chat_format, model):
#     strings = [format(sys, ins, chat_format).strip()]
#     t = model.embedder_tokenizer
#     print(f"{strings=}", flush=True)
#     print(f"{t.padding_side=}")
#     inputs = t(
#         strings,
#         return_tensors="pt",
#         padding="max_length",
#         # max_length=trainer.model.embedder.max_length,
#         max_length=64,
#         truncation=True,
#     )
#     inputs = {f"embedder_{k}": v for k, v in inputs.items()}
#     gen_kwargs = {}#copy.copy(trainer.gen_kwargs)
#     max_length = model.config.max_seq_length
#     gen_kwargs["max_length"] = max_length
#     outputs = model.generate(inputs, generation_kwargs=gen_kwargs)
#     output_strings = model.tokenizer.batch_decode(
#         outputs, skip_special_tokens=True)
#     return output_strings[0]


@torch.inference_mode()
def invert2(sys, ins, chat_format, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model
    model = model.to(device)
    target_llm = model.embedder.model
    target_tokenizer = model.embedder_tokenizer
    inverter_tokenizer = model.tokenizer
    llm_output_processor = model.embedding_transform
    inverter = model.encoder_decoder
    
    strings = [format(sys, ins, chat_format).strip()]
    label_strings = [format(sys, ins, chat_format=False).strip()]
    inverter_tokens = inverter_tokenizer(label_strings, return_tensors="pt")
    target_tokens = target_tokenizer(strings, return_tensors="pt")
    target_embedder_tokens = {f"embedder_{k}":v for k,v in target_tokens.items()}
    inverter_tokens = {k:v.to(device) for k,v in inverter_tokens.items()}
    target_tokens = {k:v.to(device) for k,v in target_tokens.items()}
    target_embedder_tokens = {k:v.to(device) for k,v in target_embedder_tokens.items()}

    target_output = target_llm.generate(**target_tokens, do_sample=False, top_p=None, max_new_tokens=16, return_dict_in_generate=True, output_logits=True)
    info = dict()
    info["tgt_in"] = inverter_tokenizer.batch_decode(inverter_tokens["input_ids"].reshape(-1, 1))
    info["tgt_out"] = []
    for logits in target_output.logits:
        topk = torch.topk(torch.softmax(logits, axis=-1), axis=-1, k=5)
        toks = target_tokenizer.batch_decode(topk.indices[0, ..., None], skip_special_tokens=False)
        probs = topk.values.tolist()
        print(toks)
        print(probs)
        info["tgt_out"].append(dict(probs=topk.values.tolist(), toks=toks))

    embeds, attn_mask = model.embed_and_project(**target_embedder_tokens)
    print(embeds.shape)
    info["perplexities"] = []
    info["guesses"] = []
    for gen_steps in range(embeds.shape[-2]):
        output = inverter(
                inputs_embeds=embeds[..., :gen_steps+1, :], 
                labels=inverter_tokens["input_ids"], 
                )
        print(f"{inverter_tokens['input_ids']=}")
        print(f"{torch.max(output.logits, axis=-1).indices=}")
        token_perplexities = torch.take_along_dim(-torch.log_softmax(output.logits, axis=-1), inverter_tokens["input_ids"][..., None], axis=-1)
        print(token_perplexities)
        info["perplexities"].append(token_perplexities.reshape(-1).tolist())
        guess = inverter_tokenizer.batch_decode(torch.max(output.logits, axis=-1)[1])[0]
        info["guesses"].append(guess)
        print(guess)
    return info

with open("data/viz.json", "w") as file:
    for prompt in [
            "Make a list of 10 ways to help students improve their study skills.",
            "Reverse this string: 'Make your bed and come downstairs for breakfast'",
            "Task: What are some of your favorite websites, and why do you visit them often?",
            "You are Shakespeare. Write a poem about airplanes.",
            "Explain the concept of cogging torque."
                   ]:
        invert_info = invert2("", prompt, True, model)
        print(json.dumps(invert_info), file=file)

