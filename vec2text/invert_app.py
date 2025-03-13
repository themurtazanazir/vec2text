import os
os.environ["TMPDIR"]="/workspace/mnazir/vec2text/temp/"
os.environ["HF_HUB_CACHE"]="/workspace/mnazir/vec2text/huggingface/hub/"
os.environ["HF_HOME"]="/workspace/mnazir/vec2text/huggingface/"
os.environ["VEC2TEXT_CACHE"]="/workspace/mnazir/vec2text/vec2text/"
os.environ["WANDB_DIR"]="/workspace/mnazir/vec2text/"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import transformers
from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments
import copy

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cmd = "--per_device_train_batch_size 250 --per_device_eval_batch_size 250 --max_seq_length 64 --num_train_epochs 100 --max_eval_samples 1000 --eval_steps 25000 --warmup_steps 25000 --learning_rate 0.0002 --dataset_name one_million_instructions --model_name_or_path t5-base --use_wandb=0 --experiment inversion_from_hidden_states --bf16=1 --embedder_torch_dtype bfloat16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --embedder_model_name llama2-random_k-alr --max_new_tokens 16 --output_dir /home/mnazir/vec2text/data/inversion/experiments/llama2-random_k-alr-16_toks/ --exp_group_name llama2-base --extra_tokens 100"
cmd = "--per_device_train_batch_size 250 --per_device_eval_batch_size 250 --max_seq_length 64 --num_train_epochs 100 --max_eval_samples 1000 --eval_steps 25000 --warmup_steps 25000 --learning_rate 0.0002 --dataset_name one_million_instructions --model_name_or_path t5-base --use_wandb=0 --experiment inversion_from_hidden_states --bf16=1 --embedder_torch_dtype bfloat16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --embedder_model_name llama2_chat-random_k-alr --max_new_tokens 16 --output_dir /workspace/llama2_chat-random_k-alr-16-toks-bugfix-4-nodes/ --exp_group_name llama2-chat --extra_tokens 100"


parser = transformers.HfArgumentParser(
(ModelArguments, DataArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses(cmd.split())
experiment = experiment_from_args(model_args, data_args, training_args)
# from vec2text import analyze_utils
# experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
#     "jxm/t5-base__llama-7b__one-million-instructions__emb"
# )
# print(f"{trainer=}")
# print(f"{trainer.model=}")
# trainer.model.use_frozen_embeddings_as_input = False
# trainer.args.per_device_eval_batch_size = 16

model = experiment.load_model()


def format(system_message, instruction, chat_format):
    if chat_format:
        return f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
    else:
        return system_message + "\n\n" + instruction

def generate(system, instruction, chat_format=False):
    inp_string = format(system, instruction, chat_format)

    tok = model.embedder_tokenizer([inp_string], return_tensors='pt', padding=True)
    tok = {k:v.to(device) for k, v in tok.items()}
    out = model.embedder.model.generate(**tok, max_new_tokens=48)
    return embedder_tokenizer.batch_decode(out)[0]

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
