import json
import optimize_transfer
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
os.environ["HF_HOME"]="/home/mnazir/vec2text/data/test/huggingface/"
os.environ["HF_HUB_CACHE"]="/home/mnazir/vec2text/data/test/huggingface/hub/"
os.environ["VEC2TEXT_CACHE"]="/home/mnazir/vec2text/data/test/vec2text/"
os.environ["WANDB_DIR"]="/home/mnazir/vec2text/data/test/"

nltk.download("punkt_tab")


def invert(sys, ins, chat_format):
    strings = [format(sys, ins, chat_format).strip()]
    print(f"{strings=}", flush=True)
    t = trainer.embedder_tokenizer
    print(f"{t.padding_side=}")
    inputs = t(
        strings,
        return_tensors="pt",
        padding="max_length",
        # max_length=trainer.model.embedder.max_length,
        max_length=64,
        truncation=True,
    )
    inputs = {f"embedder_{k}": v for k, v in inputs.items()}
    gen_kwargs = copy.copy(trainer.gen_kwargs)
    max_length = trainer.model.config.max_seq_length
    gen_kwargs["max_length"] = max_length
    outputs = trainer.generate(inputs, generation_kwargs={"max_new_tokens": 64})
    output_strings = trainer.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_strings[0]


def get_overlap_toks(llama2_tokenizer, other_tokenizer, llama2_chosen_toks):
    llama_vocab = llama2_tokenizer.get_vocab()
    other_vocab = other_tokenizer.get_vocab()

    overlap = set(llama_vocab).intersection(set(other_vocab))  # in string formats
    chosen_strings = {k for k, v in llama_vocab.items() if v in llama2_chosen_toks}

    overlap_with_chosen = chosen_strings.intersection(overlap)
    print(f"{len(overlap_with_chosen)=}")

    remaining = list(overlap - overlap_with_chosen)
    overlap = list(sorted(overlap))
    print(f"{len(overlap)=}")
    import random

    random.seed(4673)
    random.shuffle(overlap)
    # random.shuffle(remaining)

    # total_overlap_chosen = list(overlap_with_chosen) + remaining[:4200-len(overlap_with_chosen)]
    # total_overlap_chosen = remaining
    total_overlap_chosen = overlap[:]
    llama_overlap_toks = [llama_vocab[k] for k in total_overlap_chosen]
    other_overlap_toks = [other_vocab[k] for k in total_overlap_chosen]

    return llama_overlap_toks, other_overlap_toks


def get_logprobs(model, tokenizer, embedder_input_ids, embedder_attention_mask, max_new_tokens):
    # inputs = tokenizer(strings, return_tensors='pt', padding='max_length',
    #         # max_length=trainer.model.embedder.max_length,
    #         max_length=64,
    #         truncation=True)
    # embedder_input_ids = inputs.input_ids
    # embedder_attention_mask = inputs.attention_mask
    device = next(model.parameters()).device
    embedder_input_ids = embedder_input_ids.to(device)
    embedder_attention_mask = embedder_attention_mask.to(device)
    output = model.generate(
        input_ids=embedder_input_ids,
        attention_mask=embedder_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    ##!!  this part is usually in lms and not in embedder.
    logits = torch.cat([i.unsqueeze(1) for i in output.scores], dim=1)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    return logprobs


def generate(embedder_input_ids, embedder_attention_mask, optimize_fn, generation_steps_to_use, debug=False):

    llama_overlap_toks, other_overlap_toks = get_overlap_toks(
        trainer.embedder_tokenizer, other_tokenizer, model.embedder.chosen_tokens
    )
    # messages = [{"role":"system", "content":sys},{"role":"user", "content":prompt}]
    # text = other_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    other_logprobs = get_logprobs(
        other_llm, other_tokenizer, embedder_input_ids, embedder_attention_mask,
            max_new_tokens=generation_steps_to_use
    )
    if optimize_fn is not None:
        llama_overlap_toks, other_overlap_toks = get_overlap_toks(
            trainer.embedder_tokenizer, other_tokenizer, model.embedder.chosen_tokens
        )
        # messages = [{"role":"system", "content":sys},{"role":"user", "content":prompt}]
        # text = other_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if debug:
            _, decoded = torch.max(other_logprobs, dim=-1)
            decoded_string = other_tokenizer.decode(decoded)
            print(f"{decoded_string=}")
        import numpy as np

        llama_unembed = model.embedder.model.lm_head.weight.data.float()
        batch_hidden_states = []
        for lps in other_logprobs:
            llama2_hidden_state = optimize_fn(
                llama_unembed, lps, llama_overlap_toks, other_overlap_toks
            )
            batch_hidden_states.append(llama2_hidden_state)
        batch_hidden_states = torch.stack(batch_hidden_states)  # b x max_toks x dims
        # b x max_toks x vocab
        llama2_logits = batch_hidden_states @ llama_unembed.T
        llama2_logprobs = torch.nn.functional.log_softmax(llama2_logits, dim=-1)
    else:
        # assuming TOKENIZER IS EXACTLY same
        llama2_logprobs = other_logprobs
    print(f"{other_logprobs.shape=}")
    if debug:
        _, decoded = torch.max(other_logprobs, dim=-1)
        decoded_string = other_tokenizer.decode(decoded)
        print(f"{decoded_string=}")
    import numpy as np

    llama_unembed = model.embedder.model.lm_head.weight.data.float()
    batch_hidden_states = []
    for lps in other_logprobs:
        llama2_hidden_state = optimize_fn(
            llama_unembed, lps, llama_overlap_toks, other_overlap_toks
            )
        batch_hidden_states.append(llama2_hidden_state)
    batch_hidden_states = torch.stack(batch_hidden_states)  # b x max_toks x dims
    # b x max_toks x vocab
    llama2_logits = batch_hidden_states @ llama_unembed.T
    llama2_logprobs = torch.nn.functional.log_softmax(llama2_logits, dim=-1)
    llama2_logprobs = llama2_logprobs[:, :, model.embedder.chosen_tokens]
    alr = llama2_logprobs[:, :, 1:] - llama2_logprobs[:, :, 0:1]
    embeddings = model.embedding_transform(alr)
    attention_mask = torch.ones(
        (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
    )

    print(f"{embeddings.shape=}")
    output = model.encoder_decoder.generate(
        # required: input embeddings
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        # optional: input IDs (for starting generation).
        # typically not set unless generating prefixes for
        # reranking.
        max_length=64,
        # **generation_kwargs,
    )
    return output

    strings = model.tokenizer.batch_decode(output, skip_special_tokens=True)
    return strings


def eval_generation_metrics(
    trainer, dataloader: torch.utils.data.DataLoader, transform_fn,
    generation_steps_to_use=16,
) -> Dict[str, float]:
    # Get decoded text. Note that this is different than `preds`, which
    # is used to compute the loss.
    preds_sample_list, preds_sample_labels_list = _get_decoded_sequences(
        trainer,
        dataloader=dataloader,
        n=10000,
        transform_fn=transform_fn,
        generation_steps_to_use=generation_steps_to_use,
    )

    # Log BLEU, log table of text.
    decoded_preds = trainer.tokenizer.batch_decode(
        preds_sample_list, skip_special_tokens=True
    )
    decoded_labels = trainer.tokenizer.batch_decode(
        preds_sample_labels_list, skip_special_tokens=True
    )
    bleu_result = trainer._text_comparison_metrics(
        predictions_ids=preds_sample_list,
        predictions_str=decoded_preds,
        references_ids=preds_sample_labels_list,
        references_str=decoded_labels,
    )
    trainer._log_preds_table(
        table_key="val_text_preds",
        decoded_preds=decoded_preds,
        decoded_labels=decoded_labels,
    )

    if not len(decoded_preds):
        return {}
    print("[pred]", decoded_preds[0])
    print("[true]", decoded_labels[0])
    print("\n\n")
    print("[pred]", decoded_preds[1])
    print("[true]", decoded_labels[1])
    print("\n\n")
    print("[pred]", decoded_preds[2])
    print("[true]", decoded_labels[2])

    # Compute sims of eval data using embedder.
    preds_sample = torch.tensor(preds_sample_list, device=trainer.args.device)[:128]
    preds_sample_labels = torch.tensor(
        preds_sample_labels_list, device=trainer.args.device
    )[:128]

    # Log num tokens.
    num_tokens_metrics = {
        "pred_num_tokens": (
            (preds_sample != trainer.pad_token_id)
            & (preds_sample != trainer.bos_token_id)
        )
        .sum(1)
        .float()
        .mean()
        .item(),
        "true_num_tokens": (
            (preds_sample_labels != trainer.pad_token_id)
            & (preds_sample_labels != trainer.bos_token_id)
        )
        .sum(1)
        .float()
        .mean()
        .item(),
    }

    # Fix eos token on generated text.
    # bos_token_id = trainer.embedder_tokenizer.pad_token_id
    # assert (preds_sample[:, 0] == bos_token_id).all()
    eos_token_id = trainer.embedder_tokenizer.eos_token_id
    if eos_token_id is not None:
        eos_tokens = (
            torch.ones(
                (len(preds_sample), 1),
                dtype=torch.long,
                device=trainer.args.device,
            )
            * eos_token_id
        )
        preds_sample = torch.cat((preds_sample[:, 1:], eos_tokens), dim=1)
        # assert preds_sample.shape == preds_sample_labels.shape

    sim_result = {"emb_cos_sim": 0, "emb_cos_sim_sem": 0}

    # Store stuff for access later.
    # trainer.preds_emb = preds_emb.cpu()
    # trainer.labels_emb = labels_emb.cpu()
    trainer.preds_sample_list = preds_sample_list
    trainer.preds_sample_labels_list = preds_sample_labels_list

    metrics = {**num_tokens_metrics, **bleu_result, **sim_result}
    return metrics


def _get_decoded_sequences(
    trainer,
    dataloader: torch.utils.data.DataLoader,
    n: int,
    transform_fn,
    generation_steps_to_use=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Iterates through eval dataset and does decoding.

    TODO: do this better. We shouldn't need to iterate through eval set twice
    but I don't want to copy 1000 lines of code to change their eval loop...

    Probably want custom eval eventually. Also this depends on eval data being
    in the same order which is annoying.
    """
    assert not trainer.model.training

    gen_kwargs = copy.copy(trainer.gen_kwargs)

    all_preds = []
    all_labels = []
    for step, inputs in enumerate(
        tqdm.tqdm(dataloader, desc="generating from val", leave=False)
    ):
        # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        inputs_cuda = {k: v.to(trainer.args.device) for k, v in inputs.items()}
        max_length = trainer.model.config.max_seq_length
        gen_kwargs["max_length"] = max_length
        with torch.no_grad():
            generated_text = generate(
                embedder_input_ids=inputs_cuda["embedder_input_ids"],
                embedder_attention_mask=inputs_cuda["embedder_attention_mask"],
                optimize_fn=transform_fn,
                generation_steps_to_use=generation_steps_to_use,
                # generation_kwargs=gen_kwargs
            )
        if generated_text.shape[1] < max_length:
            # Pad generated text to max length
            pad_tokens = (
                torch.ones(
                    (generated_text.shape[0], max_length - generated_text.shape[1]),
                    dtype=torch.long,
                    device=generated_text.device,
                )
                * trainer.pad_token_id
            )
            generated_text = torch.cat((generated_text, pad_tokens), dim=1)

        true_input_ids = inputs["input_ids"]
        if true_input_ids.shape[1] < max_length:
            # Pad true text to max length
            # Pad generated text to max length
            pad_tokens = (
                torch.ones(
                    (true_input_ids.shape[0], max_length - true_input_ids.shape[1]),
                    dtype=torch.long,
                    device=true_input_ids.device,
                )
                * trainer.pad_token_id
            )
            true_input_ids = torch.cat((true_input_ids, pad_tokens), dim=1)

        all_preds.extend(generated_text.cpu().tolist())
        all_labels.extend(true_input_ids.cpu().tolist())
        if len(all_preds) >= n:
            break

    return all_preds, all_labels


def format(system_message, instruction, chat_format):
    if chat_format:
        return f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
    else:
        return system_message + "\n\n" + instruction


def get_val_datasets():
    from vec2text.data_helpers import dataset_from_args

    embedder_tokenizer = other_tokenizer
    text_column_name = "text"
    max_seq_length = experiment.model_args.max_seq_length
    padding = False
    tokenizer = trainer.tokenizer

    def tokenize_fn(examples) -> Dict[str, torch.Tensor]:
        if "prefix" not in examples:
            examples["prefix"] = [""] * len(examples[text_column_name])
            examples["suffix"] = examples[text_column_name]

        formatted_text = [
            other_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": instruction},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for (system_message, instruction) in zip(
                examples["prefix"], examples["suffix"]
            )
        ]
        output = tokenizer(
            examples[text_column_name],  # dont invert in the chat format
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = other_tokenizer(
            text=formatted_text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    raw_datasets = dataset_from_args(experiment.data_args)
    val_datasets_dict = load_standard_val_datasets()
    for ds_name in ["wikibio", "arxiv", "ag_news", ]:
        val_datasets_dict.pop(ds_name)
    val_datasets_dict["one_million_instructions"] = raw_datasets["validation"]

    for name, dataset in val_datasets_dict.items():
        max_eval_samples = min(len(dataset), experiment.data_args.max_eval_samples)
        val_datasets_dict[name] = val_datasets_dict[name].select(
            range(max_eval_samples)
        )
        val_datasets_dict[name] = val_datasets_dict[name].add_column(
            "idx", range(len(val_datasets_dict[name]))
        )
        val_datasets_dict[name].set_format("pt")

    ALLOWED_COLUMN_NAMES = {"frozen_embeddings"}

    for key in val_datasets_dict:
        column_names = list(val_datasets_dict[key].features)
        column_names = [c for c in column_names if c not in ALLOWED_COLUMN_NAMES]
        val_datasets_dict[key] = dataset_map_multi_worker(
            dataset=val_datasets_dict[key],
            map_fn=tokenize_fn,
            batched=True,
            batch_size=1024,
            num_proc=get_num_proc(),
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    val_datasets_dict = val_datasets_dict.filter(lambda ex: ex["length"] > 1)

    return val_datasets_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cmd = "--per_device_train_batch_size 250 --per_device_eval_batch_size 250 --max_seq_length 64 --num_train_epochs 100 --max_eval_samples 1000 --eval_steps 25000 --warmup_steps 25000 --learning_rate 0.0002 --dataset_name one_million_instructions --model_name_or_path t5-base --use_wandb=0 --experiment inversion_from_hidden_states --bf16=1 --embedder_torch_dtype bfloat16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 1 --mock_embedder 0 --embedder_model_name llama2_chat-random_k-alr --max_new_tokens 16 --output_dir /home/mnazir/vec2text/data/test/experiments/llama2_chat-random_k-alr-16-toks-bugfix-4-nodes/ --exp_group_name llama2-chat --extra_tokens 100"


parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses(cmd.split())
experiment = experiment_from_args(model_args, data_args, training_args)

model = experiment.load_model()


ckpt = experiment._get_checkpoint()
model.embedder.cpu()
from vec2text.utils import MockEmbedder
# del model.embedder
# model.embedder = MockEmbedder()
print("CKPT", ckpt)
trainer = experiment.trainer_cls(
    model=model,
    data_collator=experiment.get_collator(tokenizer=model.tokenizer),
    args=experiment.training_args,
)
trainer._load_from_checkpoint(ckpt)
trainer.model.eval()



# other_llm_name = "Qwen/Qwen2.5-7B-Instruct"
# other_llm_name = "meta-llama/Llama-3.1-8B-Instruct"
# other_llm_name = "google/gemma-3-4b-it"
# other_llm_name = "meta-llama/Llama-2-7b-chat-hf"
other_llm_name = "mistralai/Mistral-7B-Instruct-v0.3"
#other_llm_name = "meta-llama/Llama-2-13b-chat-hf"
other_llm = AutoModelForCausalLM.from_pretrained(
    other_llm_name, torch_dtype=torch.bfloat16
)
other_llm.eval()
other_llm.to(device)
other_tokenizer = AutoTokenizer.from_pretrained(other_llm_name)
other_tokenizer.padding_side = "left"
other_tokenizer.pad_token = other_tokenizer.eos_token

# invert from llama once to set its chosen_tokens
output = invert("", "reverse", True)
print(output)

val_datasets_dict = get_val_datasets()
metrics = []
for gen_steps in [16, 32]:
    for key in val_datasets_dict:
        for transform_fn in [
            optimize_transfer.optimize_transform,
            #optimize_transfer.optimize_transform_matt,
        ]:
            dl = trainer.get_eval_dataloader(val_datasets_dict[key])
            out = eval_generation_metrics(trainer, dl, transform_fn=transform_fn,  generation_steps_to_use=gen_steps)
            metrics.append(
                {
                    "ds": key,
                    "tranform_fn": transform_fn.__name__,
                    "metrics": out,
                    "embedder": other_llm_name,
                    "gen_steps":gen_steps,
                }
            )

with open(f"transform_metrics_{other_llm_name.replace('/', '__')}_new_test.json", "w") as f:
    json.dump(metrics, f, indent=4)
# val_datasets_dict = load_standard_val_datasets()
# for name, dataset in val_datasets_dict.items():
#     max_eval_samples = min(len(dataset), experiment.data_args.max_eval_samples)
#     val_datasets_dict[name] = val_datasets_dict[name].select(
#         range(max_eval_samples)
#     )
#     val_datasets_dict[name] = val_datasets_dict[name].add_column(
#         "idx", range(len(val_datasets_dict[name]))
#     )
#     val_datasets_dict[name].set_format("pt")
