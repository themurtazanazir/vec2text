#!/bin/bash
source /home/mattfinlays/miniconda3/etc/profile.d/conda.sh
conda activate v2t
echo "Active Conda Environment: $CONDA_DEFAULT_ENV"

export DATA_DIR="/home/mattfinlays/vec2text/data"
export HF_HUB_CACHE=${DATA_DIR}/inversion/huggingface/hub/
export HF_HOME=${DATA_DIR}/inversion/huggingface/
export VEC2TEXT_CACHE=${DATA_DIR}/inversion/vec2text/
export WANDB_DIR=${DATA_DIR}/inversion/inversion_vec2text/
export TMPDIR=${DATA_DIR}/inversion/temp/

python vec2text/run.py --per_device_train_batch_size 230\
	--per_device_eval_batch_size 230\
	--max_seq_length 64\
	--num_train_epochs 40\
	--max_eval_samples 1000\
	--eval_steps 5000\
	--warmup_steps 5000\
	--learning_rate $1\
	--dataset_name one_million_instructions\
	--model_name_or_path t5-base\
	--use_wandb=1\
	--experiment inversion_from_hidden_states\
	--bf16=1 --embedder_torch_dtype bfloat16\
	--lr_scheduler_type constant_with_warmup\
	--use_frozen_embeddings_as_input 1\
	--mock_embedder 0\
	--embedder_model_name "gpt2-random_k-alr"\
	--max_new_tokens 1\
	--output_dir ${DATA_DIR}/inversion/gpt2-random_k-alr-single-token-LR${1}/\
	--exp_group_name gpt2-random_k-alr-single-token_lr_0.0002\
	--use_less_data 50000\
	--extra_tokens 100
