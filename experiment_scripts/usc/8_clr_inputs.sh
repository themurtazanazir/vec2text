#!/bin/bash
source /home/mattfinlays/miniconda3/etc/profile.d/conda.sh
conda activate v2t
echo "Active Conda Environment: $CONDA_DEFAULT_ENV"

export DATA_DIR="/home/mattfinlays/vec2text/data"
export HF_HUB_CACHE=${DATA_DIR}/inversion/huggingface/hub/
export HF_HOME=${DATA_DIR}/inversion/huggingface/
export VEC2TEXT_CACHE=${DATA_DIR}/inversion/vec2text/
# export CUDA_LAUNCH_BLOCKING = 1
export WANDB_DIR=${DATA_DIR}/inversion/inversion_vec2text/
export TMPDIR=${DATA_DIR}/inversion/temp/

python vec2text/run.py \
	--per_device_train_batch_size 240 \
	--per_device_eval_batch_size 240 \
	--max_seq_length 16 \
	--num_train_epochs 40 \
	--max_eval_samples 500 \
	--eval_steps 25000 \
	--warmup_steps 100000 \
	--learning_rate 0.0002 \
	--dataset_name one_million_instructions \
	--model_name_or_path t5-base \
	--use_wandb=1 \
	--experiment inversion_from_random_transformed_hidden_states \
	--bf16=1 \
	--embedder_torch_dtype bfloat16 \
	--lr_scheduler_type constant_with_warmup \
	--use_frozen_embeddings_as_input 1 \
	--mock_embedder 0 \
	--use_wandb 1 \
	--use_less_data 1000000 \
	--embedder_model_name gpt2 \
	--max_new_tokens 8 \
	--output_dir ${DATA_DIR}/inversion/8_input_clr/ \
	--exp_group_name clr_ablations \
	>> out_8_clr_ablation.log
