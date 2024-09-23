export HF_HUB_CACHE=/data/inversion/huggingface/hub/
export HF_HOME=/data/inversion/huggingface/
export VEC2TEXT_CACHE=/data/inversion/vec2text/
# export CUDA_LAUNCH_BLOCKING = 1
export WANDB_DIR=/data/inversion/inersion_vec2text/


nohup python run.py --per_device_train_batch_size 240 --per_device_eval_batch_size 240 --max_seq_length 16 --num_train_epochs 40 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --learning_rate 0.0002 --dataset_name one_million_instructions --model_name_or_path t5-base --use_wandb=1 --experiment inversion_from_hidden_states --bf16=1 --embedder_torch_dtype bfloat16 --lr_scheduler_type constant_with_warmup --use_frozen_embeddings_as_input 0 --mock_embedder 0 --use_wandb 1 --use_less_data 1000000 --embedder_model_name gpt2 --max_new_tokens 16 --output_dir /data/inversion/hidden_saves/ --exp_group_name 2024-09-22-hidden_states_exp-19_57 &
