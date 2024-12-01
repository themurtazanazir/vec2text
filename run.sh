export TMPDIR=/data/inversion/temp/
export HF_HUB_CACHE=/data/inversion/huggingface/hub/
export HF_HOME=/data/inversion/huggingface/
export VEC2TEXT_CACHE=/data/inversion/vec2text/
# export CUDA_LAUNCH_BLOCKING = 1
export WANDB_DIR=/data/inversion/inversion_vec2text/

python -m pip install -e .
rm -r build

nohup python vec2text/run.py 
                    --per_device_train_batch_size 240\
                    --per_device_eval_batch_size 240\
                    --max_seq_length 64\
                    --num_train_epochs 100\
                    --max_eval_samples 1000\
                    --eval_steps 25000\
                    --warmup_steps 100000\
                    --learning_rate 0.0002\
                    --dataset_name one_million_instructions\
                    --model_name_or_path t5-base\
                    --use_wandb=1\
                    --experiment reverse_inversion_from_hidden_states\
                    --bf16=1 --embedder_torch_dtype bfloat16\
                    --lr_scheduler_type constant_with_warmup\
                    --use_frozen_embeddings_as_input 1\
                    --mock_embedder 0\
                    --embedder_model_name gpt2-random_k-alr\
                    --max_new_tokens 1\
                    --output_dir /data/inversion/reverse_hidden_random_k_alr/\
                    --exp_group_name gpt2-reverse
                    --extra_tokens 100 &

