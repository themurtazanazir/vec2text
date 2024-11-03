export TMPDIR=/data/inversion/temp/
export HF_HUB_CACHE=/data/inversion/huggingface/hub/
export HF_HOME=/data/inversion/huggingface/
export VEC2TEXT_CACHE=/data/inversion/vec2text/
# export CUDA_LAUNCH_BLOCKING = 1
export WANDB_DIR=/data/inversion/inversion_vec2text/

python -m pip install .

nohup python vec2text/run.py --per_device_train_batch_size 230\
                    --per_device_eval_batch_size 230\
                    --max_seq_length 64\
                    --num_train_epochs 100\
                    --max_eval_samples 1000\
                    --eval_steps 25000\
                    --warmup_steps 25000\
                    --learning_rate 0.0002\
                    --dataset_name one_million_instructions\
                    --model_name_or_path t5-base\
                    --use_wandb=1\
                    --experiment inversion_from_random_transformed_hidden_states\
                    --bf16=1 --embedder_torch_dtype bfloat16\
                    --lr_scheduler_type constant_with_warmup\
                    --use_frozen_embeddings_as_input 1\
                    --mock_embedder 0\
                    --embedder_model_name meta-llama/Llama-2-7b-hf\
                    --max_new_tokens 4\
                    --output_dir /data/inversion/hidden_saves_random_transformed_llama2_4_bit/\
                    --exp_group_name llama-random-transformed-4-bit &
