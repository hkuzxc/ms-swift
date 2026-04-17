# 8卡 H20 141GB，Qwen3.5-397B-A17B，LoRA + DeepSpeed Zero-3 Offload
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /mnt/tidalfs-bdsz01/dataset/llm_ckpt/qwen3.5/Qwen3.5-397B-A17B \
    --tuner_type lora \
    --dataset 'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --experts_impl grouped_mm \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --group_by_length true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output/Qwen3.5-397B-A17B \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero3_offload
