export LD_LIBRARY_PATH=/mnt/tidalfs-bdsz01/usr/xiangyi3/miniconda3/envs/swift_qwen_35/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export HTTP_PROXY=10.7.4.2:3128
export HTTPS_PROXY=10.7.4.2:3128
# Debug version: 1 card for debugging
# Usage: bash examples/models/qwen3_5/packing_debug.sh

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model /mnt/tidalfs-bdsz01/dataset/llm_ckpt/Qwen3.5-0.8B \
    --save_safetensors true \
    --dataset 'AI-ModelScope/LongAlpaca-12k' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 1 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --packing true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir megatron_output_debug/Qwen3.5-35B-A3B \
    --eval_steps 5 \
    --save_steps 5 \
    --max_length 32768 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --no_save_optim true \
    --no_save_rng true \
    --moe_expert_capacity_factor 2 \
    --mtp_num_layers 1 \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 0.64 \
    --attention_backend flash \
    --train_iters 10
