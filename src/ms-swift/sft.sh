CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_PIXELS=2000 \
VIDEO_MAX_PIXELS=16384 \
swift sft \
    --model /home/huangyibin/model/Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --freeze_vit true \
    --dataset '/home/huangyibin/v2lo/general/a800_general.json' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --split_dataset_ratio 0.01 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 2 \
    --eval_steps 250 \
    --save_steps 250 \
    --save_total_limit 50 \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir output/sft \
    --dataloader_num_workers 4 \
    --dataset_num_proc 64 \
    --attn_impl flash_attn \
    --lazy_tokenize true \
    --deepspeed zero2 \

# MAX_PIXELS=200704 \
