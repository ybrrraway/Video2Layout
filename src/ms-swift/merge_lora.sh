# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /home/huangyibin/ms-swift/examples/train/multimodal/space/output/only_answer/v0-20250910-183116/checkpoint-500 \
    --merge_lora true
