#!/bin/bash
# REMINDER: this script uses test data split and should ONLY be used for debugging. DO NOT use for training.

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TARGET_DIR="$SCRIPT_DIR/.."
cd "$TARGET_DIR" || exit

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/huangyibin/model/Qwen/Qwen2.5-VL-3B-Instruct
python3 -m verl.trainer.main \
    config=/home/huangyibin/EasyR1/examples/config.yaml \
    data.train_files=/home/huangyibin/v2lo/real/esayr1_data.json \
    data.val_files=/home/huangyibin/v2lo/real/esayr1_data.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=32 \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=25 \
    trainer.experiment_name=rl \
    trainer.save_checkpoint_path=checkpoints/rl \