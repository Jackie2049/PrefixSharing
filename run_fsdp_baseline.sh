#!/bin/bash
set -ex

# verl 0.8.0 + FSDP baseline - PS=OFF
export CUDA_VISIBLE_DEVICES=1
export NCCL_TIMEOUT=600

source /jiangdingfeng/miniconda3/etc/profile.d/conda.sh
conda activate env-flex

cd /jiangdingfeng/zy/Termius/flex-attention

python3 -m verl.trainer.main_ppo \
    --config-path /jiangdingfeng/zy/Termius/PrefixSharing/dependency/verl_cdd9014f/verl/trainer/config \
    --config-name ppo_trainer \
    model_engine=dp \
    actor_rollout_ref.model.path=/jiangdingfeng/zy/Termius/models/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    algorithm.adv_estimator=grpo \
    data.train_files=/jiangdingfeng/zy/Termius/data/train.parquet \
    data.val_files=/jiangdingfeng/zy/Termius/data/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    data.shuffle=false \
    trainer.project_name=ps_fsdp_baseline \
    trainer.experiment_name=fsdp_off_baseline \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.logger='["console"]' \
    trainer.val_before_train=false \
    trainer.val_only=false \
    trainer.default_local_dir=./checkpoints/ps_fsdp_baseline

echo DONE
