#!/bin/bash
set -e

# ============================================================
# 0720 HBM test: PS ON + PS_TIMING=1 + RAY_DEDUP_LOGS=0 + rank0-only
#
# Changes vs 0717:
#   + export RAY_DEDUP_LOGS=0 (关闭Ray日志去重)
#   + rank0-only filter (只rank0打印PS-TIMING)
#
# Key config:
#   ENABLE_PREFIX_SHARING=1
#   PS_TIMING=1, RAY_DEDUP_LOGS=0
#   step data, max_prompt=6000, max_response=3000
#   micro_batch=4, TP=2, PP=2, SP=True
# ============================================================

source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTHONPATH=/home/ma-user/work/l00561472/prefix-sharing
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0
export ENABLE_PREFIX_SHARING=1
export PREFIX_SHARING_BACKEND=flash_atten_npu

export ENABLE_TRAINING_MONITOR=1
export TRAINING_MONITOR_SAVE_DIR=/home/ma-user/work/l00561472/metrics/0721-seq4K-micro4-rolloutN4-tp2-pp2-sp-PS-timing-nodedup-step-pl3000-rl1000

export PS_TIMING=1
export RAY_DEDUP_LOGS=0
export USE_FIXED_ROLLOUT=/home/ma-user/work/l00561472/data/synthetic_prefix_bs8_n4_step4_pl3000_rl1000_step.json
export TRANSFORMERS_VERBOSITY=error

cd /home/ma-user/work/l00561472
mkdir -p log

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    data.train_files=/home/ma-user/work/tmp/data/512_gsm8k/train.parquet \
    data.val_files=/home/ma-user/work/tmp/data/512_gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=3000 \
    data.max_response_length=1000 \
    +data.force_group_size=4 \
    actor_rollout_ref.model.path=/data/ttelab-gy1/w50057371/model/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=/data/ttelab-gy1/w50057371/model/Qwen2.5-0.5B \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.balance_batch=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.ref.megatron.use_remove_padding=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.context_parallel_size=1 \
    2>&1 | tee log/0721-seq4K-micro4-rolloutN4-tp2-pp2-sp-PS-timing-nodedup-step-pl3000-rl1000.log
