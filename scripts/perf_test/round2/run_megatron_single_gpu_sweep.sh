#!/usr/bin/env bash
# Megatron single-GPU end-to-end training profile runner
# Scans batch_size × prompt_len × response_len × PS on/off

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../perf_results/megatron_single_gpu"
mkdir -p "$RESULTS_DIR"

PREFIX=/home/zxw/miniconda3/envs/verl080
MODEL_PATH=/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987
DATA_PATH=/home/zxw/verldir/data

cd /home/zxw/verldir

# Parameter sweeps
BATCH_SIZES="2 4 8"
PROMPT_LENS="64 128 256 512"
RESPONSE_LENS="32 64 128 256"

TOTAL=$(( $(echo $BATCH_SIZES | wc -w) * $(echo $PROMPT_LENS | wc -w) * $(echo $RESPONSE_LENS | wc -w) * 2 ))
COUNT=0

for bs in $BATCH_SIZES; do
  for pl in $PROMPT_LENS; do
    for rl in $RESPONSE_LENS; do
      for ps in 0 1; do
        COUNT=$((COUNT + 1))
        echo "========================================"
        echo "[$COUNT/$TOTAL] Megatron: bs=${bs} prompt=${pl} response=${rl} PS=${ps}"
        echo "========================================"

        # OOM guard: cap total seqlen
        total_seq=$((pl + rl))
        gpu_mem="0.5"
        if [ "$total_seq" -le 256 ]; then gpu_mem="0.6"; fi
        if [ "$total_seq" -ge 1024 ]; then
          # Only run bs=2 for long sequences
          if [ "$bs" -gt 2 ]; then
            echo "SKIP (OOM risk)"
            continue
          fi
          gpu_mem="0.4"
        fi

        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export CUDA_VISIBLE_DEVICES=0
        export VLLM_USE_V1=1
        export FLASHINFER_DISABLE_VERSION_CHECK=1
        export ENABLE_PREFIX_SHARING=$ps
        export PREFIX_SHARING_PATCHSET=verl080_mcore0161_ms0160

        LOGFILE="${RESULTS_DIR}/meg_bs${bs}_p${pl}_r${rl}_ps${ps}.log"
        # total_training_steps=3 (1 warmup + 2 measured)
        # val_before_train=True so we get val metrics
        timeout 300 \
          $PREFIX/bin/python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=gae \
            data.train_files="['$DATA_PATH/train.parquet']" \
            data.val_files="['$DATA_PATH/test.parquet']" \
            data.train_batch_size=$bs \
            data.max_prompt_length=$pl \
            data.max_response_length=$rl \
            data.filter_overlong_prompts=True \
            data.truncation=error \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.ppo_mini_batch_size=$bs \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
            actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
            actor_rollout_ref.actor.megatron.param_offload=False \
            actor_rollout_ref.actor.megatron.optimizer_offload=False \
            actor_rollout_ref.actor.megatron.grad_offload=False \
            actor_rollout_ref.actor.megatron.use_distributed_optimizer=False \
            actor_rollout_ref.actor.megatron.sequence_parallel=False \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.max_num_seqs=$((bs * 2)) \
            actor_rollout_ref.rollout.max_model_len=$((pl + rl + 16)) \
            actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem \
            actor_rollout_ref.rollout.n=1 \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
            actor_rollout_ref.rollout.agent.num_workers=1 \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
            actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
            actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
            actor_rollout_ref.ref.megatron.param_offload=True \
            trainer.balance_batch=True \
            trainer.critic_warmup=0 \
            trainer.logger='["console"]' \
            trainer.project_name=verl_megatron_test \
            trainer.experiment_name=meg_bs${bs}_p${pl}_r${rl}_ps${ps} \
            trainer.n_gpus_per_node=1 \
            trainer.nnodes=1 \
            trainer.save_freq=-1 \
            trainer.test_freq=1 \
            trainer.total_epochs=1 \
            trainer.total_training_steps=3 \
            trainer.val_before_train=False \
            model_engine=megatron \
            2>&1 | tee "$LOGFILE"

        echo "Done: $LOGFILE"
        # Brief delay to let GPU cool
        sleep 5
      done
    done
  done
done

echo "=== All $COUNT Megatron single GPU tests complete ==="
