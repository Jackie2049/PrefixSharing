#!/usr/bin/env bash
# Focused Megatron single-GPU end-to-end training profiler
# Runs key parameter combinations, extracts timing from logs
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULTS_DIR="${SCRIPT_DIR}/../perf_results/megatron_single_gpu"
mkdir -p "$RESULTS_DIR"

PREFIX=/home/zxw/miniconda3/envs/verl080
MODEL_PATH=/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987
DATA_PATH=/home/zxw/verldir/data
cd /home/zxw/verldir

# Key combinations: 8 configs × 2 PS (on/off) = 16 runs
CONFIGS=(
  "2|64|32|0.6|8"
  "4|64|32|0.6|16"
  "8|64|32|0.6|32"
  "2|128|32|0.6|8"
  "2|64|64|0.6|12"
  "2|256|128|0.5|12"
  "2|512|256|0.4|12"
)

TOTAL=$(( ${#CONFIGS[@]} * 2 ))
COUNT=0

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r bs pl rl gmem max_num_seqs <<< "$cfg"

  for ps in 0 1; do
    COUNT=$((COUNT + 1))
    echo "========================================"
    echo "[$COUNT/$TOTAL] bs=${bs} prompt=${pl} response=${rl} PS=${ps} gmem=${gmem}"
    echo "========================================"

    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export CUDA_VISIBLE_DEVICES=0
    export VLLM_USE_V1=1
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export ENABLE_PREFIX_SHARING=$ps
    export PREFIX_SHARING_PATCHSET=verl080_mcore0161_ms0160

    LOGFILE="${RESULTS_DIR}/meg_bs${bs}_p${pl}_r${rl}_ps${ps}.log"

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
        actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
        actor_rollout_ref.rollout.max_model_len=$((pl + rl + 16)) \
        actor_rollout_ref.rollout.gpu_memory_utilization=$gmem \
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
    sleep 3
  done
done

echo "=== Megatron single GPU profile done ==="

# Extract timing summary
echo ""
echo "=== TIMING SUMMARY ==="
echo "Config | PS | step_time | gen_time | update_weights | grad_norm | entropy | HBM_GB | throughput"
echo "------|----|----------|---------|---------------|----------|--------|-------|----------"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r bs pl rl gmem max_num_seqs <<< "$cfg"
  for ps in 0 1; do
    LOGFILE="${RESULTS_DIR}/meg_bs${bs}_p${pl}_r${rl}_ps${ps}.log"
    if [ -f "$LOGFILE" ]; then
      STEP_TIME=$(grep -oP 'timing_s/step:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "N/A")
      GEN_TIME=$(grep -oP 'timing_s/gen:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "N/A")
      UPDATE_TIME=$(grep -oP 'timing_s/update_weights:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "N/A")
      GRAD_NORM=$(grep -oP 'actor/grad_norm:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "N/A")
      ENTROPY=$(grep -oP 'actor/entropy:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "N/A")
      HBM=$(grep -oP 'actor/perf/max_memory_allocated_gb:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "N/A")
      THROUGHPUT=$(grep -oP 'perf/throughput:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "N/A")
      echo "bs${bs}_p${pl}_r${rl} | PS=${ps} | ${STEP_TIME:-N/A}s | ${GEN_TIME:-N/A}s | ${UPDATE_TIME:-N/A}s | ${GRAD_NORM:-N/A} | ${ENTROPY:-N/A} | ${HBM:-N/A}GB | ${THROUGHPUT:-N/A}"
    fi
  done
done

echo "=== END ==="
