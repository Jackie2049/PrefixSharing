#!/bin/bash
# PrefixSharing Round 3 - Experiment B: Batch-size scaling / HBM capacity (FSDP only)
# Purpose: Verify PS=ON reduced HBM enables larger batch, improving effective throughput
# Model: Qwen3-1.7B (16Q/8KV/128D, ~1.7B params, GQA 2:1)
# Runs 3 training steps per config, discard 1st warmup
# Constraint: GPU1 occupied, 7 GPUs available (0,2,3,4,5,6,7)
#             Megatron impossible on 4090 (DDP buffer > remaining HBM)
#             DP=7 => batch must be divisible by 7 (n=1)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../perf_results/round3_exp_b_logs"
mkdir -p "$LOG_DIR"

PYTHON=/home/zxw/miniconda3/envs/verl080/bin/python3
source ~/miniconda3/bin/activate verl080

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B"
DATA_DIR="/home/zxw/Termius/proj_prefix-sharing/data"
VERL_DIR="/home/zxw/verldir"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: Model config.json not found at $MODEL_PATH"
  exit 1
fi
echo "Using model: $MODEL_PATH"

FREE_GPUS="0,2,3,4,5,6,7"

# Batch-size scaling: FSDP DP=7, n=1, batch=7k
# Format: prompt|response|max_model_len|gpu_mem|batch_list
SCALING_TESTS=(
  "512|64|608|0.85|7,14,21,28,35,42,56,70,84"
  "1024|64|1088|0.85|7,14,21,28,35,42"
  "256|32|320|0.85|7,14,21,28,42,56,70,84,98"
)

TOTAL_TESTS=0
for spec in "${SCALING_TESTS[@]}"; do
  IFS='|' read -r _ _ _ _ batch_list <<< "$spec"
  for bs in $(echo "$batch_list" | tr ',' ' '); do
    TOTAL_TESTS=$((TOTAL_TESTS + 2))
  done
done

COUNT=0
SKIPPED=0

for spec in "${SCALING_TESTS[@]}"; do
  IFS='|' read -r prompt_len response_len max_model_len gpu_mem batch_list <<< "$spec"

  DATA_PATH="$DATA_DIR/train_ps_prompt${prompt_len}.parquet"
  TEST_DATA="$DATA_DIR/test_ps_prompt${prompt_len}.parquet"

  MAX_BATCH_PS0=0
  MAX_BATCH_PS1=0

  for bs in $(echo "$batch_list" | tr ',' ' '); do
    for ps in 0 1; do
      COUNT=$((COUNT + 1))
      LOGFILE="${LOG_DIR}/fsdp_7gpu_dp7_qwen3_1.7b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

      if [ -f "$LOGFILE" ] && grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
        SKIPPED=$((SKIPPED + 1))
        echo "[SKIP $COUNT/$TOTAL_TESTS] FSDP DP=7 bs=$bs p=$prompt_len r=$response_len PS=$ps"
        if [ "$ps" = "0" ]; then MAX_BATCH_PS0=$bs; else MAX_BATCH_PS1=$bs; fi
        continue
      fi

      rm -f "$LOGFILE" 2>/dev/null

      echo "========================================"
      echo "[RUN $COUNT/$TOTAL_TESTS] FSDP DP=7 bs=$bs prompt=$prompt_len response=$response_len PS=$ps"
      echo "========================================"

      if [ "$ps" = "1" ]; then
        export ENABLE_PREFIX_SHARING=1
        export VERL_USE_EXTERNAL_MODULES=prefix_sharing
        export PREFIX_SHARING_PATCHSET=verl080_fsdp
      else
        export ENABLE_PREFIX_SHARING=0
        unset VERL_USE_EXTERNAL_MODULES
        unset PREFIX_SHARING_PATCHSET
      fi

      export CUDA_VISIBLE_DEVICES=$FREE_GPUS
      NGPU=7

      cd $VERL_DIR

      timeout 900 $PYTHON -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="['$DATA_PATH']" \
        data.val_files="['$TEST_DATA']" \
        data.train_batch_size=$bs \
        data.max_prompt_length=$prompt_len \
        data.max_response_length=$response_len \
        data.filter_overlong_prompts=True \
        data.truncation=left \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=$bs \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem \
        actor_rollout_ref.rollout.max_num_seqs=$bs \
        actor_rollout_ref.rollout.max_model_len=$max_model_len \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
        actor_rollout_ref.rollout.agent.num_workers=1 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        trainer.balance_batch=True \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name=verl_r3b_fsdp_7gpu_dp7_ps${ps} \
        trainer.experiment_name=7gpu_dp7_qwen3_1.7b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
        trainer.n_gpus_per_node=$NGPU \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=3 \
        trainer.val_before_train=False \
        2>&1 | tee "$LOGFILE" || true

      if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
        echo "SUCCESS: $LOGFILE"
        if [ "$ps" = "0" ]; then MAX_BATCH_PS0=$bs; else MAX_BATCH_PS1=$bs; fi
      else
        echo "FAILED/OOM: $LOGFILE"
      fi

      ray stop --force 2>/dev/null || true
      sleep 5
    done
  done

  echo ""
  echo "=== MAX BATCH SUMMARY: FSDP DP=7 p=$prompt_len r=$response_len ==="
  echo "PS=OFF max_batch=${MAX_BATCH_PS0}, PS=ON max_batch=${MAX_BATCH_PS1}"
  echo ""
done

# ---- Extract detailed timing summary ----
echo ""
echo "=== ROUND 3 EXPERIMENT B: BATCH-SCALE TIMING SUMMARY ==="
echo "Engine | Parallel | BS | Prompt | Response | PS | step_s | gen_s | old_log_prob_s | update_actor_s | update_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy | OOM?"
echo "------|----------|----|--------|----------|----|--------|------|---------------|---------------|----------------|-------------|--------------|----------------|--------|-----"

for spec in "${SCALING_TESTS[@]}"; do
  IFS='|' read -r prompt_len response_len max_model_len gpu_mem batch_list <<< "$spec"

  for bs in $(echo "$batch_list" | tr ',' ' '); do
    for ps in 0 1; do
      LOGFILE="${LOG_DIR}/fsdp_7gpu_dp7_qwen3_1.7b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

      IS_OOM="N"
      if [ -f "$LOGFILE" ]; then
        if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
          STEP_TIME=$(grep -oP 'timing_s/step:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          GEN_TIME=$(grep -oP 'timing_s/gen:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          OLD_LOGPROB=$(grep -oP 'timing_s/old_log_prob:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          UPDATE_ACTOR=$(grep -oP 'timing_s/update_actor:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          UPDATE_WEIGHTS=$(grep -oP 'timing_s/update_weights:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          ACTOR_HBM=$(grep -oP 'actor/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
          CRITIC_HBM=$(grep -oP 'critic/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
          THROUGHPUT=$(grep -oP 'perf/throughput:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          ENTROPY=$(grep -oP 'actor/entropy:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
          echo "fsdp | 7gpu_dp7 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | ${STEP_TIME} | ${GEN_TIME} | ${OLD_LOGPROB} | ${UPDATE_ACTOR} | ${UPDATE_WEIGHTS} | ${ACTOR_HBM} | ${CRITIC_HBM} | ${THROUGHPUT} | ${ENTROPY} | ${IS_OOM}"
        else
          IS_OOM="Y"
          echo "fsdp | 7gpu_dp7 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | OOM | - | - | - | - | - | - | - | - | ${IS_OOM}"
        fi
      else
        echo "fsdp | 7gpu_dp7 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | MISSING | - | - | - | - | - | - | - | - | MISS"
      fi
    done
  done
done

echo "=== Skipped: ${SKIPPED} ==="
echo "=== END ==="
