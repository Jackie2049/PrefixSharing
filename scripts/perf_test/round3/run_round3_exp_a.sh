#!/bin/bash
# PrefixSharing Round 3 - Experiment A: Phase-level attribution (FSDP only)
# Purpose: Determine if PS=ON slows training-side phases, and where
# Uses verl marked_timer: gen, old_log_prob, RefPolicy, update_actor, update_weights
# Model: Qwen3-1.7B (16Q/8KV/128D, ~1.7B params, GQA 2:1)
# Engine: FSDP (7GPU DP=7)
# Constraint: GPU1 occupied by other user, 7 GPUs available (0,2,3,4,5,6,7)
#             Megatron impossible on 4090 colocate (DDP buffer > remaining HBM for vLLM)
#             DP=7 => batch*n must be divisible by 7 => n=1, batch=7k

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../perf_results/round3_exp_a_logs"
mkdir -p "$LOG_DIR"

PYTHON=/home/zxw/miniconda3/envs/verl080/bin/python3
source ~/miniconda3/bin/activate verl080

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

# Model paths
MODEL_PATH="/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B"
DATA_DIR="/home/zxw/Termius/proj_prefix-sharing/data"
VERL_DIR="/home/zxw/verldir"

# Verify model exists
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: Model config.json not found at $MODEL_PATH"
  exit 1
fi
echo "Using model: $MODEL_PATH"

FREE_GPUS="0,2,3,4,5,6,7"

# FSDP DP=7 test matrix - batch must be divisible by 7, n=1
# Format: batch|prompt|response|max_model_len|gpu_mem
TESTS=(
  "14|256|32|320|0.85"
  "14|512|64|608|0.85"
  "21|256|32|320|0.85"
  "14|1024|64|1088|0.85"
  "28|256|32|320|0.85"
  "28|512|64|608|0.85"
  "7|1024|64|1088|0.85"
)

TOTAL=$(( ${#TESTS[@]} * 2 ))
COUNT=0
SKIPPED=0

for test_spec in "${TESTS[@]}"; do
  IFS='|' read -r bs prompt_len response_len max_model_len gpu_mem <<< "$test_spec"

  DATA_PATH="$DATA_DIR/train_ps_prompt${prompt_len}.parquet"
  TEST_DATA="$DATA_DIR/test_ps_prompt${prompt_len}.parquet"

  for ps in 0 1; do
    COUNT=$((COUNT + 1))
    LOGFILE="${LOG_DIR}/fsdp_7gpu_dp7_qwen3_1.7b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

    # Skip if already completed successfully
    if [ -f "$LOGFILE" ] && grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
      SKIPPED=$((SKIPPED + 1))
      echo "[SKIP $COUNT/$TOTAL] FSDP DP=7 bs=$bs p=$prompt_len r=$response_len PS=$ps"
      continue
    fi

    rm -f "$LOGFILE" 2>/dev/null

    echo "========================================"
    echo "[RUN $COUNT/$TOTAL] FSDP DP=7 bs=$bs prompt=$prompt_len response=$response_len PS=$ps"
    echo "========================================"

    # Configure PS environment
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

    timeout 600 $PYTHON -m verl.trainer.main_ppo \
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
      trainer.project_name=verl_r3a_fsdp_7gpu_dp7_ps${ps} \
      trainer.experiment_name=7gpu_dp7_qwen3_1.7b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
      trainer.n_gpus_per_node=$NGPU \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      trainer.total_epochs=1 \
      trainer.total_training_steps=1 \
      trainer.val_before_train=False \
      2>&1 | tee "$LOGFILE" || true

    # Check completion
    if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
      echo "SUCCESS: $LOGFILE"
    else
      echo "FAILED: $LOGFILE"
    fi

    ray stop --force 2>/dev/null || true
    sleep 5
  done
done

# ---- Extract phase-level timing summary ----
echo ""
echo "=== ROUND 3 EXPERIMENT A: PHASE-LEVEL TIMING SUMMARY ==="
echo "Engine | Parallel | BS | Prompt | Response | PS | step_s | gen_s | old_log_prob_s | ref_s | update_actor_s | update_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy"
echo "------|----------|----|--------|----------|----|--------|------|---------------|------|---------------|----------------|-------------|--------------|----------------|--------"

for test_spec in "${TESTS[@]}"; do
  IFS='|' read -r bs prompt_len response_len max_model_len gpu_mem <<< "$test_spec"

  for ps in 0 1; do
    LOGFILE="${LOG_DIR}/fsdp_7gpu_dp7_qwen3_1.7b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

    if [ -f "$LOGFILE" ]; then
      STEP_TIME=$(grep -oP 'timing_s/step:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      GEN_TIME=$(grep -oP 'timing_s/gen:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      OLD_LOGPROB=$(grep -oP 'timing_s/old_log_prob:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      REF_TIME=$(grep -oP "timing_s/RefPolicy:[0-9.]+" "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_ACTOR=$(grep -oP 'timing_s/update_actor:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_WEIGHTS=$(grep -oP 'timing_s/update_weights:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      ACTOR_HBM=$(grep -oP 'actor/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
      CRITIC_HBM=$(grep -oP 'critic/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
      THROUGHPUT=$(grep -oP 'perf/throughput:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      ENTROPY=$(grep -oP 'actor/entropy:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")

      echo "fsdp | 7gpu_dp7 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | ${STEP_TIME} | ${GEN_TIME} | ${OLD_LOGPROB} | ${REF_TIME} | ${UPDATE_ACTOR} | ${UPDATE_WEIGHTS} | ${ACTOR_HBM} | ${CRITIC_HBM} | ${THROUGHPUT} | ${ENTROPY}"
    else
      echo "fsdp | 7gpu_dp7 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | MISSING | - | - | - | - | - | - | - | - | -"
    fi
  done
done

echo "=== Skipped: ${SKIPPED} ==="
echo "=== END ==="
