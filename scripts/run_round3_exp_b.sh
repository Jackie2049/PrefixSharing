#!/bin/bash
# PrefixSharing Round 3 - Experiment B: Batch-size scaling / HBM capacity
# Purpose: Verify PS=ON reduced HBM enables larger batch, improving effective throughput
# Model: Qwen3-8B (32Q/4KV/128D, ~8.19B params)
# Runs 3 training steps per config, discard 1st warmup
# Note: TP=8 excluded (4KV heads not divisible by 8)

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

# Model paths - update QWEN3_8B path after download completes
MODEL_QWEN3_8B="${MODEL_QWEN3_8B:-/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/PLACEHOLDER}"
DATA_DIR="/home/zxw/Termius/proj_prefix-sharing/data"
VERL_DIR="/home/zxw/verldir"

# Verify model path exists
if [[ "$MODEL_QWEN3_8B" == *"PLACEHOLDER"* ]] || [ ! -d "$MODEL_QWEN3_8B" ]; then
  echo "ERROR: Qwen3-8B model path not set or does not exist: $MODEL_QWEN3_8B"
  FOUND=$(find /home/zxw/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/ -maxdepth 1 -type d 2>/dev/null | head -1)
  if [ -n "$FOUND" ]; then
    MODEL_QWEN3_8B="$FOUND"
    echo "Auto-detected: $MODEL_QWEN3_8B"
  else
    echo "Model not found. Exiting."
    exit 1
  fi
fi
echo "Using model: $MODEL_QWEN3_8B"

# Batch-size scaling matrix
# Format: engine|parallel|prompt|response|max_model_len|gpu_mem|optim_offload|distributed_optim|batch_list
# We sweep batch sizes to find max without OOM for both PS=ON and PS=OFF
SCALING_TESTS=(
  # Megatron 1GPU TP=1 - optimizer_offload mandatory, small batches only
  "megatron|1gpu_tp1|256|32|320|0.85|True|False|2,4,8,12,16"
  # Megatron 8GPU TP=2 - distributed optimizer, moderate batches
  "megatron|8gpu_tp2|512|64|608|0.5|False|True|8,16,24,32"
  # Megatron 8GPU TP=4 - optimizer_offload for safety, larger batches possible
  "megatron|8gpu_tp4|512|64|608|0.5|True|False|16,32,48,64"
  # FSDP 8GPU DP=8 - rollout constrained (full model on each GPU for inference)
  "fsdp|8gpu_dp8|512|64|608|0.85|False|False|4,8,16,24,32"
)

TOTAL_TESTS=0
for spec in "${SCALING_TESTS[@]}"; do
  IFS='|' read -r _ _ _ _ _ _ _ _ batch_list <<< "$spec"
  for bs in $(echo "$batch_list" | tr ',' ' '); do
    TOTAL_TESTS=$((TOTAL_TESTS + 2))  # PS=0 and PS=1
  done
done

COUNT=0
SKIPPED=0

for spec in "${SCALING_TESTS[@]}"; do
  IFS='|' read -r engine parallel prompt_len response_len max_model_len gpu_mem optim_offload distributed_optim batch_list <<< "$spec"

  MODEL_PATH="$MODEL_QWEN3_8B"
  DATA_PATH="$DATA_DIR/train_ps_prompt${prompt_len}.parquet"
  TEST_DATA="$DATA_DIR/test_ps_prompt${prompt_len}.parquet"

  # Track max successful batch for each PS setting
  MAX_BATCH_PS0=0
  MAX_BATCH_PS1=0

  for bs in $(echo "$batch_list" | tr ',' ' '); do
    for ps in 0 1; do
      COUNT=$((COUNT + 1))
      LOGFILE="${LOG_DIR}/${engine}_${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

      # Skip if already completed successfully
      if [ -f "$LOGFILE" ] && grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
        SKIPPED=$((SKIPPED + 1))
        echo "[SKIP $COUNT/$TOTAL_TESTS] $engine $parallel bs=$bs p=$prompt_len r=$response_len PS=$ps (already completed)"
        # Track max batch
        if [ "$ps" = "0" ]; then MAX_BATCH_PS0=$bs; else MAX_BATCH_PS1=$bs; fi
        continue
      fi

      # Remove incomplete/failed log
      rm -f "$LOGFILE" 2>/dev/null

      echo "========================================"
      echo "[RUN $COUNT/$TOTAL_TESTS] engine=$engine parallel=$parallel bs=$bs prompt=$prompt_len response=$response_len PS=$ps"
      echo "========================================"

      # Configure PS environment
      if [ "$ps" = "1" ]; then
        export ENABLE_PREFIX_SHARING=1
        export VERL_USE_EXTERNAL_MODULES=prefix_sharing
        if [ "$engine" = "megatron" ]; then
          export PREFIX_SHARING_PATCHSET=verl080_mcore0161_ms0160
        else
          export PREFIX_SHARING_PATCHSET=verl080_fsdp
        fi
      else
        export ENABLE_PREFIX_SHARING=0
        unset VERL_USE_EXTERNAL_MODULES
        unset PREFIX_SHARING_PATCHSET
      fi

      # Determine GPU allocation and TP size
      if [ "$parallel" = "1gpu_tp1" ]; then
        export CUDA_VISIBLE_DEVICES=0
        NGPU=1
        TP_SIZE=1
      elif [ "$parallel" = "8gpu_tp2" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NGPU=8
        TP_SIZE=2
      elif [ "$parallel" = "8gpu_tp4" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NGPU=8
        TP_SIZE=4
      elif [ "$parallel" = "8gpu_dp8" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NGPU=8
        TP_SIZE=1
      fi

      cd $VERL_DIR

      if [ "$engine" = "megatron" ]; then
        timeout 900 $PYTHON -m verl.trainer.main_ppo \
          algorithm.adv_estimator=gae \
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
          actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
          actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
          actor_rollout_ref.actor.megatron.param_offload=False \
          actor_rollout_ref.actor.megatron.optimizer_offload=$optim_offload \
          actor_rollout_ref.actor.megatron.grad_offload=$optim_offload \
          actor_rollout_ref.actor.megatron.use_distributed_optimizer=$distributed_optim \
          actor_rollout_ref.actor.megatron.sequence_parallel=False \
          actor_rollout_ref.rollout.name=vllm \
          actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
          actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem \
          actor_rollout_ref.rollout.max_num_seqs=$bs \
          actor_rollout_ref.rollout.max_model_len=$max_model_len \
          actor_rollout_ref.rollout.n=1 \
          actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
          actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
          actor_rollout_ref.rollout.agent.num_workers=1 \
          actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
          actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
          actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
          actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
          actor_rollout_ref.ref.megatron.param_offload=True \
          actor_rollout_ref.ref.megatron.optimizer_offload=True \
          actor_rollout_ref.ref.megatron.grad_offload=True \
          trainer.balance_batch=True \
          trainer.critic_warmup=0 \
          trainer.logger='["console"]' \
          trainer.project_name=verl_r3b_${engine}_${parallel}_ps${ps} \
          trainer.experiment_name=${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
          trainer.n_gpus_per_node=$NGPU \
          trainer.nnodes=1 \
          trainer.save_freq=-1 \
          trainer.test_freq=-1 \
          trainer.total_epochs=1 \
          trainer.total_training_steps=3 \
          trainer.val_before_train=False \
          model_engine=megatron \
          2>&1 | tee "$LOGFILE" || true
      else
        # FSDP path
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
          actor_rollout_ref.rollout.n=2 \
          actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
          actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
          actor_rollout_ref.rollout.agent.num_workers=1 \
          actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
          actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
          actor_rollout_ref.ref.fsdp_config.param_offload=False \
          trainer.balance_batch=True \
          trainer.critic_warmup=0 \
          trainer.logger='["console"]' \
          trainer.project_name=verl_r3b_fsdp_${parallel}_ps${ps} \
          trainer.experiment_name=${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
          trainer.n_gpus_per_node=$NGPU \
          trainer.nnodes=1 \
          trainer.save_freq=-1 \
          trainer.test_freq=-1 \
          trainer.total_epochs=1 \
          trainer.total_training_steps=3 \
          trainer.val_before_train=False \
          2>&1 | tee "$LOGFILE" || true
      fi

      # Check completion
      if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
        echo "SUCCESS: $LOGFILE"
        # Track max batch
        if [ "$ps" = "0" ]; then MAX_BATCH_PS0=$bs; else MAX_BATCH_PS1=$bs; fi
      else
        echo "FAILED/OOM: $LOGFILE"
      fi

      # Clean up Ray between runs
      ray stop --force 2>/dev/null || true
      sleep 5
    done
  done

  # Report max batch for this scaling scenario
  echo ""
  echo "=== MAX BATCH SUMMARY: $engine $parallel p=$prompt_len r=$response_len ==="
  echo "PS=OFF max_batch=${MAX_BATCH_PS0}, PS=ON max_batch=${MAX_BATCH_PS1}"
  echo ""
done

# ---- Extract detailed timing summary ----
echo ""
echo "=== ROUND 3 EXPERIMENT B: BATCH-SCALE TIMING SUMMARY ==="
echo "Engine | Parallel | BS | Prompt | Response | PS | step_s | gen_s | old_log_prob_s | update_actor_s | update_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy | OOM?"
echo "------|----------|----|--------|----------|----|--------|------|---------------|---------------|----------------|-------------|--------------|----------------|--------|-----"

for spec in "${SCALING_TESTS[@]}"; do
  IFS='|' read -r engine parallel prompt_len response_len max_model_len gpu_mem optim_offload distributed_optim batch_list <<< "$spec"

  for bs in $(echo "$batch_list" | tr ',' ' '); do
    for ps in 0 1; do
      LOGFILE="${LOG_DIR}/${engine}_${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

      IS_OOM="N"
      if [ -f "$LOGFILE" ]; then
        if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
          # Extract metrics from last step (discard warmup)
          STEP_TIME=$(grep -oP 'timing_s/step:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          GEN_TIME=$(grep -oP 'timing_s/gen:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          OLD_LOGPROB=$(grep -oP 'timing_s/old_log_prob:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          UPDATE_ACTOR=$(grep -oP 'timing_s/update_actor:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          UPDATE_WEIGHTS=$(grep -oP 'timing_s/update_weights:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          ACTOR_HBM=$(grep -oP 'actor/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
          CRITIC_HBM=$(grep -oP 'critic/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
          THROUGHPUT=$(grep -oP 'perf/throughput:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
          ENTROPY=$(grep -oP 'actor/entropy:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
          echo "${engine} | ${parallel} | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | ${STEP_TIME} | ${GEN_TIME} | ${OLD_LOGPROB} | ${UPDATE_ACTOR} | ${UPDATE_WEIGHTS} | ${ACTOR_HBM} | ${CRITIC_HBM} | ${THROUGHPUT} | ${ENTROPY} | ${IS_OOM}"
        else
          IS_OOM="Y"
          echo "${engine} | ${parallel} | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | OOM | - | - | - | - | - | - | - | - | ${IS_OOM}"
        fi
      else
        IS_OOM="MISS"
        echo "${engine} | ${parallel} | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | MISSING | - | - | - | - | - | - | - | - | ${IS_OOM}"
      fi
    done
  done
done

echo "=== Skipped: ${SKIPPED} ==="
echo "=== END ==="
