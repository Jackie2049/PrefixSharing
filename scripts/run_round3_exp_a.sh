#!/bin/bash
# PrefixSharing Round 3 - Experiment A: Phase-level attribution
# Purpose: Determine if PS=ON slows training-side phases, and where
# Uses existing verl marked_timer: gen, old_log_prob, RefPolicy, update_actor, update_weights
# Model: Qwen3-8B (32Q/4KV/128D, ~8.19B params)
# Engine: Megatron (TP=1/2/4) + FSDP (DP=8)
# Note: TP=8 excluded (4KV heads not divisible by 8)

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

# Model paths - update QWEN3_8B path after download completes
MODEL_QWEN3_8B="${MODEL_QWEN3_8B:-/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/PLACEHOLDER}"
DATA_DIR="/home/zxw/Termius/proj_prefix-sharing/data"
VERL_DIR="/home/zxw/verldir"

# Verify model path exists
if [[ "$MODEL_QWEN3_8B" == *"PLACEHOLDER"* ]] || [ ! -d "$MODEL_QWEN3_8B" ]; then
  echo "ERROR: Qwen3-8B model path not set or does not exist: $MODEL_QWEN3_8B"
  echo "Please set MODEL_QWEN3_8B environment variable or update this script."
  # Try to find the snapshot automatically
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

# Test matrix for Experiment A (phase-level attribution)
# Format: engine|parallel|batch|prompt|response|max_model_len|gpu_mem|optim_offload|distributed_optim
# Key: optimizer_offload needed for TP=1/4 (Adam states 32GB > GPU 24GB)
TESTS=(
  # Megatron 1-GPU TP=1 - must offload optimizer
  "megatron|1gpu_tp1|4|256|32|320|0.85|True|False"
  "megatron|1gpu_tp1|4|512|64|608|0.85|True|False"
  "megatron|1gpu_tp1|8|256|32|320|0.85|True|False"
  # Megatron 8-GPU TP=2 - distributed optimizer across DP=4
  "megatron|8gpu_tp2|8|256|32|320|0.6|False|True"
  "megatron|8gpu_tp2|16|512|64|608|0.5|False|True"
  # Megatron 8-GPU TP=4 - optimizer_offload safer than distributed_optim (DP=2 only)
  "megatron|8gpu_tp4|16|256|32|320|0.5|True|False"
  "megatron|8gpu_tp4|16|512|64|608|0.5|True|False"
  "megatron|8gpu_tp4|16|1024|64|1088|0.5|True|False"
  # FSDP 8-GPU DP=8
  "fsdp|8gpu_dp8|8|256|32|320|0.85|False|False"
  "fsdp|8gpu_dp8|16|512|64|608|0.85|False|False"
  "fsdp|8gpu_dp8|8|1024|64|1088|0.85|False|False"
)

TOTAL=$(( ${#TESTS[@]} * 2 ))
COUNT=0
SKIPPED=0

for test_spec in "${TESTS[@]}"; do
  IFS='|' read -r engine parallel bs prompt_len response_len max_model_len gpu_mem optim_offload distributed_optim <<< "$test_spec"

  MODEL_PATH="$MODEL_QWEN3_8B"
  DATA_PATH="$DATA_DIR/train_ps_prompt${prompt_len}.parquet"
  TEST_DATA="$DATA_DIR/test_ps_prompt${prompt_len}.parquet"

  for ps in 0 1; do
    COUNT=$((COUNT + 1))
    LOGFILE="${LOG_DIR}/${engine}_${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

    # Skip if already completed successfully
    if [ -f "$LOGFILE" ] && grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
      SKIPPED=$((SKIPPED + 1))
      echo "[SKIP $COUNT/$TOTAL] $engine $parallel bs=$bs p=$prompt_len r=$response_len PS=$ps"
      continue
    fi

    # Remove incomplete/failed log
    rm -f "$LOGFILE" 2>/dev/null

    echo "========================================"
    echo "[RUN $COUNT/$TOTAL] engine=$engine parallel=$parallel bs=$bs prompt=$prompt_len response=$response_len PS=$ps"
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
      timeout 600 $PYTHON -m verl.trainer.main_ppo \
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
        trainer.project_name=verl_r3a_${engine}_${parallel}_ps${ps} \
        trainer.experiment_name=${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
        trainer.n_gpus_per_node=$NGPU \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.val_before_train=False \
        model_engine=megatron \
        2>&1 | tee "$LOGFILE" || true
    else
      # FSDP path
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
        trainer.project_name=verl_r3a_fsdp_${parallel}_ps${ps} \
        trainer.experiment_name=${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
        trainer.n_gpus_per_node=$NGPU \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.val_before_train=False \
        2>&1 | tee "$LOGFILE" || true
    fi

    # Check completion
    if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
      echo "SUCCESS: $LOGFILE"
    else
      echo "FAILED: $LOGFILE"
    fi

    # Clean up Ray between runs
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
  IFS='|' read -r engine parallel bs prompt_len response_len max_model_len gpu_mem optim_offload distributed_optim <<< "$test_spec"

  for ps in 0 1; do
    LOGFILE="${LOG_DIR}/${engine}_${parallel}_qwen3_8b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

    if [ -f "$LOGFILE" ]; then
      # Phase-level timing metrics
      STEP_TIME=$(grep -oP 'timing_s/step:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      GEN_TIME=$(grep -oP 'timing_s/gen:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      OLD_LOGPROB=$(grep -oP 'timing_s/old_log_prob:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      REF_TIME=$(grep -oP "timing_s/RefPolicy:[0-9.]+" "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_ACTOR=$(grep -oP 'timing_s/update_actor:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_WEIGHTS=$(grep -oP 'timing_s/update_weights:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      # HBM metrics (handle np.float64() wrapper)
      ACTOR_HBM=$(grep -oP 'actor/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
      CRITIC_HBM=$(grep -oP 'critic/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
      THROUGHPUT=$(grep -oP 'perf/throughput:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      ENTROPY=$(grep -oP 'actor/entropy:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")

      echo "${engine} | ${parallel} | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | ${STEP_TIME} | ${GEN_TIME} | ${OLD_LOGPROB} | ${REF_TIME} | ${UPDATE_ACTOR} | ${UPDATE_WEIGHTS} | ${ACTOR_HBM} | ${CRITIC_HBM} | ${THROUGHPUT} | ${ENTROPY}"
    else
      echo "${engine} | ${parallel} | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | MISSING | - | - | - | - | - | - | - | - | -"
    fi
  done
done

echo "=== Skipped: ${SKIPPED}, Completed: $(ls ${LOG_DIR}/*.log 2>/dev/null | xargs grep -l 'timing_s/step' 2>/dev/null | wc -l), Failed: $(ls ${LOG_DIR}/*.log 2>/dev/null | xargs grep -L 'timing_s/step' 2>/dev/null | wc -l) ==="
echo "=== END ==="
