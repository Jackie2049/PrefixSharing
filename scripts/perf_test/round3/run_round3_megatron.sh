#!/bin/bash
# PrefixSharing Round 3 - Megatron with Qwen3-0.6B
# Test Phase-level attribution (Exp A) + Batch-size scaling (Exp B)
# Engine: Megatron 1GPU TP=1 on 4090
# Model: Qwen3-0.6B (16Q/8KV/1024D/28L, ~0.6B params, bf16 ~1.2GB)
# Key differences from FSDP:
#   - algorithm.adv_estimator=gae (not grpo)
#   - actor.megatron.* flags instead of actor.fsdp_config.*
#   - ref.megatron.* flags
#   - model_engine=megatron at end
#   - DDP buffer ~2.4GB but Qwen3-0.6B small enough to fit
#   - No DP divisibility constraint (1GPU)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../perf_results/round3_megatron_logs"
mkdir -p "$LOG_DIR"

PYTHON=/home/zxw/miniconda3/envs/verl080/bin/python3
source ~/miniconda3/bin/activate verl080

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
DATA_DIR="/home/zxw/Termius/proj_prefix-sharing/data"
VERL_DIR="/home/zxw/verldir"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: Model config.json not found at $MODEL_PATH"
  exit 1
fi
echo "Using model: $MODEL_PATH"

###############################################
# Section A: Phase-level attribution (Exp A)
# 7 configs × 2 PS = 14 tests
###############################################
echo ""
echo "============================================="
echo "  SECTION A: PHASE-LEVEL ATTRIBUTION"
echo "============================================="

A_TESTS=(
  "4|256|32|320|0.6"
  "8|256|32|320|0.6"
  "16|256|32|320|0.6"
  "4|512|64|608|0.6"
  "8|512|64|608|0.6"
  "16|512|64|608|0.6"
  "4|1024|64|1088|0.6"
  "8|1024|64|1088|0.6"
)

A_TOTAL=$(( ${#A_TESTS[@]} * 2 ))
A_COUNT=0

for test_spec in "${A_TESTS[@]}"; do
  IFS='|' read -r bs prompt_len response_len max_model_len gpu_mem <<< "$test_spec"

  for ps in 0 1; do
    A_COUNT=$((A_COUNT + 1))
    LOGFILE="${LOG_DIR}/megatron_1gpu_tp1_qwen3_0.6b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

    if [ -f "$LOGFILE" ] && grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
      echo "[SKIP A $A_COUNT/$A_TOTAL] bs=$bs p=$prompt_len r=$response_len PS=$ps"
      continue
    fi

    rm -f "$LOGFILE" 2>/dev/null

    echo "[RUN A $A_COUNT/$A_TOTAL] Megatron 1GPU TP=1 bs=$bs prompt=$prompt_len response=$response_len PS=$ps"

    if [ "$ps" = "1" ]; then
      export ENABLE_PREFIX_SHARING=1
      export VERL_USE_EXTERNAL_MODULES=prefix_sharing
      export PREFIX_SHARING_PATCHSET=verl080_mcore0161_ms0160
    else
      export ENABLE_PREFIX_SHARING=0
      unset VERL_USE_EXTERNAL_MODULES
      unset PREFIX_SHARING_PATCHSET
    fi

    export CUDA_VISIBLE_DEVICES=0

    cd $VERL_DIR

    timeout 600 $PYTHON -m verl.trainer.main_ppo \
      algorithm.adv_estimator=gae \
      data.train_files="['$DATA_DIR/train_ps_prompt${prompt_len}.parquet']" \
      data.val_files="['$DATA_DIR/test_ps_prompt${prompt_len}.parquet']" \
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
      actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
      actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
      actor_rollout_ref.actor.megatron.param_offload=False \
      actor_rollout_ref.actor.megatron.optimizer_offload=False \
      actor_rollout_ref.actor.megatron.grad_offload=False \
      actor_rollout_ref.actor.megatron.use_distributed_optimizer=False \
      actor_rollout_ref.actor.megatron.sequence_parallel=False \
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
      actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
      actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
      actor_rollout_ref.ref.megatron.param_offload=True \
      trainer.balance_batch=True \
      trainer.critic_warmup=0 \
      trainer.logger='["console"]' \
      trainer.project_name=verl_r3m_1gpu_tp1_qwen3_0.6b_ps${ps} \
      trainer.experiment_name=1gpu_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
      trainer.n_gpus_per_node=1 \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      trainer.total_epochs=1 \
      trainer.total_training_steps=1 \
      trainer.val_before_train=False \
      model_engine=megatron \
      2>&1 | tee "$LOGFILE" || true

    if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
      echo "SUCCESS A: $LOGFILE"
    else
      tail -20 "$LOGFILE"
      echo "FAILED A: $LOGFILE"
    fi

    ray stop --force 2>/dev/null || true
    sleep 5
  done
done

###############################################
# Section B: Batch-size scaling (Exp B)
# 3 prompt specs × multiple batch × 2 PS
###############################################
echo ""
echo "============================================="
echo "  SECTION B: BATCH-SIZE SCALING"
echo "============================================="

B_TESTS=(
  "256|32|320|0.7|4,8,16,24,32"
  "512|64|608|0.7|4,8,16,24,32"
  "1024|64|1088|0.7|4,8,16,24"
)

B_TOTAL=0
for spec in "${B_TESTS[@]}"; do
  IFS='|' read -r _ _ _ _ batch_list <<< "$spec"
  for bs in $(echo "$batch_list" | tr ',' ' '); do
    B_TOTAL=$((B_TOTAL + 2))
  done
done

B_COUNT=0

for spec in "${B_TESTS[@]}"; do
  IFS='|' read -r prompt_len response_len max_model_len gpu_mem batch_list <<< "$spec"

  MAX_BATCH_PS0=0
  MAX_BATCH_PS1=0

  for bs in $(echo "$batch_list" | tr ',' ' '); do
    for ps in 0 1; do
      B_COUNT=$((B_COUNT + 1))
      LOGFILE="${LOG_DIR}/megatron_1gpu_tp1_qwen3_0.6b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}_scaling.log"

      if [ -f "$LOGFILE" ] && grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
        echo "[SKIP B $B_COUNT/$B_TOTAL] bs=$bs p=$prompt_len r=$response_len PS=$ps"
        if [ "$ps" = "0" ]; then MAX_BATCH_PS0=$bs; else MAX_BATCH_PS1=$bs; fi
        continue
      fi

      rm -f "$LOGFILE" 2>/dev/null

      echo "[RUN B $B_COUNT/$B_TOTAL] Megatron 1GPU TP=1 bs=$bs prompt=$prompt_len response=$response_len PS=$ps"

      if [ "$ps" = "1" ]; then
        export ENABLE_PREFIX_SHARING=1
        export VERL_USE_EXTERNAL_MODULES=prefix_sharing
        export PREFIX_SHARING_PATCHSET=verl080_mcore0161_ms0160
      else
        export ENABLE_PREFIX_SHARING=0
        unset VERL_USE_EXTERNAL_MODULES
        unset PREFIX_SHARING_PATCHSET
      fi

      export CUDA_VISIBLE_DEVICES=0

      cd $VERL_DIR

      timeout 600 $PYTHON -m verl.trainer.main_ppo \
        algorithm.adv_estimator=gae \
        data.train_files="['$DATA_DIR/train_ps_prompt${prompt_len}.parquet']" \
        data.val_files="['$DATA_DIR/test_ps_prompt${prompt_len}.parquet']" \
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
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
        actor_rollout_ref.actor.megatron.param_offload=False \
        actor_rollout_ref.actor.megatron.optimizer_offload=True \
        actor_rollout_ref.actor.megatron.grad_offload=False \
        actor_rollout_ref.actor.megatron.use_distributed_optimizer=False \
        actor_rollout_ref.actor.megatron.sequence_parallel=False \
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
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
        actor_rollout_ref.ref.megatron.param_offload=True \
        trainer.balance_batch=True \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name=verl_r3b_megatron_1gpu_qwen3_0.6b_ps${ps} \
        trainer.experiment_name=1gpu_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}_scaling \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=3 \
        trainer.val_before_train=False \
        model_engine=megatron \
        2>&1 | tee "$LOGFILE" || true

      if grep -q 'timing_s/step' "$LOGFILE" 2>/dev/null; then
        echo "SUCCESS B: $LOGFILE"
        if [ "$ps" = "0" ]; then MAX_BATCH_PS0=$bs; else MAX_BATCH_PS1=$bs; fi
      else
        echo "FAILED B: $LOGFILE"
      fi

      ray stop --force 2>/dev/null || true
      sleep 5
    done
  done

  echo ""
  echo "=== MAX BATCH SUMMARY: p=$prompt_len r=$response_len ==="
  echo "PS=OFF max_batch=${MAX_BATCH_PS0}, PS=ON max_batch=${MAX_BATCH_PS1}"
  echo ""
done

###############################################
# SUMMARY - Section A
###############################################
echo ""
echo "=== ROUND 3 MEGATRON EXP A: PHASE-LEVEL TIMING SUMMARY ==="
echo "Engine | Parallel | BS | Prompt | Response | PS | step_s | gen_s | old_log_prob_s | ref_s | update_actor_s | update_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy"
echo "------|----------|----|--------|----------|----|--------|------|---------------|------|---------------|----------------|-------------|--------------|----------------|--------"

for test_spec in "${A_TESTS[@]}"; do
  IFS='|' read -r bs prompt_len response_len max_model_len gpu_mem <<< "$test_spec"
  for ps in 0 1; do
    LOGFILE="${LOG_DIR}/megatron_1gpu_tp1_qwen3_0.6b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"
    if [ -f "$LOGFILE" ]; then
      STEP_TIME=$(grep -oP 'timing_s/step:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      GEN_TIME=$(grep -oP 'timing_s/gen:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      OLD_LOGPROB=$(grep -oP 'timing_s/old_log_prob:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      REF_TIME=$(grep -oP 'timing_s/RefPolicy:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_ACTOR=$(grep -oP 'timing_s/update_actor:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_WEIGHTS=$(grep -oP 'timing_s/update_weights:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      ACTOR_HBM=$(grep -oP 'actor/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
      CRITIC_HBM=$(grep -oP 'critic/perf/max_memory_allocated_gb:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
      THROUGHPUT=$(grep -oP 'perf/throughput:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      ENTROPY=$(grep -oP 'actor/entropy:(?:np\.float64\()?[0-9.]+\)?' "$LOGFILE" | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "FAIL")
      echo "megatron | 1gpu_tp1 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | ${STEP_TIME} | ${GEN_TIME} | ${OLD_LOGPROB} | ${REF_TIME} | ${UPDATE_ACTOR} | ${UPDATE_WEIGHTS} | ${ACTOR_HBM} | ${CRITIC_HBM} | ${THROUGHPUT} | ${ENTROPY}"
    else
      echo "megatron | 1gpu_tp1 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | MISSING | - | - | - | - | - | - | - | - | -"
    fi
  done
done

###############################################
# SUMMARY - Section B
###############################################
echo ""
echo "=== ROUND 3 MEGATRON EXP B: BATCH-SCALE TIMING SUMMARY ==="
echo "Engine | Parallel | BS | Prompt | Response | PS | step_s | gen_s | old_log_prob_s | update_actor_s | update_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy | OOM?"
echo "------|----------|----|--------|----------|----|--------|------|---------------|---------------|----------------|-------------|--------------|----------------|--------|-----"

for spec in "${B_TESTS[@]}"; do
  IFS='|' read -r prompt_len response_len max_model_len gpu_mem batch_list <<< "$spec"
  for bs in $(echo "$batch_list" | tr ',' ' '); do
    for ps in 0 1; do
      LOGFILE="${LOG_DIR}/megatron_1gpu_tp1_qwen3_0.6b_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}_scaling.log"
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
          echo "megatron | 1gpu_tp1 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | ${STEP_TIME} | ${GEN_TIME} | ${OLD_LOGPROB} | ${UPDATE_ACTOR} | ${UPDATE_WEIGHTS} | ${ACTOR_HBM} | ${CRITIC_HBM} | ${THROUGHPUT} | ${ENTROPY} | ${IS_OOM}"
        else
          IS_OOM="Y"
          echo "megatron | 1gpu_tp1 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | OOM | - | - | - | - | - | - | - | - | ${IS_OOM}"
        fi
      else
        echo "megatron | 1gpu_tp1 | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | MISSING | - | - | - | - | - | - | - | - | MISS"
      fi
    done
  done
done

echo ""
echo "=== END ==="
