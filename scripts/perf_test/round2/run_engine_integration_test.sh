#!/bin/bash
# PrefixSharing engine integration test sweep
# Covers: Megatron (1GPU/TP=2/TP=8) + FSDP (DP=8)
# Dimensions: batch_size, prompt_len, response_len, PS ON/OFF
# All runs: 1 training step, collect timing/HBM metrics

set -uo pipefail
# Note: removed -e so the sweep continues even if individual runs fail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../perf_results/engine_logs"
mkdir -p "$LOG_DIR"

PYTHON=/home/zxw/miniconda3/envs/verl080/bin/python3
source ~/miniconda3/bin/activate verl080

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

MODEL_QWEN25="/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_QWEN3="/home/zxw/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
DATA_DIR="/home/zxw/Termius/proj_prefix-sharing/data"
VERL_DIR="/home/zxw/verldir"

# Test matrix definition
# Format: engine|parallel|model|batch|prompt|response|max_model_len|gpu_mem|data_file
TESTS=(
  # Megatron 1-GPU, TP=1 (Qwen2.5-0.5B)
  "megatron|1gpu_tp1|qwen25|8|256|32|320|0.6|train_ps_prompt256.parquet"
  "megatron|1gpu_tp1|qwen25|8|512|32|576|0.5|train_ps_prompt512.parquet"
  "megatron|1gpu_tp1|qwen25|8|1024|32|1088|0.6|train_ps_prompt1024.parquet"
  "megatron|1gpu_tp1|qwen25|16|256|32|320|0.6|train_ps_prompt256.parquet"
  "megatron|1gpu_tp1|qwen25|16|512|64|608|0.5|train_ps_prompt512.parquet"
  # Megatron 8-GPU, TP=2, DP=4 (Qwen2.5-0.5B)
  "megatron|8gpu_tp2|qwen25|16|256|32|320|0.6|train_ps_prompt256.parquet"
  "megatron|8gpu_tp2|qwen25|16|512|32|576|0.5|train_ps_prompt512.parquet"
  # Megatron 8-GPU, TP=8 (Qwen3-0.6B - needs 16KV heads for TP=8)
  "megatron|8gpu_tp8|qwen3|16|256|32|320|0.6|train_ps_prompt256.parquet"
  "megatron|8gpu_tp8|qwen3|16|512|32|576|0.5|train_ps_prompt512.parquet"
  # FSDP 8-GPU, DP=8 (Qwen2.5-0.5B)
  "fsdp|8gpu_dp8|qwen25|16|256|32|320|0.6|train_ps_prompt256.parquet"
  "fsdp|8gpu_dp8|qwen25|16|512|64|608|0.5|train_ps_prompt512.parquet"
  "fsdp|8gpu_dp8|qwen25|32|256|32|320|0.6|train_ps_prompt256.parquet"
)

TOTAL=$(( ${#TESTS[@]} * 2 ))  # ×2 for PS ON/OFF
COUNT=0

for test_spec in "${TESTS[@]}"; do
  IFS='|' read -r engine parallel model bs prompt_len response_len max_model_len gpu_mem data_file <<< "$test_spec"

  # Determine model path
  if [ "$model" = "qwen25" ]; then
    MODEL_PATH="$MODEL_QWEN25"
  else
    MODEL_PATH="$MODEL_QWEN3"
  fi

  DATA_PATH="$DATA_DIR/$data_file"
  TEST_DATA="$DATA_DIR/test_ps_prompt${prompt_len}.parquet"

  for ps in 0 1; do
    COUNT=$((COUNT + 1))
    echo "========================================"
    echo "[$COUNT/$TOTAL] engine=$engine parallel=$parallel model=$model bs=$bs prompt=$prompt_len response=$response_len PS=$ps"
    echo "========================================"

    LOGFILE="${LOG_DIR}/${engine}_${parallel}_${model}_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

    # Configure environment based on engine and PS
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

    # Determine GPU allocation
    if [ "$parallel" = "1gpu_tp1" ]; then
      export CUDA_VISIBLE_DEVICES=0
      NGPU=1
      TP_SIZE=1
    elif [ "$parallel" = "8gpu_tp2" ]; then
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      NGPU=8
      TP_SIZE=2
    elif [ "$parallel" = "8gpu_tp8" ]; then
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      NGPU=8
      TP_SIZE=8
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
        actor_rollout_ref.actor.megatron.optimizer_offload=False \
        actor_rollout_ref.actor.megatron.grad_offload=False \
        actor_rollout_ref.actor.megatron.use_distributed_optimizer=False \
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
        trainer.balance_batch=True \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name=verl_${engine}_${parallel}_ps${ps} \
        trainer.experiment_name=${parallel}_${model}_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
        trainer.n_gpus_per_node=$NGPU \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.val_before_train=False \
        model_engine=megatron \
        2>&1 | tee "$LOGFILE"
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
        trainer.project_name=verl_fsdp_${parallel}_ps${ps} \
        trainer.experiment_name=${parallel}_${model}_bs${bs}_p${prompt_len}_r${response_len}_ps${ps} \
        trainer.n_gpus_per_node=$NGPU \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.val_before_train=False \
        2>&1 | tee "$LOGFILE"
    fi

    echo "Done: $LOGFILE"

    # Clean up Ray between runs
    ray stop --force 2>/dev/null || true
    sleep 5
  done
done

# ---- Extract timing summary ----
echo ""
echo "=== ENGINE INTEGRATION TEST TIMING SUMMARY ==="
echo "Engine | Parallel | Model | BS | Prompt | Response | PS | step_time_s | gen_time_s | update_actor_s | update_weights_s | actor_HBM_GB | critic_HBM_GB | throughput_tok/s | entropy"
echo "------|----------|-------|----|--------|----------|----|------------|----------|---------------|----------------|-------------|--------------|----------------|--------"

for test_spec in "${TESTS[@]}"; do
  IFS='|' read -r engine parallel model bs prompt_len response_len max_model_len gpu_mem data_file <<< "$test_spec"

  for ps in 0 1; do
    LOGFILE="${LOG_DIR}/${engine}_${parallel}_${model}_bs${bs}_p${prompt_len}_r${response_len}_ps${ps}.log"

    if [ -f "$LOGFILE" ]; then
      STEP_TIME=$(grep -oP 'timing_s/step:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      GEN_TIME=$(grep -oP 'timing_s/gen:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_ACTOR=$(grep -oP 'timing_s/update_actor:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      UPDATE_WEIGHTS=$(grep -oP 'timing_s/update_weights:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      ACTOR_HBM=$(grep -oP 'actor/perf/max_memory_allocated_gb:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      CRITIC_HBM=$(grep -oP 'critic/perf/max_memory_allocated_gb:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      THROUGHPUT=$(grep -oP 'perf/throughput:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")
      ENTROPY=$(grep -oP 'actor/entropy:[0-9.]+' "$LOGFILE" | tail -1 | cut -d: -f2 || echo "FAIL")

      echo "${engine} | ${parallel} | ${model} | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | ${STEP_TIME} | ${GEN_TIME} | ${UPDATE_ACTOR} | ${UPDATE_WEIGHTS} | ${ACTOR_HBM} | ${CRITIC_HBM} | ${THROUGHPUT} | ${ENTROPY}"
    else
      echo "${engine} | ${parallel} | ${model} | ${bs} | ${prompt_len} | ${response_len} | PS=${ps} | MISSING | - | - | - | - | - | - | -"
    fi
  done
done

echo "=== END ==="
