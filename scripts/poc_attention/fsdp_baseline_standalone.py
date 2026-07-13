#!/usr/bin/env python3
"""verl 0.8.0 + FSDP 单卡正向验证 — PS=OFF 基线.

Simplest possible: load model via HuggingFace + FSDP wrapper, 1 forward+backward.
No Ray, no vLLM, no verl engine — just FSDP+model.
"""
import os, sys, json, time

os.environ["ENABLE_PREFIX_SHARING"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29501"

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/dependency/verl_cdd9014f")

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/jiangdingfeng/zy/Termius/models/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
DEVICE = "cuda"
DTYPE = torch.bfloat16

def main():
    print("=" * 60)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    print(f"verl 0.8.0 + FSDP 单卡基线 (PS=OFF)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: Qwen2.5-0.5B")
    print(f"  Torch: {torch.__version__}")
    print(f"  Rank: {rank}, World: {dist.get_world_size()}")
    print("=" * 60)

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model
    print(f"[  ] Loading model...")
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=DTYPE, trust_remote_code=True
    )
    model.train()
    print(f"[OK] Model loaded in {time.perf_counter() - t0:.1f}s ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # 3. Wrap with FSDP (no auto_wrap_policy, manual wrap for small model)
    print(f"[  ] Wrapping with FSDP...")
    t0 = time.perf_counter()
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=DTYPE, reduce_dtype=DTYPE, buffer_dtype=DTYPE,
        ),
    )
    torch.cuda.synchronize()
    print(f"[OK] FSDP wrapped in {time.perf_counter() - t0:.1f}s")
    print(f"  GPU mem: {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB")
    torch.cuda.reset_peak_memory_stats()

    # 4. Simple forward+backward
    text = "What is 91 + 24?"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    print(f"[  ] Forward pass...")
    t0 = time.perf_counter()
    output = model(input_ids=input_ids, labels=input_ids)
    torch.cuda.synchronize()
    fwd_time = time.perf_counter() - t0

    loss = output.loss
    logits = output.logits

    print(f"[OK] Forward: {fwd_time:.3f}s")
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  Peak GPU: {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB")

    print(f"[  ] Backward pass...")
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = time.perf_counter() - t0
    print(f"[OK] Backward: {bwd_time:.3f}s")

    # 5. Summary
    print()
    print("=" * 60)
    print("FSDP BASELINE PASSED")
    print(f"  Forward:  {fwd_time:.3f}s")
    print(f"  Backward: {bwd_time:.3f}s")
    print(f"  Peak GPU: {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GB")
    print(f"  Loss: {loss.item():.4f}")
    print("=" * 60)

    result = {
        "status": "PASSED",
        "fwd_time_s": round(fwd_time, 3),
        "bwd_time_s": round(bwd_time, 3),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "loss": round(loss.item(), 6),
        "logits_shape": list(logits.shape),
    }
    out_path = os.path.join(os.path.dirname(__file__), "fsdp_baseline_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Result: {out_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
