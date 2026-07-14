# 实验工具：固定 Rollout Replay

## 目标

为 PrefixSharing 的精度对齐和性能对比提供一个轻量测试工具：首次运行保留真实 rollout 输出；后续运行跳过 vLLM / agent rollout，直接把该输出注入 verl 主流程。这样 PS=OFF 和 PS=ON 的 actor 训练看到相同 response，消除 rollout 随机性，同时继续经过真实的 reward、old logprob、advantage、FSDP forward/backward、PrefixSharing 和 restore 链路。

第一版只服务 `verl_cdd9014f` 的 `RayPPOTrainer + FSDP` 测试验证，不进入 PrefixSharing core，也不改变未显式开启 replay 时的训练行为。

## 最小用法

精度验证只需两次运行，不需要额外执行一次 PS=OFF replay：

```text
Run 1: PS=OFF + 正常 rollout + capture + dump_off
Run 2: PS=ON  + replay  + dump_on
cmp_diag_verl080(dump_on, dump_off)
```

第一轮的真实 rollout 既是 baseline，也顺带生成 replay fixture；第二轮只替换 rollout 结果，后续 actor 训练仍走原生主流程。

```bash
# Run 1：正常生成 response，保存其 rollout 输出，同时产生 baseline dump。
ENABLE_PREFIX_SHARING=0 \
PREFIX_SHARING_ROLLOUT_CAPTURE=/path/replay/rollout.pt \
PREFIX_SHARING_DIAG_DUMP=/path/replay/dump_off \
python3 -m verl.trainer.main_ppo ...

# Run 2：不请求真实 rollout，读取 Run 1 的固定结果，同时产生 PS dump。
ENABLE_PREFIX_SHARING=1 \
PREFIX_SHARING_PATCHSET=verl080_fsdp \
VERL_USE_EXTERNAL_MODULES=prefix_sharing \
PREFIX_SHARING_ROLLOUT_REPLAY=/path/replay/rollout.pt \
PREFIX_SHARING_DIAG_DUMP=/path/replay/dump_on \
python3 -m verl.trainer.main_ppo ...

python3 prefix-sharing/prefix_sharing/tools/cmp_diag_verl080.py \
  --dir-on /path/replay/dump_on \
  --dir-off /path/replay/dump_off \
  --tag train \
  --output /path/replay/precision-report.json
```

两次运行必须使用同一初始 checkpoint、同一训练配置、同一数据顺序和相同的 `rollout.n`。建议先限制为一个训练 step，避免 optimizer 更新引入跨 step 状态差异。

## 设计

### 拦截边界

拦截 `RayPPOTrainer` 中的：

```python
combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
```

该边界是正确的最小边界：

- `combined_gen_batch` 已由真实 dataloader、prompt 和 rollout 配置准备好；
- `combined_gen_output` 是实际 rollout 结果，包含 response 及其附属字段；
- 后续 `batch.union(gen_batch_output)`、reward、old/ref logprob、advantage、actor update 均保持 verl 原生逻辑；
- 不需要序列化或重建 FSDP 内部 micro-batch，也不需要自行调 engine。

capture / replay 只作用于 driver 侧的全局 `DataProto`。后续仍由 verl 原有逻辑进行 DP/FSDP 分发，因此第一版天然覆盖单卡和 2 卡 FSDP。

### Fixture

复用 verl `DataProto.save_to_disk()` / `DataProto.load_from_disk()` 存储 `combined_gen_output`。这是内部测试 fixture，只允许读取本地受信任文件，不面向不可信输入。

同目录额外保存 `manifest.json`：

```json
{
  "schema_version": 1,
  "prefix_sharing_commit": "...",
  "verl_snapshot": "cdd9014f",
  "torch_version": "...",
  "rollout_request_fingerprint": "...",
  "rollout_n": 2,
  "temperature": 1.0,
  "batch_size": 8
}
```

`rollout_request_fingerprint` 由 `combined_gen_batch` 中会影响生成语义的 tensor / meta 字段计算，例如 `input_ids`、attention mask、position ids、temperature、`rollout.n`。不纳入每轮随机生成但不影响 token 语义的 `uid`。Replay 时必须重新计算并比较；不一致则 fail-fast，禁止把错误 fixture 注入另一批 prompt。

capture 文件写入临时路径后原子 rename；目标已存在时默认报错，防止意外覆盖基线。

### 模式与环境变量

| 环境变量 | 行为 |
|---|---|
| 未设置 | 无 replay 行为，完全保持现有训练 |
| `PREFIX_SHARING_ROLLOUT_CAPTURE=/path/rollout.pt` | 调真实 rollout，保存输出与 manifest，然后把原输出继续返回主流程 |
| `PREFIX_SHARING_ROLLOUT_REPLAY=/path/rollout.pt` | 不调真实 rollout，加载保存的输出并返回主流程 |

capture 与 replay 不可同时设置。replay 返回的 `DataProto.meta_info` 中应移除历史 `timing`，并写入 `rollout_replayed=true`，避免把首次 vLLM 生成耗时误记入第二轮性能结果。

计划中的代码注释必须明确两轮高效用法，放在 capture/replay 分支旁：

```python
# A normal PS=OFF run both creates the baseline dump and captures rollout
# output. Replaying it in the PS=ON run avoids a third PS=OFF replay run.
```

## 精度与性能的关系

### 精度对齐

replay 解决“ON/OFF 的 response 不同”问题，但不替代 comparator。精度流程为：

1. Run 1 capture 的真实 rollout 产生 `dump_off`；
2. Run 2 replay 同一输出产生 `dump_on`；
3. `cmp_diag_verl080.py` 先校验 input / mask / label 一致，再比较 RoPE、attention、packed logits、restore 后的 2D logprob / entropy；
4. 后续补充 loss 与关键梯度 dump/比较；
5. 任一关键比较失败时比较器必须非零退出。

`verify_p0_correctness.py` 继续只验证 KV builder 等价性和梯度图，不替代真实 FSDP replay 精度验证。

### 性能对比

使用同一 replay fixture 运行 PS=OFF / PS=ON，计时范围只包含 actor 侧：

- actor forward；
- actor forward + backward；
- 峰值显存。

不将 rollout、Ray/vLLM 初始化或历史 rollout timing 纳入比较。性能模式先 warmup 10 次，再采样 30 次；每次从同一 fixture 复制输入，避免 batch 被下游原地修改。

## 最小实现范围

建议新增：

```text
prefix_sharing/integrations/verl_rollout_replay.py
    RolloutReplayController
    maybe_capture_or_replay_rollout(...)

prefix_sharing/setup/patches/verl080_fsdp/...
    对 RayPPOTrainer 的 rollout 调用做薄 patch
```

`RolloutReplayController` 只负责：环境变量解析、fixture 读写、request fingerprint、元数据校验和返回 replay output。它不理解 PrefixSharing plan、KV store、FSDP micro-batch 或 loss。

第一版不支持：

- Megatron、NPU、TP、PP；
- async / fully-async trainer；
- 多 step optimizer 演化后的严格训练曲线复现；
- 不同模型、不同 `rollout.n` 或不同 prompt batch 间复用 fixture。

## 测试与验收

按 TDD 开发：

1. unit：环境变量互斥、fixture round-trip、manifest 校验、request fingerprint 稳定性与不一致 fail-fast；
2. integration fake：capture 分支只调用一次真实 generator，replay 分支零次调用 generator，返回的 `DataProto` 与 capture 输出等价；
3. optional verl：1 卡 FSDP 下 Run 1 / Run 2 的 dump 中 input ids、attention mask、label mask 完全一致；
4. optional verl：`cmp_diag_verl080.py` 结果为 `all_passed=true` 才标记精度通过；
5. optional 2 卡 FSDP：同一 fixture 正常分发、无 collective 超时、输出比较通过。

完成后，`test_verl080_restore_e2e.py` 应使用 replay fixture 替换当前 TODO/skip 的真实精度用例；设备缺失时允许 skip，具备 verl+GPU 的验证环境不得跳过。

## 开发计划

1. 实现 `RolloutReplayController` 与 unit test，不改训练默认行为。
2. 在 `verl080_fsdp` setup patch 中以最小方式包裹 rollout 调用，补 fake integration test。
3. 用单卡 FSDP capture -> replay 跑通两轮流程，验证 dump 输入一致。
4. 增强 `cmp_diag_verl080.py` 的 input preflight、非零退出、loss/gradient 比较。
5. 将 replay 流程固化到 `test_verl080_restore_e2e.py`，再扩展到 2 卡与性能采样。
