# 实验工具：固定 Rollout Replay

## 目标

为 PrefixSharing 的精度对齐和性能对比提供一个轻量测试工具：首次运行保留真实 rollout 输出；后续运行跳过 vLLM / agent rollout，直接把该输出注入 verl 主流程。这样 PS=OFF 和 PS=ON 的 actor 训练看到相同 response，消除 rollout 随机性，同时继续经过真实的 reward、old logprob、advantage、FSDP forward/backward、PrefixSharing 和 restore 链路。

它不要求用户事先准备 tokenized JSON 或手工构造 `DataProto`。第一轮正常训练会由 verl 原有 dataloader / tokenizer 从训练数据自动生成 tokenized prompt，并在真实 rollout 后自动生成 replay fixture；capture 本身就是获取固定数据的过程。

第一版只服务 `verl_cdd9014f` 的 `RayPPOTrainer + FSDP` 测试验证，不进入 PrefixSharing core，也不改变未显式开启 replay 时的训练行为。

## 最小用法

精度验证只需两次运行，不需要额外执行一次 PS=OFF replay：

```text
Run 1: PS=OFF + 正常 rollout + capture + dump_off
Run 2: PS=ON  + replay  + dump_on
cmp_diag_verl080(dump_on, dump_off)
```

第一轮的真实 rollout 既是 baseline，也顺带生成 replay fixture；第二轮只替换 rollout 结果，后续 actor 训练仍走原生主流程。用户无需导出、编辑或提前持有任何 tokenized 数据文件。

```bash
# Run 1：正常生成 response，保存其 rollout 输出，同时产生 baseline dump。
ENABLE_PREFIX_SHARING=0 \
PREFIX_SHARING_CAPTURE_ROLLOUT=/path/replay/rollout.json \
PREFIX_SHARING_DIAG_DUMP=/path/replay/dump_off \
python3 -m verl.trainer.main_ppo ...

# Run 2：不请求真实 rollout，读取 Run 1 的固定结果，同时产生 PS dump。
ENABLE_PREFIX_SHARING=1 \
PREFIX_SHARING_PATCHSET=verl080_fsdp \
VERL_USE_EXTERNAL_MODULES=prefix_sharing \
PREFIX_SHARING_FIXED_ROLLOUT=/path/replay/rollout.json \
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

### 数据来源与边界

新 replay 的输入不是外部 JSON，而是首次正常运行自动产出的真实数据：

```text
Run 1: 原始训练数据 -> verl dataloader/tokenizer -> combined_gen_batch
       -> 真实 rollout -> capture combined_gen_output 到 fixture

Run 2: 同一原始训练数据 -> verl dataloader/tokenizer -> combined_gen_batch
       -> 从 fixture 读取相同的 combined_gen_output
       -> 原生 reward / actor 训练
```

因此，replay 阶段仍保留本轮由 dataloader 产生的 `batch` 和 `combined_gen_batch`，只替换 `combined_gen_output`。这是必要的：后续 `batch.repeat(...).union(gen_batch_output)` 仍需使用本轮训练输入及其当前 worker / DP 分发语义。fixture 不承担“替代训练数据集”的职责。

这也明确了第一版边界：它解决“没有预先 tokenized 的固定 rollout 数据”问题，但要求 Run 2 仍可访问与 Run 1 相同的原始训练数据、tokenizer、配置和数据顺序。若未来需要完全脱离原始数据集复现一个 step，应另行设计完整训练 batch capture；那是更重的 micro-batch replay，不属于本工具。

### Fixture

当前实现把 rollout 输出中精度验证需要的 tensor 字段保存为 JSON：`input_ids`、`attention_mask`、`position_ids`、`prompts`、`responses`、`response_mask` 以及可选的 reward / rollout logprob 字段。replay 使用已有 JSON 注入逻辑重建 `DataProto`，不要求预先准备 tokenized 文件。

这是受控的本地实验 fixture，不面向不可信输入。第一版以“同一初始 checkpoint、同一训练配置、同一数据顺序、单个训练 step”为使用前提；不额外引入 manifest 或 request fingerprint。若后续需要跨数据集、跨 step 的严格复现，再单独增加完整 batch capture 和校验机制。

### 模式与环境变量

| 环境变量 | 行为 |
|---|---|
| 未设置 | 无 replay 行为，完全保持现有训练 |
| `PREFIX_SHARING_CAPTURE_ROLLOUT=/path/rollout.json` | 调真实 rollout，保存第一个训练 rollout 输出，然后把原输出继续返回主流程 |
| `PREFIX_SHARING_FIXED_ROLLOUT=/path/rollout.json` | 训练调用不调真实 rollout，加载 JSON 并返回固定输出 |

capture 与 replay 不可同时设置。两种模式均绕过 `meta_info["validate"] == True` 的验证调用；capture 只记录第一个训练 rollout，replay 只替换训练 rollout。replay 返回前会清空 `meta_info["timing"]`，避免把首次 vLLM 生成耗时记入第二轮性能结果。

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

实现收归现有 `prefix_sharing.tools`，不新增 integration 或 core 子模块。它是实验数据工具，不是 PrefixSharing 运行时能力；`core/`、`backends/`、`integrations/` 不应知道 fixture、环境变量或文件格式。

现有 `tools/inject_fixed_rollout.py` 已经会在 `RayPPOTrainer.fit()` 启动时替换 `generate_sequences`，应作为本方案的直接基础，而不是另起一套工具。其当前 JSON 注入与本方案的 capture/replay 服务不同场景：

| 工具模式 | 输入来源 | 适用场景 | 是否保留原始 `DataProto` |
|---|---|---|---|
| 固定 JSON 注入（已有） | 预先 tokenized 的人工 JSON | 构造特定 token / 快速调试 | 否，需重建和 padding |
| synthetic prefix（已有） | 预先 tokenized JSON 中的一条序列 | 人为制造共享前缀的性能实验 | 否，重新构造 batch |
| capture/replay（新增） | 首次正常训练自动捕获的 rollout tensor 字段 | ON/OFF 精度、actor 性能对比 | 否，JSON 重建所需字段 |

三种模式都是“在 rollout 边界提供固定输出”，应共享一个小的控制器和同一个调用点；不能把 JSON 注入伪装成 replay，也不能为了 replay 删除仍有调试价值的 synthetic prefix 工具。

`inject_fixed_rollout.py` 是工具层的唯一入口：

```text
prefix_sharing/tools/inject_fixed_rollout.py
    patch_capture_rollout(...)  # 记录首个训练 rollout
    patch_fixed_rollout(...)    # 注入 JSON 固定输出
    _save_dataproto_to_json(...)

prefix_sharing/setup/patches/verl080_fsdp/rollout_patch.py
    在 RayPPOTrainer.fit() 入口绑定两个 rollout 对象
```

工具模块负责 fixture 读写、训练/验证调用筛选和固定输出注入；setup patch 只读取环境变量并绑定工具函数。两者都不理解 PrefixSharing plan、KV store、FSDP micro-batch 或 loss。

setup patch 在 `RayPPOTrainer.fit()` 入口分别绑定 `actor_rollout_wg` 与 `async_rollout_manager`；实际训练会走自己的原生 `generate_sequences` 调用。工具根据请求的 `meta_info["validate"]` 仅处理训练调用，避免 validation 影响 fixture。工具必须保留下面这段注释，说明两轮精度流程为何只要一次真实 baseline：

```python
# A normal PS=OFF run both creates the baseline dump and captures rollout
# output. Replaying it in the PS=ON run avoids a third PS=OFF replay run.
```

第一版不支持：

- Megatron、NPU、TP、PP；
- async / fully-async trainer；
- 多 step optimizer 演化后的严格训练曲线复现；
- 不同模型、不同 `rollout.n` 或不同 prompt batch 间复用 fixture。

## 测试与验收

按 TDD 开发：

1. unit：环境变量互斥、无需外部 tokenized JSON 的 capture fixture round-trip、capture / replay 跳过 validation；
2. integration fake：capture 分支只调用一次真实 training generator，replay 分支零次调用训练 generator，返回的 `DataProto` 与预期输出等价；
3. optional verl：1 卡 FSDP 下 Run 1 / Run 2 的 dump 中 input ids、attention mask、label mask 完全一致；
4. optional verl：`cmp_diag_verl080.py` 结果为 `all_passed=true` 才标记精度通过；
5. optional 2 卡 FSDP：同一 fixture 正常分发、无 collective 超时、输出比较通过。

完成后，`test_verl080_restore_e2e.py` 应使用 replay fixture 替换当前 TODO/skip 的真实精度用例；设备缺失时允许 skip，具备 verl+GPU 的验证环境不得跳过。

## 开发计划

1. 已完成：在 `tools/inject_fixed_rollout.py` 增加 capture/replay，并通过 setup patch 接入 verl080 FSDP。
2. 已完成：补充 fake unit/integration 覆盖，保护 validation 跳过、capture/replay 时序与环境变量互斥。
3. 用单卡 FSDP capture -> replay 跑通两轮流程，验证 dump 输入一致。
4. 增强 `cmp_diag_verl080.py` 的 input preflight、非零退出、loss/gradient 比较。
5. 将 replay 流程固化到 `test_verl080_restore_e2e.py`，再扩展到 2 卡与性能采样。
