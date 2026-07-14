# Rollout Capture / Replay 使用指南

本工具用于在 verl 训练中捕获一次真实 rollout 轨迹，并在后续训练中重放该轨迹。它的用途是让 PrefixSharing 开关前后的 actor 训练消费相同的 prompt / response 数据，从而消除 rollout 采样差异对精度对齐和性能对比的干扰。

它不需要事先准备 tokenized JSON。第一次正常训练会由 verl 的 dataloader、tokenizer 和 rollout 引擎生成数据，并自动把首个训练 rollout 写为 JSON fixture。

## 适用范围

- 当前支持：`verl_cdd9014f`、`RayPPOTrainer`、FSDP 路径。
- 当前不支持：Megatron、NPU、TP / PP、async / fully-async trainer。
- capture / replay 只处理训练 rollout；`meta_info["validate"] == True` 的 validation rollout 保持原生行为。
- 一个 fixture 对应第一个训练 rollout。精度对齐建议只运行一个 training step；多 step 训练会重复同一份固定轨迹，不代表真实训练曲线。

## 环境变量

| 环境变量 | 含义 |
|---|---|
| `PREFIX_SHARING_CAPTURE_ROLLOUT=/path/rollout.json` | 正常执行第一个训练 rollout，并把输出写入 JSON fixture。 |
| `PREFIX_SHARING_FIXED_ROLLOUT=/path/rollout.json` | 跳过训练 rollout，读取 JSON fixture 并注入其输出。 |
| `ENABLE_PREFIX_SHARING=0` | 关闭 PrefixSharing，作为 baseline。 |
| `ENABLE_PREFIX_SHARING=1` | 开启 PrefixSharing。 |
| `PREFIX_SHARING_DIAG_DUMP=/path/dump` | 保存 ON/OFF 精度诊断数据；性能计时时应关闭。 |
| `PREFIX_SHARING_PATCHSET=verl080_fsdp` | 显式安装 FSDP patch set。 |
| `VERL_USE_EXTERNAL_MODULES=prefix_sharing` | 让 verl 导入 PrefixSharing 外部模块。 |

`PREFIX_SHARING_CAPTURE_ROLLOUT` 和 `PREFIX_SHARING_FIXED_ROLLOUT` 互斥；同时设置会直接报错。

## 运行前检查

两次精度运行必须保持以下内容一致：

- 初始 checkpoint；
- 训练配置、模型和 tokenizer；
- 数据集、数据顺序、shuffle 配置和随机种子；
- `rollout.n`、temperature、batch size 与 sequence length 配置；
- 训练步数，推荐只跑一个 training step。

fixture 只替换 rollout 输出，不替换训练 dataloader 产生的当前 batch。因此 Run 2 仍必须能够读取与 Run 1 相同的原始训练数据。

## 精度对齐：两次运行

精度验证不需要额外执行一次 PS=OFF replay。第一次 PS=OFF 正常 rollout 同时就是 baseline 和 fixture capture；第二次 PS=ON 重放它。

```text
Run 1: PS=OFF + 正常 rollout + capture + dump_off
Run 2: PS=ON  + fixed replay  + dump_on
Compare: cmp_diag_verl080(dump_on, dump_off)
```

先选择一个空的工作目录：

```bash
export REPLAY_DIR=/path/to/replay_case
mkdir -p "$REPLAY_DIR"
```

### Run 1：捕获 PS=OFF baseline

在正常 PPO 启动命令前增加：

```bash
ENABLE_PREFIX_SHARING=0 \
PREFIX_SHARING_PATCHSET=verl080_fsdp \
VERL_USE_EXTERNAL_MODULES=prefix_sharing \
PREFIX_SHARING_CAPTURE_ROLLOUT="$REPLAY_DIR/rollout.json" \
PREFIX_SHARING_DIAG_DUMP="$REPLAY_DIR/dump_off" \
python3 -m verl.trainer.main_ppo ...
```

日志中应出现 `Capture enabled`，并在首个训练 rollout 后出现 `Capture complete`。确认生成了：

```text
$REPLAY_DIR/rollout.json
$REPLAY_DIR/dump_off/
```

如果训练在 validation 后没有进入 training step，不会生成 `rollout.json`；增加一个训练 step 后重新执行。

### Run 2：重放并开启 PrefixSharing

使用与 Run 1 相同的训练命令和配置，只切换 PrefixSharing 与 rollout 数据来源：

```bash
ENABLE_PREFIX_SHARING=1 \
PREFIX_SHARING_PATCHSET=verl080_fsdp \
VERL_USE_EXTERNAL_MODULES=prefix_sharing \
PREFIX_SHARING_FIXED_ROLLOUT="$REPLAY_DIR/rollout.json" \
PREFIX_SHARING_DIAG_DUMP="$REPLAY_DIR/dump_on" \
python3 -m verl.trainer.main_ppo ...
```

日志中应出现 `Replay enabled` 与 `Returning fixed rollout data, skipping generation`。validation 仍可能调用真实 rollout；这是预期行为，固定数据只用于训练 actor 的 rollout。

### 对比诊断结果

```bash
PYTHONPATH=prefix-sharing \
python3 prefix-sharing/prefix_sharing/tools/cmp_diag_verl080.py \
  --dir-on "$REPLAY_DIR/dump_on" \
  --dir-off "$REPLAY_DIR/dump_off" \
  --tag train \
  --output "$REPLAY_DIR/precision-report.json"
```

检查 `precision-report.json` 中的 `all_passed`，并查看任一失败项的最大绝对误差、余弦相似度和对应 dump 文件。replay 只保证训练数据一致，不替代比较器；只有关键 logprob / entropy / logits / attention 等比较满足阈值，才能宣称精度对齐通过。

## 性能对比：使用同一 fixture

性能对比必须排除真实 rollout、模型初始化和诊断 dump 开销。先复用上面的 `rollout.json`，再额外执行两次 replay：

```text
PS=OFF + fixed replay + 仅记录 actor 性能
PS=ON  + fixed replay + 仅记录 actor 性能
```

PS=OFF 的 capture run 含真实 rollout，不能直接与 PS=ON replay run 比较整体 step 时间。两边都使用 `PREFIX_SHARING_FIXED_ROLLOUT`，并关闭 `PREFIX_SHARING_DIAG_DUMP`：

```bash
# Baseline performance run
ENABLE_PREFIX_SHARING=0 \
PREFIX_SHARING_PATCHSET=verl080_fsdp \
VERL_USE_EXTERNAL_MODULES=prefix_sharing \
PREFIX_SHARING_FIXED_ROLLOUT="$REPLAY_DIR/rollout.json" \
python3 -m verl.trainer.main_ppo ...

# PrefixSharing performance run
ENABLE_PREFIX_SHARING=1 \
PREFIX_SHARING_PATCHSET=verl080_fsdp \
VERL_USE_EXTERNAL_MODULES=prefix_sharing \
PREFIX_SHARING_FIXED_ROLLOUT="$REPLAY_DIR/rollout.json" \
python3 -m verl.trainer.main_ppo ...
```

比较 actor forward、forward + backward 和峰值显存。不要将 `gen`、Ray / vLLM 初始化或 validation 时间计入结论。建议先 warmup 10 次，再记录 30 个稳定样本；每次都使用相同 fixture 和相同的设备拓扑。

## Fixture 内容与限制

capture 会写入 rollout 输出中的 `input_ids`、`attention_mask`、`position_ids`、`prompts`、`responses`、`response_mask`，以及存在时的 reward / rollout logprob 字段。replay 再用这些字段重建 `DataProto`。

该 JSON 是受控本地实验文件，不应从不可信来源加载。当前实现不提供跨数据集、跨 checkpoint 或跨多 step 的严格复现保证；这些场景需要完整训练 batch capture 与额外一致性校验。

## 常见问题

**没有生成 fixture**

确认 capture run 至少实际执行了一个训练 step。validation 不会触发 capture。

**Run 2 仍在生成训练 response**

确认 `PREFIX_SHARING_FIXED_ROLLOUT` 指向存在的 JSON 文件，并确认 FSDP patch set 已安装。日志应出现 `Replay enabled`。

**Run 2 的 batch size 或字段冲突**

通常表示两次运行的数据顺序、`rollout.n`、worker 数或训练配置不同。删除旧 fixture，从相同配置的 PS=OFF run 重新 capture。

**精度结果不一致**

先确认两个 dump 都来自 `tag=train`，并核查 checkpoint、数据顺序、随机种子和 fixture 是否相同。replay 后仍有差异时，再按 comparator 报告定位 PrefixSharing 的计算语义。

**想复现多步训练曲线**

当前工具不适合该场景，因为它只保存首个训练 rollout 并在后续训练调用中重复使用。请不要把此模式用于训练收敛性结论。
