# GeoAnchor3D 审稿回复实验

本目录保存补充实验入口和分析代码。除两个必要配置钩子外，训练与评估仍复用仓库原入口；服务器运行结果统一写入 `/data/lcx/chat-scene01/outputs/reviewer_response/`。

## 先读：本次代码审计发现

原 `models/chat3d.py` 将 `self.alpha_ablation_mode` 硬编码为 `1`，且模式 1 的注释写“固定 0.5”，实现却填入 `1.0`。这会使默认模型绕过动态 gate。现在已改为配置驱动，并将默认值设为论文 Full IGGA 所需的：

```text
model.gate_granularity=per_head
model.alpha_ablation_mode=0
model.fixed_gate_value=0.5
```

请先确认生成论文主结果的服务器代码是否也存在该硬编码。如果已有 Full checkpoint 是在模式 1 下训练的，它不能被当作 dynamic per-head Full IGGA；需要用修正后的配置重训 Full setting。若服务器训练版本本来就是模式 0，则现有 Full checkpoint 可继续使用。

## 实验清单

| 实验 | 审稿问题 | 计算类型 | 是否重训 7B 主模型 | 入口 |
|---|---|---|---|---|
| Full dynamic per-head | 重建缺失的 GeoAnchor3D Full checkpoint | 完整训练 | **是** | `run_full_per_head.sh` |
| Dynamic scalar gate | per-head 是否优于共享标量 | 完整训练 | **是** | `run_dynamic_scalar.sh` |
| Per-head w/o gate prior | 是否只是 task classifier | 完整训练 | **是** | `run_per_head_no_gate.sh` |
| ScanRefer 内部 gate 分析 | 同一 task 内是否随指令复杂度变化 | 现有 checkpoint 评估+统计 | **否** | `run_within_task_gating.sh` |
| Layer-wise geometry probe | 深层几何遗忘及 GATH 缓解作用 | 冻结主模型+训练小线性头 | **否** | `run_layerwise_geometry_probe.sh` |
| 参数/延迟/显存/训练步成本 | IGGA 推理开销与 GATH 训练开销 | 现有 checkpoint benchmark | **否** | `run_efficiency.sh` |

“训练小线性头”只拟合每层 `hidden state -> (x,y,z)` 的探针，不更新 Chat-Scene 或 GeoAnchor3D。

## 统一准备

所有命令都从仓库根目录运行。三个完整训练 setting 必须使用相同的初始 checkpoint、数据组合、学习率、batch size 和 seed。当前统一使用 Chat-Scene 官方 checkpoint 作为 `INIT_CHECKPOINT` 和 `BASELINE_CHECKPOINT`。

三个完整训练脚本固定采用双卡 DDP：`torchrun --nproc_per_node=2`，两张卡都会参与训练。每卡 batch size 固定为 `8`，双卡全局 batch size 为 `16`。训练入口不再读取外部 `BATCH_SIZE`，避免旧的 `BATCH_SIZE=16` 环境变量导致 OOM。

每个 epoch 完成训练后，Rank 0 会先保存 checkpoint，再开始多任务评估。因此即使评估生成或指标计算失败，该轮权重仍会保留。双卡评估在预测文件合并前和指标计算后显式同步；分布式 collective 超时默认设为 120 分钟，可通过 `TORCH_DISTRIBUTED_TIMEOUT_MINUTES` 覆盖。

服务器路径、双卡设置及已确认的训练/验证任务组合已写入 `lib/common.sh`，仍可通过同名环境变量覆盖。每次登录服务器只需激活环境并进入仓库：

```bash
conda activate chat-scene
cd /data/lcx/chat-scene/Chat-Scene
```

所有正式命令统一通过 `launch_background.sh` 启动。它使用 `nohup + setsid` 脱离 SSH 会话，并把日志和 PID 放入该次实验输出目录。`libtinfo.so.6: no version information available` 是 Conda 动态库警告，不会导致实验退出。

如果论文最终训练组合不同，请以论文实际组合替换 `TRAIN_TAG` 和 `VAL_TAG`。当前仓库 `scripts/run.sh` 的活动配置只包含 `scanrefer#obj_align#nr3d_caption#scanqa`，与注释中的全任务配方不同，不能未经确认就用于审稿表格。

## 1. Full dynamic per-head（需要完整重训）

原 Full checkpoint 暂时无法定位，因此先从 Chat-Scene checkpoint 重训正确的 Full setting：

```bash
bash reviewer_response/launch_background.sh full_per_head
```

关键配置为 `gate_granularity=per_head`、`alpha_ablation_mode=0`、`use_gate_supervision=True`、`gate_loss_weight=1.0`、`coord_loss_weight=0.1`。训练完成后，将最后一个 epoch 的 checkpoint 设置为后续实验使用的 `FULL_CHECKPOINT`。

## 2. Dynamic scalar（需要完整重训）

```bash
bash reviewer_response/launch_background.sh dynamic_scalar
```

关键配置为 `gate_granularity=scalar`、`alpha_ablation_mode=0`、`use_gate_supervision=True`。所有 attention heads 共享一个 instruction-conditioned gate，但其余结构与 Full setting 保持一致。

## 3. Per-head w/o gate prior（需要完整重训）

```bash
bash reviewer_response/launch_background.sh per_head_no_gate
```

关键配置为 `gate_granularity=per_head`、`use_gate_supervision=False`、`gate_loss_weight=0`。LM loss 仍可端到端学习 gate，因此这是“无 task-type prior”，不是“固定 gate”。

## 4. Within-task gating（仅评估）

```bash
bash reviewer_response/launch_background.sh within_task_gating
```

脚本先用 Full checkpoint 评估 ScanRefer，再按 prompt 中显式空间关系短语的数量分为 `0 / 1 / 2+`，输出均值、标准差及 95% CI：

```text
/data/lcx/chat-scene01/outputs/reviewer_response/within_task_gating/<time>/within_task_gating.json
/data/lcx/chat-scene01/outputs/reviewer_response/within_task_gating/<time>/within_task_gating.csv
```

若已有包含 `prompt`、`gate_values` 的合并预测文件，可跳过生成，直接运行：

```bash
python reviewer_response/analyze_within_task_gating.py \
  --predictions /path/to/preds_scanrefer.json \
  --output-json "$REVIEWER_OUTPUT_ROOT/within_task_gating.json" \
  --output-csv "$REVIEWER_OUTPUT_ROOT/within_task_gating.csv"
```

这是一项词表启发式分析。写论文时应公开关系词表，并同时报告每组样本数；不要只挑选满足单调趋势的词或样本。

## 5. Layer-wise geometry probe（不重训主模型）

```bash
bash reviewer_response/launch_background.sh layerwise_geometry_probe
```

默认对第 `4,8,...,32` 层提取对象 token，使用固定 prompt，按 scene 做确定性 train/test 切分，目标是每个场景中心化后的 `(x,y,z)`。输出 baseline 与 Full 的 held-out RMSE、每轴 RMSE、R²：

```text
/data/lcx/chat-scene01/outputs/reviewer_response/layerwise_geometry_probe/<time>/baseline.json
/data/lcx/chat-scene01/outputs/reviewer_response/layerwise_geometry_probe/<time>/geoanchor3d.json
```

先用较小规模检查流程：

```bash
PROBE_MAX_SCENES=10 PROBE_EPOCHS=2 \
  bash reviewer_response/launch_background.sh layerwise_geometry_probe
```

正式结果建议至少 100 个 scene；baseline 和 Full 必须使用同一 scene split、层号、prompt、probe epoch 与 seed。

## 6. Efficiency（仅 benchmark，不更新模型）

```bash
bash reviewer_response/launch_background.sh efficiency
```

每个方法生成两份报告：

- `*_inference.json`：batch 1、beam 1、固定 32 个新 token，CUDA 同步后的模型调用延迟、吞吐和峰值显存。
- `*_training.json`：同一 ScanRefer batch 的 forward+backward 时间与峰值显存；不执行 optimizer step，也不保存更新。

正式比较必须在同一 GPU、相同 attention kernel、dtype、batch、序列设置和空闲机器状态下连续运行。GATH 的“零推理开销”应与整个 GeoAnchor3D 的开销分开表述：GATH 推理时不执行，但 IGGA 仍执行。

## 暂不主动实现的审稿要求

- **Proposal 数量/阈值**：100 proposals 是继承自 Chat-Scene 的受控上游配置，不是 GeoAnchor3D 新引入的超参数。建议在限制中承认 proposal quality 上界；若 AE 明确要求数值结果，再补 top-K/threshold 评估。
- **Handcrafted vs learned geometry**：会新增至少一个完整训练 setting，但对核心 per-head/GATH 因果链的解释力低于 P0 项。本轮先澄清 handcrafted descriptor 本身不是创新点。
- **三随机种子**：成本高。若 AE 要求统计显著性，应对 baseline、Full 和核心两个消融使用相同 seeds 重训，而不是只给 Full 补 seeds。
- **第二 LLM、13B、held-out task、跨域数据**：超出 mandatory minor 的合理范围。建议缩窄 backbone-agnostic、unseen-task 和 domain-generalization 措辞。
- **Proposal perturbation**：若后续必须补，优先做 checkpoint-only 的 top-K/box-noise 受控评估，并同时报告 target proposal recall，避免把 detector recall 下降误归因于 IGGA。

## 结果解释底线

1. Dynamic scalar 与 Full per-head 必须从相同初始化训练。
2. w/o gate prior 的 gate 仍是 instruction-conditioned，只移除 task-type loss。
3. Within-task 分析只证明 gate 与显式关系复杂度相关，不等价于因果证明。
4. Probe 必须按 scene 切分，不能把同一 scene 的对象/重复 prompt 分到训练和测试两侧。
5. 如果 checkpoint 加载报告出现大量 `missing_keys`、`shape_mismatches`，该次结果不可用于论文。
