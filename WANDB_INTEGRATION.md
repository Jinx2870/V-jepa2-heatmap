# Wandb 集成说明

本文档说明如何在 V-JEPA 训练中使用 Weights & Biases (wandb) 进行实时可视化监控。

## 功能特性

✅ **已集成模块**：
- `evals/action_anticipation_frozen/eval.py` - 动作预判探针训练

✅ **记录的指标**：
- 训练/验证准确率（action/verb/noun）
- 训练/验证召回率（action/verb/noun）
- Epoch 级别的汇总指标
- 完整的超参数配置

✅ **特点**：
- 可选启用，默认关闭
- 与原有 CSV 日志并存
- 仅在 rank=0 时记录，避免分布式冲突
- 自动容错，wandb 失败不影响训练

---

## 快速开始

### 1. 安装 wandb（如果尚未安装）
```bash
pip install wandb
```

### 2. 登录 wandb
首次使用需要登录（会提示输入 API key）：
```bash
wandb login
```

或者直接设置环境变量：
```bash
export WANDB_API_KEY=your_api_key_here
```

### 3. 修改配置文件启用 wandb

编辑您的训练配置文件（例如 `configs/eval/vitl/ego4d_keystep.yaml`），在顶部添加：

```yaml
eval_name: action_anticipation_frozen

# Wandb configuration
use_wandb: true  # 启用 wandb
wandb_project: vjepa-action-anticipation  # 项目名称
wandb_name: ego4d-keystep-vitl-run1  # 本次运行的名称

experiment:
  # ... 其余配置不变 ...
```

### 4. 运行训练
```bash
python -m evals.main \
  --fname configs/eval/vitl/ego4d_keystep.yaml \
  --devices cuda:0
```

### 5. 查看训练曲线
训练开始后，终端会输出 wandb 链接，例如：
```
wandb: 🚀 View run at https://wandb.ai/your-username/vjepa-action-anticipation/runs/xxxxx
```

点击链接即可在浏览器中实时查看：
- 📈 训练/验证曲线
- ⚙️ 超参数配置
- 📊 系统资源监控（GPU/CPU/内存）

---

## 配置参数说明

### 必需参数
- **`use_wandb`** (bool): 是否启用 wandb，默认 `false`
  
### 可选参数
- **`wandb_project`** (str): wandb 项目名称，默认 `"vjepa-action-anticipation"`
- **`wandb_name`** (str): 本次运行的名称，默认使用 `tag` 字段值

### 示例配置
```yaml
# 最小配置 - 使用默认项目名
use_wandb: true

# 完整配置 - 自定义项目和运行名称
use_wandb: true
wandb_project: my-custom-project
wandb_name: experiment-v1-lr1e-4
```

---

## 记录的指标说明

### EGO4D KeyStep（单动作任务）
- `epoch`: 当前 epoch 数
- `train/action_accuracy`: 训练集动作准确率
- `train/action_recall`: 训练集动作平均召回率
- `val/action_accuracy`: 验证集动作准确率
- `val/action_recall`: 验证集动作平均召回率

### Epic-Kitchens（verb+noun 任务）
- `epoch`: 当前 epoch 数
- `train/action_accuracy`: 训练集联合动作准确率
- `train/verb_accuracy`: 训练集动词准确率
- `train/noun_accuracy`: 训练集名词准确率
- `train/action_recall`: 训练集联合动作召回率
- `train/verb_recall`: 训练集动词召回率
- `train/noun_recall`: 训练集名词召回率
- `val/*`: 对应的验证集指标

### 超参数配置（自动记录）
- dataset, frames_per_clip, frames_per_second
- resolution, batch_size, num_epochs
- use_bfloat16, use_focal_loss
- num_probe_blocks, num_heads
- train_anticipation_time_sec, val_anticipation_time_sec

---

## 离线模式（无网络环境）

如果在没有网络的环境中训练，可以启用离线模式：

```bash
export WANDB_MODE=offline
python -m evals.main --fname configs/eval/vitl/ego4d_keystep.yaml --devices cuda:0
```

训练完成后，可以在有网络的机器上同步日志：
```bash
wandb sync /path/to/wandb/run/folder
```

---

## 禁用 wandb

如果不想使用 wandb，只需确保配置文件中：
```yaml
use_wandb: false  # 或者直接删除这一行
```

训练会正常进行，只使用 CSV 日志。

---

## 常见问题

### Q1: wandb 初始化失败怎么办？
**A**: 代码已做容错处理，wandb 失败不会中断训练，会自动回退到仅使用 CSV 日志。终端会显示警告信息。

### Q2: 分布式训练时会重复记录吗？
**A**: 不会。代码确保只有 rank=0 的进程记录到 wandb。

### Q3: CSV 日志还会保存吗？
**A**: 会。wandb 和 CSV 日志是并行的，互不影响。CSV 文件仍然保存在：
```
<folder>/<eval_name>/<tag>/log_r0.csv
```

### Q4: 如何对比多次实验？
**A**: 在 wandb 项目页面，可以：
- 选择多个 run 进行曲线对比
- 使用表格视图对比超参数和最终指标
- 导出数据到 CSV

### Q5: 如何设置不同的 wandb 账户/团队？
**A**: 在配置文件中添加：
```yaml
use_wandb: true
wandb_entity: your-team-name  # wandb 团队名称
wandb_project: project-name
```

---

## 技术细节

### 实现位置
- **工具类**: `vjepa2/src/utils/logging.py` - `WandbLogger`
- **集成点**: `vjepa2/evals/action_anticipation_frozen/eval.py` - 第 162-184 行（初始化），第 422-437 行（记录）

### 日志时机
- **Epoch 结束后记录一次**，包含整个 epoch 的汇总指标
- 不记录 iteration 级别的指标（避免日志过多）

### 容错机制
1. wandb 未安装 → 自动禁用，显示警告
2. wandb 初始化失败 → 回退到仅 CSV 日志
3. wandb 记录失败 → 捕获异常，不中断训练

---

## 未来扩展

如需在其他训练脚本中集成 wandb，参考以下步骤：

1. 导入 WandbLogger:
```python
from src.utils.logging import WandbLogger
```

2. 初始化（在 main 函数中）:
```python
wandb_logger = WandbLogger(
    project="your-project",
    name="run-name",
    config={"lr": 1e-4, "batch_size": 32},
    enabled=use_wandb,
)
```

3. 记录指标（在训练循环中）:
```python
wandb_logger.log({
    "train/loss": loss.item(),
    "train/acc": accuracy,
}, step=epoch)
```

4. 结束时清理:
```python
wandb_logger.finish()
```

---

## 更多资源

- [Wandb 官方文档](https://docs.wandb.ai/)
- [Wandb Python API](https://docs.wandb.ai/ref/python)
- [Wandb 最佳实践](https://docs.wandb.ai/guides/track)

---

**注意**: 请确保不要将包含敏感信息的配置文件上传到公共 wandb 项目中。

