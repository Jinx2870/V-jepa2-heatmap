# V-JEPA2 Gaze Heatmap Project

本仓库是在 Meta V-JEPA2 代码基础上扩展的 gaze-conditioned video world model 项目。目标是在 Ego4D / Ego-exo4D 视频上，将 V-JEPA2 的视频表征与 Gazelle 提取的 gaze / scene tokens 结合，用自监督 JEPA 目标学习 gaze 条件下的未来视频 latent，并进一步用于 Ego4D Keystep Action Anticipation 下游任务。

这不是 V-JEPA2 官方 README。原始 V-JEPA2 相关实现仍保留在仓库中，本 README 只说明本项目新增的 gaze 训练与下游评估流程。

## 项目目标

- 使用 V-JEPA2 encoder 作为视频 backbone。
- 使用 Gazelle 从 RGB 帧在线提取 scene / gaze tokens，不依赖 dataloader 读取 gaze npy。
- 冻结视频 encoder，训练 gaze-conditioned causal predictor 与 `scene_proj`。
- 在 Ego4D Keystep 上加载 gaze-conditioned checkpoint，冻结 backbone，只训练 action anticipation probe。

## 核心流程

```text
Stage 1: Ego-centric self-supervised adaptation (optional)
  输入: Ego4D / Ego-exo4D 视频帧
  训练: V-JEPA2 encoder + predictor
  入口: app/main.py -> app/vjepa_ego4d/train.py

Stage 2: Gaze-conditioned JEPA training
  输入: 视频帧 + Gazelle 在线提取的 scene tokens
  训练: predictor_causal + scene_proj，encoder 冻结
  入口: app/vjepa_ego4d/train_gaze_unified.py

Downstream: Ego4D Keystep Action Anticipation
  输入: 视频 clip + anticipation time
  训练: AttentivePooler + 分类头，预训练 backbone 冻结
  入口: evals/main.py
```

更完整的端到端说明见 `docs/PIPELINE.md`。

## 主要入口函数

### 1. Gaze-conditioned 训练入口

推荐入口：

```bash
python app/vjepa_ego4d/train_gaze_unified.py \
  --fname configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml
```

对应代码：

- 文件：`app/vjepa_ego4d/train_gaze_unified.py`
- CLI 参数解析：`_parse_args()`
- 主函数：`main(args: Dict[str, Any])`
- 配置字段：`--fname <yaml>`

该脚本会读取 YAML 配置，初始化分布式环境、视频 dataloader、V-JEPA2 encoder、Gazelle、causal predictor、优化器和训练循环。

也可以通过通用 launcher 启动：

```bash
python app/main.py \
  --fname configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml \
  --devices cuda:0 cuda:1
```

此时配置文件里的 `app: vjepa_ego4d_gaze_unified` 会通过 `app/scaffold.py` 调到：

- 文件：`app/vjepa_ego4d_gaze_unified/train.py`
- 函数：`main(args, resume_preempt=False)`
- 实际转发到：`app.vjepa_ego4d.train_gaze_unified.main`

### 2. Downstream 评估 / 训练入口

```bash
python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml
```

可临时覆盖 checkpoint 和输出目录：

```bash
python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml \
  --checkpoint /path/to/stage2/e800.pt \
  --folder /path/to/eval_output \
  --override_config_folder
```

对应代码：

- 文件：`evals/main.py`
- CLI 参数解析：`parser`
- 单进程 / 多进程入口：`process_main(...)`
- 下游任务分发：`evals/scaffold.py`
- Gaze 版 action anticipation 模型：`evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar_gazelle.py`

### 3. 可选 Stage 1 自监督适配入口

如果需要先在 ego-centric 视频上适配 V-JEPA2 encoder，可使用原有 V-JEPA2 app launcher：

```bash
python app/main.py --fname configs/train/vitg16/ego4d-stage1-256px-16f.yaml
```

对应代码：

- 通用入口：`app/main.py`
- app 分发：`app/scaffold.py`
- 训练实现：`app/vjepa_ego4d/train.py`

当前仓库没有专门提交 Stage 1 的 Ego4D yaml，可以复制已有 V-JEPA2 train config 后，将 `app`、`data.datasets`、`folder` 和 `meta.pretrain_checkpoint` 等字段改成本地路径。

## 关键代码结构

```text
app/
  main.py                                  # 通用训练 launcher
  vjepa_ego4d/
    train.py                              # Ego4D 自监督适配训练
    train_gaze_unified.py                 # 本项目 Stage 2 gaze-conditioned 主入口
    ego4d.py                              # Ego4D / gaze video dataloader
  vjepa_ego4d_gaze_unified/
    train.py                              # app/main.py 兼容转发入口

src/
  models/
    vjepa_gazelle_unified.py              # V-JEPA2 + Gazelle unified model
    vision_transformer.py                 # V-JEPA2 encoder
    predictor.py                          # 原始 predictor 相关模块
  utils/
    checkpoint_loader.py                  # checkpoint 加载工具
    distributed.py                        # 分布式初始化
    logging.py                            # 日志 / wandb

configs/
  train/vitg16/
    ego4d-256px-16f-gazelle224-grid16.yaml
    ego4d-256px-64f-gazelle224-grid16.yaml
  eval/vitl/
    ego4d_keystep_gazelle_actionpred.yaml
    ego4d_keystep.yaml

evals/
  main.py                                  # 下游 eval launcher
  action_anticipation_frozen/              # Ego4D Keystep action anticipation

docs/
  PIPELINE.md                              # 更详细的三阶段 pipeline 说明
```

## 环境安装

建议使用 Python 3.11 及以上，并在仓库根目录安装依赖：

```bash
cd /data3/lg2/human_wm/V-jepa2-heatmap-new
pip install -r requirements.txt
pip install -e .
```

本项目还依赖外部 Gazelle 仓库和权重。配置文件中需要设置：

```yaml
data:
  gazelle:
    python_path: /path/to/gazelle
    checkpoint: /path/to/gazelle/checkpoints/gazelle_dinov2_vitb14_inout.pt
    model_name: gazelle_dinov2_vitb14_inout
```

如果 Gazelle 不在默认 Python 搜索路径中，必须保证 `python_path` 指向 Gazelle repo 根目录。

## 数据准备

Stage 2 训练需要一个视频列表文件，每行是一个 mp4 的绝对路径，例如：

```text
/data3/lg2/human_wm/Ego4d/exo_data/takes/<take_id>/cam01.mp4
/data3/lg2/human_wm/Ego4d/exo_data/takes/<take_id>/cam02.mp4
```

默认配置中使用：

```yaml
data:
  datasets:
    - /data3/lg2/human_wm/Ego4d/test/output2.txt
```

Downstream Ego4D Keystep 还需要动作标注：

```yaml
experiment:
  data:
    base_path: /data3/lg2/human_wm/Ego4d/exo_data/takes
    dataset_train: /data3/lg2/human_wm/Ego4d/test/annotations/keystep_train.json
    dataset_val: /data3/lg2/human_wm/Ego4d/test/annotations/keystep_val.json
    video_list_file: /data3/lg2/human_wm/Ego4d/test/output2.txt
```

## 常用配置

### Stage 2 gaze-conditioned 训练

主配置：

```text
configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml
```

重要字段：

```yaml
app: vjepa_ego4d_gaze_unified
folder: /path/to/output/logs

data:
  condition_mode: gazelle
  crop_size: 256
  patch_size: 16
  tubelet_size: 2
  datasets:
    - /path/to/video_list.txt
  gazelle:
    input_size: 224
    python_path: /path/to/gazelle
    checkpoint: /path/to/gazelle_dinov2_vitb14_inout.pt

meta:
  pretrain_checkpoint: /path/to/vitl.pt_or_stage1_latest.pt
  context_encoder_key: target_encoder
  load_predictor: false

model:
  condition_mode: gazelle
  model_name: vit_large
  pred_depth: 12
  pred_embed_dim: 384
```

### Downstream action anticipation

主配置：

```text
configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml
```

重要字段：

```yaml
model_kwargs:
  checkpoint: /path/to/stage2/e800.pt
  module_name: evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar_gazelle
  pretrain_kwargs:
    encoder: {model_name: vit_large, checkpoint_key: encoder}
    predictor: {model_name: vit_predictor, checkpoint_key: predictor_causal}
    scene_proj_checkpoint_key: scene_proj
```

`checkpoint` 必须指向 Stage 2 训练保存出的 gaze-conditioned checkpoint。

## Checkpoint 约定

Stage 2 训练保存的 checkpoint 包含以下关键字段：

| Key | 含义 | 下游是否使用 |
| --- | --- | --- |
| `encoder` | 冻结的 V-JEPA2 video encoder | 是 |
| `predictor_causal` | gaze-conditioned causal predictor | 是 |
| `scene_proj` | Gazelle scene token 到 predictor 空间的投影层 | 是 |
| `opt` | optimizer 状态 | 训练恢复使用 |
| `scaler` | mixed precision scaler 状态 | 训练恢复使用 |
| `config` | 当次训练配置 | 复现实验使用 |

下游配置中的 checkpoint key 必须和这些保存名一致，尤其是：

- `predictor.checkpoint_key: predictor_causal`
- `scene_proj_checkpoint_key: scene_proj`

## 快速开始

### 1. 进入仓库并安装依赖

```bash
cd /data3/lg2/human_wm/V-jepa2-heatmap-new
pip install -r requirements.txt
pip install -e .
```

### 2. 检查配置路径

编辑：

```text
configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml
configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml
```

重点确认：

- `data.datasets`
- `data.gazelle.python_path`
- `data.gazelle.checkpoint`
- `meta.pretrain_checkpoint`
- `folder`
- `model_kwargs.checkpoint`
- Ego4D Keystep annotation 路径

### 3. 运行 Stage 2 gaze-conditioned 训练

```bash
python app/vjepa_ego4d/train_gaze_unified.py \
  --fname configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml
```

多 GPU 可使用通用 launcher：

```bash
python app/main.py \
  --fname configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml \
  --devices cuda:0 cuda:1
```

训练输出会保存到配置中的 `folder`，常见文件包括 `latest.pt` 和按 epoch 保存的 `e*.pt`。

### 4. 运行下游 Action Anticipation

```bash
python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml
```

指定某个 Stage 2 checkpoint：

```bash
python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml \
  --checkpoint logs/ego4d-heatmap-online-256px-gazelle224-grid16-n/e800.pt
```

## 训练语义说明

Stage 2 虽然常被口头称为 gaze fine-tuning，但实现上仍是自监督 JEPA 训练：

- 监督信号不是动作标签。
- Gaze / Gazelle tokens 作为条件输入。
- 训练目标是 latent prediction loss，包括 teacher forcing 和 auto-regressive loss。
- 真正的有监督训练发生在 Downstream action anticipation 阶段。

## 常见问题

### Stage 2 可以不跑 Stage 1 吗？

可以。直接将 `meta.pretrain_checkpoint` 指向 Meta V-JEPA2 的 `vitl.pt` 即可。若希望 encoder 更适配 ego-centric 视频分布，可以先跑 Stage 1，再将 Stage 1 的 `latest.pt` 写入 Stage 2 配置。

### 为什么 Gazelle 输入是 224？

当前配置将 Gazelle 强制到 `input_size: 224`，使其 scene token 网格为 `16x16`，和 V-JEPA2 `crop_size=256`、`patch_size=16` 得到的视频 patch 网格对齐。

### Downstream 报缺少 predictor 或 scene_proj 怎么办？

检查下游 YAML 中的 checkpoint key：

```yaml
pretrain_kwargs:
  predictor:
    checkpoint_key: predictor_causal
  scene_proj_checkpoint_key: scene_proj
```

同时确认 `model_kwargs.checkpoint` 指向的是 Stage 2 gaze-conditioned checkpoint，而不是原始 V-JEPA2 权重。

### Dataloader 是否读取 gaze 标注？

不读取。当前 unified 路线中 dataloader 只返回视频帧，Gazelle 在模型内部从 RGB 帧在线提取 scene / gaze tokens。

## 参考文档

- `docs/PIPELINE.md`：本项目三阶段流程的详细说明。
- `evals/action_anticipation_frozen/EGO4D_KEYSTEP_TRAINING.md`：Ego4D Keystep 下游训练细节。
- `WANDB_INTEGRATION.md`：WandB 日志配置说明。

## License

本仓库基于 Meta V-JEPA2 代码扩展，原始代码版权和 license 见仓库中的 `LICENSE` 文件。
