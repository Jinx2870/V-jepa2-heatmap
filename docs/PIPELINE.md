# Pipeline：两阶段训练 + Action Anticipation 下游

本文档描述 `V-jepa2-heatmap-new` 的**正确端到端流程**：在 Ego-exo4D 视频上先做自监督表征适配，再做 Gaze 条件的世界模型训练，最后在 Ego4D Keystep 上做 Action Anticipation。

---

## 总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 1：Self-Supervised Adaptation（Ego-centric Video Representation）      │
│  输入：Ego-exo4D 视频（仅帧）                                                   │
│  骨干：V-JEPA 2 encoder（从 Meta vit*.pt 初始化）                               │
│  训练：encoder + predictor（标准 JEPA TF+AR，无 gaze）                          │
│  产出：ego 适配后的 encoder（+ target_encoder / predictor）                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 2：Gaze-Conditioned Adaptation（Video + Gaze Feature）                 │
│  输入：视频帧 + Gazelle 提取的 scene/gaze tokens                                │
│  视频 encoder：Stage 1 的 encoder（冻结，不再更新）                              │
│  训练：predictor_causal + scene_proj（JEPA TF+AR，仍以自监督为主）               │
│  产出：encoder / predictor_causal / scene_proj                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Downstream：Action Anticipation（有监督）                                     │
│  输入：视频 clip + anticipation time；条件：Gazelle + scene_proj（与 Stage 2 一致）│
│  骨干：Stage 2 checkpoint 中的 encoder / predictor / scene_proj（全部冻结）     │
│  训练：仅 AttentivePooler + 动作分类头                                          │
│  数据：Ego4D Keystep 标注（keystep_train.json / keystep_val.json）              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 三阶段对照表

| 阶段 | 目标 | 监督信号 | 可训练参数 | 入口 |
|------|------|----------|------------|------|
| **Stage 1** | Ego-centric 视频表征 | 无（JEPA 预测未来 latent） | encoder、predictor | `app/vjepa_ego4d/train.py` |
| **Stage 2** | Gaze 条件动态建模 | 无（JEPA）；gaze 作条件 | predictor_causal、scene_proj | `app/vjepa_ego4d/train_gaze_unified.py` |
| **Downstream** | 动作预判 | Keystep 动作标签 | 分类 probe | `evals/main.py` |

> **说明**：Stage 2 在论文表述里常叫 “fine-tuning”，在本仓库实现里仍是 **自监督 JEPA**（不是用动作标签训练 encoder）。**真正的有监督学习**发生在 Downstream。

---

## 共享数据与依赖

| 资源 | 路径（示例） | 用途 |
|------|----------------|------|
| 视频列表 | `Ego4d/test/output2.txt` | Stage 1/2 训练；Downstream 筛选 take |
| Ego4D 视频 | 列表中每行一个 MP4 绝对路径 | Stage 1/2 |
| Keystep 标注 | `Ego4d/test/annotations/keystep_{train,val}.json` | Downstream |
| 视频根目录 | `Ego4d/exo_data/takes` | Downstream 解析 cam*.mp4 |
| Meta V-JEPA2 权重 | `features/vitl.pt`（或 vitg.pt） | Stage 1 初始化 |
| Gazelle 代码与权重 | `gazelle/` repo + `gazelle_dinov2_vitb14_inout.pt` | Stage 2 / Downstream |

---

## Stage 1：Self-Supervised Adaptation

### 做什么

在 Ego-exo4D 视频上继续 V-JEPA 2 的 **masked latent 预测**（Teacher Forcing + Auto-Regressive），使 encoder 适应 ego 视角分布。

- **输入**：仅视频 clip `[B, C, T, H, W]`
- **不使用**：Gazelle / gaze / 动作标签
- **初始化**：Meta 发布的 `vitl.pt` / `vitg.pt`（`meta.pretrain_checkpoint`）

### 代码映射

| 组件 | 路径 |
|------|------|
| 训练入口 | `app/vjepa_ego4d/train.py` |
| 数据集 | `app/vjepa_ego4d/ego4d.py` → `Ego4DVideoACDataset` |
| 数据加载 | `init_data()` |
| Encoder / Predictor | `src/models/vision_transformer.py`、`src/models/predictor.py` |
| 参考配置 | `../vjepa2/configs/train/vitg16/ego4d-256px-8f.yaml`（`app: vjepa_ego4d`） |

> 本仓库 `configs/train/` 下**暂无** Stage 1 专用 yaml，可复制 `vjepa2` 的配置并改 `data.datasets` 为本地 `output2.txt`。

### 命令（示例）

```bash
# 通过 app/main.py 启动（需自备 Stage 1 yaml，app 字段为 vjepa_ego4d）
python -m app.main --fname configs/train/vitg16/ego4d-stage1-256px-16f.yaml

# 或直接调用 train（需自行加载 yaml 并传入 args）
```

### Checkpoint 产出

Stage 1 保存键（`app/vjepa_ego4d/train.py`）：

| Key | 含义 |
|-----|------|
| `encoder` | 训练中的 context encoder |
| `target_encoder` | EMA target encoder（推理时常用此份） |
| `predictor` | 标准 AC predictor（若 `load_predictor=true` 则一并训练） |

Stage 2 加载 encoder 时，config 里通常设：

```yaml
meta:
  pretrain_checkpoint: /path/to/stage1/latest.pt
  context_encoder_key: target_encoder   # 或 encoder，与 Stage 1 ckpt 一致即可
```

若跳过 Stage 1、直接用 Meta 权重，可设 `pretrain_checkpoint: features/vitl.pt`（与当前 Stage 2 默认配置一致）。

---

## Stage 2：Gaze-Conditioned Adaptation

### 做什么

在 **冻结 Stage 1 encoder** 的前提下，用 **视频帧 + Gazelle gaze/scene tokens** 训练因果 predictor，学习 gaze 条件下的未来表征预测。

- **输入**：视频帧；Gazelle 在模型内从帧在线提取 scene tokens（非 dataloader 读 npy）
- **Encoder**：来自 Stage 1（或 `vitl.pt`），**冻结**
- **可训练**：`predictor_causal`、`scene_proj`（Gazelle → predictor 空间的线性投影）
- **损失**：TF loss + AR loss（与 V-JEPA 一致，无动作标签）

### 代码映射

| 组件 | 路径 |
|------|------|
| 训练入口 | `app/vjepa_ego4d/train_gaze_unified.py` |
| 分布式别名 | `app/vjepa_ego4d_gaze_unified/train.py` |
| 配置 | `configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml` |
| 数据集 | `ego4d.py` → `Ego4DGazeVideoDataset`（仅返回 frames） |
| 统一模型 | `src/models/vjepa_gazelle_unified.py` → `VJEPAGazelleUnifiedModel` |
| Gazelle | 外部 `gazelle` 仓库（`data.gazelle.python_path`） |

### 命令

```bash
python app/vjepa_ego4d/train_gaze_unified.py \
  --fname configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml
```

### 配置要点

```yaml
# configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml
meta:
  pretrain_checkpoint: /path/to/stage1/latest.pt   # 或 vitl.pt
  context_encoder_key: target_encoder
data:
  datasets: [/path/to/Ego4d/test/output2.txt]
  condition_mode: gazelle
  gazelle:
    checkpoint: /path/to/gazelle/checkpoints/gazelle_dinov2_vitb14_inout.pt
    input_size: 224   # → 16×16 scene tokens，与 256/16 patch 对齐
model:
  condition_mode: gazelle
  model_name: vit_large
```

### Checkpoint 产出

| Key | 含义 | 下游是否使用 |
|-----|------|--------------|
| `encoder` | 冻结的 ViT（与 Stage 1 一致） | ✅ |
| `predictor_causal` | Gaze 条件因果 predictor | ✅（config 中 `checkpoint_key: predictor_causal`） |
| `scene_proj` | Gazelle token 投影层 | ✅ |

日志目录由 config 中 `folder` 指定，例如 `logs/ego4d-heatmap-online-256px-gazelle224-grid16-n/e800.pt`。

---

## Downstream：Action Anticipation

### 做什么

在 Ego4D **Keystep** 任务上，根据当前/过去帧与 anticipation time **预测未来动作类别**。

- **输入**：`[C, T, H, W]` + `anticipation_time`
- **骨干**：加载 Stage 2 的 `encoder`、`predictor_causal`、`scene_proj`；Gazelle 冻结
- **可训练**：`AttentivePooler` + 线性分类头（multi-head probe）
- **监督**：Keystep JSON 中的 `step_unique_id` → 类别索引

### 代码映射

| 组件 | 路径 |
|------|------|
| 入口 | `evals/main.py` |
| 评估逻辑 | `evals/action_anticipation_frozen/eval.py` |
| Gaze 版模型 | `evals/.../modelcustom/vit_encoder_predictor_concat_ar_gazelle.py` |
| 无 Gaze 基线 | `vit_encoder_predictor_concat_ar.py` |
| 数据 | `evals/.../ego4d_keystep.py`、`dataloader.py` |
| 配置（Gaze） | `configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml` |
| 配置（无 Gaze） | `configs/eval/vitl/ego4d_keystep.yaml` |
| 详细说明 | `evals/action_anticipation_frozen/EGO4D_KEYSTEP_TRAINING.md` |

### 命令

```bash
python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml
```

可选覆盖 checkpoint / 输出目录：

```bash
python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml \
  --checkpoint /path/to/stage2/e800.pt \
  --folder /path/to/eval_output
```

### Checkpoint 键对齐（必查）

预训练（Stage 2）保存名与下游 config **必须一致**：

| Stage 2 保存 | 下游 `pretrain_kwargs` |
|--------------|-------------------------|
| `encoder` | `encoder.checkpoint_key: encoder` |
| `predictor_causal` | `predictor.checkpoint_key: predictor_causal` |
| `scene_proj` | `scene_proj_checkpoint_key: scene_proj` |

```yaml
# configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml
model_kwargs:
  checkpoint: /path/to/stage2/e800.pt
  module_name: evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar_gazelle
  pretrain_kwargs:
    encoder: {model_name: vit_large, checkpoint_key: encoder}
    predictor: {model_name: vit_predictor, checkpoint_key: predictor_causal, ...}
    scene_proj_checkpoint_key: scene_proj
    gazelle: {checkpoint: ..., input_size: 224, ...}
```

---

## 端到端命令汇总

```bash
# ── Stage 1：Ego4D 自监督适配（需自备 yaml，见 vjepa2/configs/train/.../ego4d-*.yaml）
python -m app.main --fname configs/train/vitg16/ego4d-stage1-256px-16f.yaml

# ── Stage 2：Gaze 条件 JEPA（encoder 冻结）
python app/vjepa_ego4d/train_gaze_unified.py \
  --fname configs/train/vitg16/ego4d-256px-16f-gazelle224-grid16.yaml

# ── Downstream：Action Anticipation（骨干冻结，训分类头）
python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep_gazelle_actionpred.yaml
```

**最小路径**（跳过 Stage 1，直接用 Meta `vitl.pt` 初始化 Stage 2）：只跑 Stage 2 + Downstream 两条命令；将 Stage 2 config 中 `meta.pretrain_checkpoint` 设为 `vitl.pt` 即可。

---

## 与旧版 `V-jepa2-heatmap` 的关系

| 本仓库 Stage | 旧仓库对应 |
|--------------|------------|
| Stage 1 | `app/vjepa_ego4d/train.py` 或 `app/vjepa/train.py`（通用预训练） |
| Stage 2 | `app/vjepa_droid/train.py`（heatmap / gazelle 条件） |
| Downstream | `evals/action_anticipation_frozen` + `ego4d_keystep_gazelle` 等 |

`V-jepa2-heatmap-new` 将 Stage 2 收敛为 `train_gaze_unified.py` + `VJEPAGazelleUnifiedModel`；Downstream 使用 `vit_encoder_predictor_concat_ar_gazelle.py` 加载同一套 ckpt 键。

---

## 常见问题

**Q: Stage 2 算不算 “Supervised Fine-Tuning”？**  
A: 实现上是 **自监督**（JEPA loss），gaze 仅作 **条件信号**；动作监督只在 Downstream。

**Q: Stage 1 能否省略？**  
A: 可以。当前默认用 `vitl.pt` 直接进 Stage 2；若需更强 ego 域适配，建议先跑 Stage 1 再把 ckpt 写入 Stage 2 的 `pretrain_checkpoint`。

**Q: gaze 从哪来？**  
A: 本仓库用 **Gazelle 模型从 RGB 帧推断** scene tokens，不读 Ego4D 原始 gaze npy。heatmap npy 路线在 `V-jepa2-heatmap` 的 `condition_mode: heatmap`。

**Q: Downstream 报缺少 predictor？**  
A: 检查 eval config 中 `predictor.checkpoint_key` 是否为 `predictor_causal`（与 Stage 2 保存键一致）。
