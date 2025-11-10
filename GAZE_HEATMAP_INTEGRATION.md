# Gaze Heatmap集成指南

本指南说明如何在V-JEPA 2模型中使用gaze heatmap来提升动作预测的准确度。

## 概述

本项目实现了将gaze heatmap作为条件特征集成到V-JEPA 2的动作预测模型中。heatmap是从ego4d数据集的第三视角视频中提取的,使用gazelle模型生成。

## 主要修改

### 1. 数据加载器 (`evals/action_anticipation_frozen/epickitchens.py`)

- 添加了`use_heatmap`和`heatmap_base_path`参数
- 支持从`.npz`文件或目录中加载heatmap
- Heatmap格式: `[T, H, W]` 其中T是帧数,H和W是空间维度

### 2. Predictor模型 (`src/models/heatmap_predictor.py`)

- 创建了`VisionTransformerPredictorHeatmap`类
- 使用CNN patch embedding处理heatmap
- 将heatmap token与视频tokens交错排列

### 3. Wrapper (`evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_heatmap_ar.py`)

- 创建了`AnticipativeWrapperHeatmap`类
- 支持在forward过程中传递heatmap

### 4. 训练流程 (`evals/action_anticipation_frozen/eval.py`)

- 修改了训练和验证循环以处理heatmap
- 自动检测模型是否支持heatmap

## 使用方法

### 1. 准备Heatmap数据

Heatmap数据应该按照以下结构组织:

**选项A: 使用.npz文件**
```
heatmap_base_path/
  video_id_1.npz  # 包含所有帧的heatmap, shape: [num_frames, H, W]
  video_id_2.npz
  ...
```

**选项B: 使用目录结构**
```
heatmap_base_path/
  video_id_1/
    frame_000000.npy
    frame_000001.npy
    ...
  video_id_2/
    ...
```

### 2. 配置文件

使用提供的配置文件模板 `configs/eval/vitg-384/ek100-heatmap.yaml`:

```yaml
experiment:
  data:
    use_heatmap: true
    heatmap_base_path: /path/to/your/heatmaps/
    # ... 其他数据配置
model_kwargs:
  module_name: evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_heatmap_ar
  pretrain_kwargs:
    predictor:
      model_name: vit_heatmap_predictor
      heatmap_patch_size: 16  # Heatmap的patch大小
      # ... 其他predictor配置
```

### 3. 运行训练

```bash
python evals/main.py --config configs/eval/vitg-384/ek100-heatmap.yaml
```

## 技术细节

### Heatmap处理流程

1. **加载**: 从文件系统加载heatmap,格式为`[T, H, W]`
2. **编码**: 使用CNN patch embedding将heatmap转换为特征向量
3. **融合**: 将heatmap token与视频frame tokens交错排列
4. **预测**: 在transformer中,heatmap token可以与视频tokens交互

### 模型架构

```
Video Frames → Encoder → [Frame Tokens]
                                    ↓
Gaze Heatmap → Heatmap Encoder → [Heatmap Token] → Predictor → Future Predictions
```

### 关键参数

- `use_heatmap`: 是否启用heatmap功能
- `heatmap_base_path`: Heatmap文件的根目录
- `heatmap_patch_size`: 处理heatmap时使用的patch大小(默认16)

## 注意事项

1. **Heatmap格式**: 确保heatmap的shape为`[T, H, W]`,其中H和W应该与视频分辨率匹配
2. **帧对齐**: Heatmap的帧数应该与视频帧数对齐
3. **缺失处理**: 如果某个视频的heatmap不存在,系统会自动使用零heatmap作为fallback
4. **内存**: 使用heatmap会增加内存使用,可能需要调整batch size

## 实验建议

1. **消融实验**: 对比使用和不使用heatmap的性能差异
2. **提前预测**: 测试使用heatmap是否能实现更提前的预测
3. **不同heatmap来源**: 尝试不同的heatmap提取方法
4. **融合策略**: 实验不同的heatmap与视频特征融合方式

## W&B可视化

你可以启用Weights & Biases来实时监控训练:

1. 在配置文件中开启:
   ```yaml
   experiment:
     wandb:
       enable: true
       project: your_wandb_project
       entity: your_team  # 可选
       name: ek100-vitg16-384-heatmap
       tags:
         - ek100
         - heatmap
         - gaze
   ```
2. 运行训练脚本后, W&B会自动记录每个epoch的`train/val`准确率和召回率, 以及每个优化器的学习率。
3. 训练结束或`val_only`评估完成后, 代码会自动调用`wandb.finish()`。

## 故障排除

### Heatmap加载失败
- 检查`heatmap_base_path`路径是否正确
- 确认heatmap文件名与视频ID匹配
- 检查heatmap文件格式是否正确

### 内存不足
- 减小batch size
- 减小heatmap分辨率
- 使用更小的patch size

### 模型不收敛
- 检查heatmap是否已正确归一化
- 调整学习率
- 检查heatmap与视频的对齐情况

## 未来改进方向

1. **多尺度Heatmap**: 支持不同分辨率的heatmap
2. **时序建模**: 更好地建模heatmap的时序信息
3. **注意力机制**: 使用注意力机制更好地融合heatmap和视频特征
4. **端到端训练**: 支持从原始gaze数据端到端训练

