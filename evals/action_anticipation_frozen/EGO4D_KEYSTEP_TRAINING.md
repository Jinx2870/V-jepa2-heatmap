# Ego4D KeyStep 训练与评估说明（action_anticipation_frozen）

本文档基于 `evals/action_anticipation_frozen` 代码，对数据集的选取与拆分策略、训练内容（训练目标与可训练参数）、以及端到端流程做系统说明。适用配置示例见 `configs/eval/vitl/ego4d_keystep.yaml`。

## 数据集选取逻辑与策略

- **数据集入口选择**
  - 通过配置项 `experiment.data.dataset` 选择数据分支：当包含 `ego4d` 或 `keystep` 时，使用 Ego4D KeyStep 分支。
  - 代码位置：`evals/action_anticipation_frozen/dataloader.py` 中的 `filter_annotations` 与 `init_data` 会分别路由到 `ego4d_keystep.filter_annotations` 与 `ego4d_keystep.make_webvid`。

- **标注解析与类别空间对齐**（train/val JSON）
  - 读取 `dataset_train` 与 `dataset_val` 两个 JSON。
  - 从 train 的 `segments[*].step_unique_id` 收集所有“叶子动作 ID”，构造稳定的 `actions: {unique_id -> class_idx}`（只以 train 中出现的类定义类别空间）。
  - 构建每个 take 的片段表：记录 `start_time`、`end_time` 与映射后的 `action` 类别索引；若某片段的 `unique_id` 未出现在 train 的类别空间，直接跳过（保证类别一致性）。
  - 同步统计 val 中出现过的类别索引集合 `val_actions`，供评估时做有效类别过滤（避免对 val 未出现的类计分）。

- **视频路径解析与回退策略**
  - 视频根目录由 `experiment.data.base_path` 指定，按以下优先级查找：
    1) `<take>/frame_aligned_videos/downscaled/448/<camera_glob>`
    2) `<take>/frame_aligned_videos/downscaled/448/cam01*.mp4`
    3) `<take>/frame_aligned_videos/downscaled/448/*.mp4`
  - 找不到或不可读的视频会被跳过；无片段的 take 也会被跳过。

## 训练/验证集的拆分方式

- **外部提供拆分**：
  - 训练集与验证集通过配置中的 `dataset_train` 与 `dataset_val` 两个 JSON 文件显式提供，不在代码中再做随机划分。
- **返回结构**：
  - `train=(paths_list, {video_path -> DataFrame(segments)})`
  - `val=(paths_list, {video_path -> DataFrame(segments)})`
  - `actions`: 统一的类别空间（以 train 定义）；`val_actions`: val 中实际出现的类集合。

## 采样与增强策略（动作预判）

- **采样窗口**（decode_videos_to_clips）：
  - 对于每个片段 `[start_time, end_time]`，随机采样：
    - `a ∈ train_anticipation_time_sec`（训练）或 `a ∈ anticipation_time_sec`（验证），表示“提前量（秒）”；
    - `ap ∈ train_anticipation_point`（训练）或 `ap ∈ val_anticipation_point`（验证），用于确定锚点靠近片段起点或终点的比例。
  - 计算锚点时间：`af = start_time * ap + (1 - ap) * end_time - a`。
  - 使用 decord 读取视频原始帧率 `vfps`，设定帧步长 `frame_step = ceil(vfps / fps)`；
  - 以 `af` 为结束点，向前取 `frames_per_clip` 帧（总帧数 `nframes = frames_per_clip * frame_step`），帧索引不足时向 0 截断。
  - 产出样本三元组：`(video, action, anticipation_time)`，其中 `video` 经过下述变换后为形状 `[C, T, H, W]`。

- **图像增强与归一化**：
  - 训练态：随机裁剪（支持 motion_shift）、随机水平翻转、归一化、可选 RandomErasing；
  - 验证态：Resize 到短边，再 CenterCrop，归一化；
  - 具体变换由 `VideoTransform` 管理，输出张量为 `[C, T, H, W]`。

- **数据管线与 ipe 估算**：
  - 使用 WebDataset 风格的数据管线：按 `paths_list` 构建 `ResampledShards`，再分发到节点与工作线程，解码为 `(video, action, anticipation_time)`，最后按批量打包。
  - 估算 `ipe`（iterations per epoch）：`num_batches = max(1, num_clips // max(1, world_size * batch_size))`，同时记录 `num_samples = num_clips`。

## 训练内容与可训练参数

- **骨干模型**：
  - 由 `model_kwargs` 指定并从 `checkpoint` 加载，包含：
    - `Encoder (VisionTransformer)`：处理 `[B, C, T, H, W]`，输出时空 patch tokens；`num_frames=frames_per_clip`；
    - `Predictor (VisionTransformerPredictor)`：基于 mask tokens 在目标未来位置生成 token 表达；
    - `AnticipativeWrapper`：根据 `anticipation_time` 计算要跳过的 token 段与要预测的位置，支持自回归式 `num_steps` 滚动与 `num_output_frames` 未来帧数。
  - 参数全部冻结（`requires_grad=false`），前向在 `no_grad()` 下执行。

- **分类探针（可训练）**：
  - 结构：`AttentivePooler`（注意力查询聚合） + 线性分类器（动作类），在 KeyStep 场景为 action-only（单 query）。
  - 支持多头并行（`multihead_kwargs`）：每个探针头拥有独立的优化器、调度器与 AMP scaler。

- **损失与优化**：
  - 损失：默认 CrossEntropy，可选 Focal Loss（`use_focal_loss=true`）以缓解类别不平衡；
  - 优化：AdamW；
  - 学习率与权重衰减：Warmup + 余弦退火调度（总步数 `num_epochs * ipe`），每次迭代步进；
  - 混合精度：可使用 bfloat16 AMP（`use_bfloat16=true`）。

## 端到端训练流程（概要）

1) 读取配置，解析数据/优化/模型参数；
2) `filter_annotations`：构建类别空间、视频路径、每视频片段表，并得到 `train`/`val` 与 `val_actions`；
3) `init_module`：加载冻结的 Encoder + Predictor + Wrapper；
4) `init_classifier`：创建一个或多个探针分类头；
5) `init_data(train/val)`：构建数据管线与 DataLoader，估算 `ipe` 与 `val_ipe`；
6) `init_opt`：为每个探针头创建优化器、调度器与（可选）AMP scaler；
7) 训练循环（按 epoch）：
   - 非 `val_only`：`train_one_epoch` 跑 `ipe` 次迭代：
     - 取 batch `(video, action, anticipation_time)`；
     - `no_grad` 下跑骨干前向得到未来 token 表达；
     - 探针前向得到 logits，计算损失并仅更新探针参数；
     - 调度器步进，记录 acc / mean class recall；
   - 验证：跑 `val_ipe` 次，`no_grad` 前向与指标计算，并用 `val_actions` 限定有效类别；
   - 记录日志与保存 `latest.pt`（仅包含探针与优化状态）。

## 配置字段关键项（示例）

- `experiment.data.dataset`: 设为 `EGO4D_KEYSTEP` 触发 Ego4D 分支；
- `experiment.data.base_path`: Ego-Exo4D takes 根路径；
- `experiment.data.dataset_train` / `dataset_val`: 训练与验证的 KeyStep JSON；
- `experiment.data.camera_glob`: 优先相机（如 `aria01*.mp4`），若匹配失败回退 `cam01*.mp4` 与 `*.mp4`；
- `experiment.data.frames_per_clip` / `frames_per_second` / `resolution`: 采样与模型维度；
- `experiment.data.train_anticipation_time_sec` / `train_anticipation_point`: 训练采样超参；
- `experiment.data.anticipation_time_sec` / `val_anticipation_point`: 验证采样超参；
- `experiment.optimization.batch_size` / `num_epochs` / `use_bfloat16` / `use_focal_loss`；
- `experiment.optimization.multihead_kwargs`: 多头探针的 LR/WD/Warmup 等；
- `model_kwargs.checkpoint`: V-JEPA2 预训练权重（需同时包含 encoder 与 predictor 权重键）；
- `model_kwargs.module_name`: 冻结骨干封装模块名（如 `evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar`）；
- `model_kwargs.pretrain_kwargs.encoder`: 例如 `{model_name: vit_large, checkpoint_key: encoder}`；
- `model_kwargs.pretrain_kwargs.predictor`: 例如 `{model_name: vit_predictor, checkpoint_key: predictor, use_mask_tokens: true, num_mask_tokens: 10}`；
- `model_kwargs.wrapper_kwargs`: 如 `{num_output_frames: 2, num_steps: 1}` 控制预测的未来帧与自回归步数。

## 关键超参影响

- `frames_per_clip` 与 `fps` 决定采样窗口时长（秒）与 token 的时间步；
- `resolution` 与主干 `patch_size` 决定空间网格大小 `grid = resolution / patch_size`，进而影响目标 token 数 `N_pred = grid^2 * (num_output_frames / tubelet)`；
- `num_output_frames` 与 `num_steps` 决定预测未来 token 的数量与自回归滚动步数；
- `camera_glob` 影响视角选择；
- `batch_size`、`world_size` 影响 `ipe`，进而影响学习率与权重衰减调度的总步数与收敛节奏。

## 注意事项

- 个别 take 缺失视频或解码异常会被跳过；
- 片段过短时帧索引会向 0 截断，可能改变分布；
- 类别严重不平衡时，建议开启 `use_focal_loss`；
- 确保 `checkpoint` 同时包含 `encoder` 与 `predictor` 两个键的权重。


