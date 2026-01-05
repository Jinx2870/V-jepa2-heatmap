视觉辅助动作预测模型设计方案

—— 基于 V-JEPA 2 Encoder 与 siMLPe MLP

1. 总体概述

本方案旨在结合 V-JEPA 2 强大的通用视觉表征能力与 siMLPe 轻量级高效的动作预测能力。我们将 V-JEPA 2 的 Vision Transformer (ViT) Encoder 作为冻结的特征提取器，提取环境与动作的视觉上下文，将其作为条件注入到 siMLPe 的 MLP 预测网络中，实现基于视觉感知的 3D 人体动作预测。

2. 数据预处理 (Data Preprocessing)

在数据进入网络之前，需要对视频和骨骼数据进行对齐和归一化处理。

2.1 视频数据 (Video Input)

采样 (Sampling): 从原始视频中采样 $T_{video}$ 帧（例如 16 帧或 64 帧），保持固定的帧率（例如 4fps）。

裁剪与缩放 (Crop & Resize): 将每帧图像调整为 V-JEPA 2 预训练时的标准分辨率（通常为 $224 \times 224$ 或 $256 \times 256$）。

归一化 (Normalization): 使用 ImageNet 或 V-JEPA 2 指定的均值和标准差对 RGB 通道进行归一化。

最终形状: [Batch, 3, T_video, H, W]

2.2 骨骼数据 (Skeleton Input)

格式: 3D 关节坐标 (XYZ)。

相对化处理 (Relative Coordinates):

为了使模型关注动作本身而非绝对位置，通常减去根节点（Root Joint，如骨盆）的坐标。

根节点的绝对轨迹通常单独预测或通过速度预测恢复。

归一化: 对所有关节坐标进行标准化（减去数据集均值，除以标准差），使其分布在 0 附近。

最终形状: [Batch, T_hist, Joints, 3]，其中 $T_{hist}$ 是历史观测帧数（如 50 帧）。

3. 网络架构与维度变化详解

阶段一：视觉特征提取 (Visual Encoding)

模块: V-JEPA 2 Encoder (Frozen ViT-L/H/g)

输入: Video Tensor $\rightarrow$ [B, 3, T_video, H, W]

Patch Embedding: 将视频切分为时空 Patch（Tubelets）。

假设 Patch 大小为 $16 \times 16$，时间跨度为 2。

Token 数量 $N_{tokens} = \frac{T_{video}}{2} \times \frac{H}{16} \times \frac{W}{16}$。

ViT 处理: 经过多层 Transformer Block。

输出 (Raw Tokens): [B, N_tokens, Dim_ViT] (例如 Dim_ViT=1024)

阶段二：视觉特征适配 (Visual Adapter / Pooling)

为了将海量的 Patch Token 转换为适合动作预测的时间序列，需要进行空间维度的压缩。

操作: Spatial Average Pooling。

Reshape: 将 Token 恢复为时空网格结构 $\rightarrow$ [B, T_feat, H_grid * W_grid, Dim_ViT]

Mean: 在空间维度 H_grid * W_grid 上求平均。

Align: 如果视觉特征帧数 $T_{feat}$ 与动作历史帧数 $T_{hist}$ 不一致，通过线性插值 (Interpolation) 对齐到 $T_{hist}$。

输出 (Visual Context): [B, T_hist, Dim_ViT]

阶段三：特征融合 (Feature Fusion)

将视觉上下文与骨骼运动历史结合。

骨骼嵌入 (Motion Embedding):

输入: [B, T_hist, Joints * 3]

操作: Linear Projection $\rightarrow$ Dim_Motion (例如 256)

视觉投影 (Visual Projection):

输入: [B, T_hist, Dim_ViT]

操作: Linear Projection $\rightarrow$ Dim_Motion (对齐维度)

融合 (Concat):

在特征通道维度拼接。

Concat([Motion, Visual]) $\rightarrow$ [B, T_hist, 2 * Dim_Motion]

再通过一个 Linear 层融合 $\rightarrow$ [B, T_hist, Hidden_Dim] (例如 1024)

阶段四：siMLPe 预测模块 (Prediction Module)

核心预测网络，输入融合后的历史特征，输出未来动作。

输入: [B, T_hist, Hidden_Dim]

1. DCT 变换 (Temporal Encoding):

操作: 对时间维度 $T_{hist}$ 进行离散余弦变换。

目的: 转换到频域，捕捉动作的平滑性和周期性。

输出: [B, T_hist, Hidden_Dim] (频域系数)

2. Spatial MLP:

操作: Linear $\rightarrow$ LayerNorm $\rightarrow$ Linear

作用: 学习骨骼点之间以及视觉环境对骨骼的影响（特征维度的交互）。

3. Temporal MLP Blocks (重复 m 次):

输入: [B, T_hist, Hidden_Dim] (转置后处理 T 维度)

结构: Res( x + MLP(Norm(x)) )

作用: 学习时间演变规律。

4. IDCT 变换 (Temporal Decoding):

操作: 离散余弦逆变换。

注意: 输出的时间长度可以是 $T_{future}$ (预测未来 N 帧)。

输出: [B, T_future, Hidden_Dim]

阶段五：回归头 (Regression Head)

操作: 简单的全连接层，将隐层维度映射回坐标维度。

计算: Linear(Hidden_Dim, Joints * 3)

输出 (Prediction): [B, T_future, Joints, 3]

4. 损失函数 (Loss Function)

训练时仅更新 Fusion 层、Adapter 和 siMLPe 模块的参数，Encoder 参数冻结。

MPJPE (Mean Per Joint Position Error):


$$\mathcal{L}_{pos} = \frac{1}{N \cdot J} \sum_{t=1}^{N} \sum_{j=1}^{J} \| \hat{P}_{t,j} - P_{t,j}^{GT} \|_2$$


其中 $\hat{P}$ 是预测坐标，$P^{GT}$ 是真实坐标。

(可选) 速度损失 (Velocity Loss):
为了保证动作的连贯性，可以增加速度约束。


$$\mathcal{L}_{vel} = \| (\hat{P}_{t+1} - \hat{P}_t) - (P_{t+1}^{GT} - P_{t}^{GT}) \|_2$$

总损失: $\mathcal{L} = \mathcal{L}_{pos} + \lambda \mathcal{L}_{vel}$

5. 实现关键点总结

步骤

关键操作

输入维度示例

输出维度示例

1. 视觉特征

V-JEPA 2 (Frozen)

[B, 3, 16, 224, 224]

[B, 8192, 1024]

2. 空间压缩

Reshape & Mean Pool

[B, 8192, 1024]

[B, 8, 1024]

3. 时间对齐

Interpolate

[B, 8, 1024]

[B, 50, 1024]

4. 骨骼输入

Flatten Joints

[B, 50, 22, 3]

[B, 50, 66]

5. 融合

Concat & Linear

Visual:[B,50,1024], Pose:[B,50,66]

[B, 50, 1024]

6. siMLPe

DCT -> MLPs -> IDCT

[B, 50, 1024]

[B, 25, 1024]

7. 输出

Regress Head

[B, 25, 1024]

[B, 25, 66]

6. 代码落地与工程实现方案 (Executable Blueprint)

本节在前面抽象设计的基础上，给出**可以直接按文件划分与接口定义去实现的具体方案**，重点包括：

- 模型文件与类结构：如何在 `vjepa2/src/models` 下新增 `motion_mlp` 与桥接网络。
- 前向计算流程：从 V-JEPA Encoder 输出到 siMLPe MLP 的精确张量维度与操作顺序。
- Ego4D 数据选取与切分：如何从 `ego_pose/train/val/body/annotation` 构造训练/验证集。

### 6.1 模型文件与类结构设计

- 新增文件 1: `vjepa2/src/models/motion_mlp.py`
  - **目标**：实现文中「阶段四：siMLPe 预测模块 + 阶段五：回归头」的完整 MLP 结构，输入为 `[B, T_hist, Hidden_Dim]`，输出为 `[B, T_future, Joints * 3]`。
  - **核心类**：`SiMLPeMotionMLP`
    - 初始化参数示例：
      - `hidden_dim`：与融合层输出一致（例如 1024）。
      - `num_joints`：骨骼关节数（例如 22）。
      - `t_hist`：历史长度（例如 50）。
      - `t_future`：预测未来长度（例如 25）。
      - `num_temporal_blocks`：Temporal MLP Block 重复次数（例如 4 或 6）。
    - forward 输入 / 输出：
      - 输入：`x_hist`，形状 `[B, T_hist, hidden_dim]`。
      - 输出：`pred`，形状 `[B, T_future, num_joints, 3]`。
    - 内部子模块（对应方案中的阶段四 + 五）：
      - `DCTLayer`：对时间维度 `T_hist` 做 1D DCT，输入/输出均为 `[B, T_hist, hidden_dim]`。
      - `SpatialMLP`：逐时间步的两层全连接 + Norm，输入/输出 `[B, T_hist, hidden_dim]`。
      - `TemporalMLPBlock`：
        - 输入 `[B, T_hist, hidden_dim]`，为方便做时间维度 MLP，可先转置为 `[B, hidden_dim, T_hist]`。
        - 结构：`x = x + MLP(Norm(x))`，其中 MLP 仅在时间维度上做 1D 卷积或 Linear。
      - `IDCTLayer`：在时间维度上做 IDCT，将频域信号转换回时域，可在实现中支持 `T_future != T_hist`。
      - `RegressionHead`：`nn.Linear(hidden_dim, num_joints * 3)`，逐时间步作用于输出，reshape 为 `[B, T_future, num_joints, 3]`。

- 新增文件 2: `vjepa2/src/models/motion_predictor.py`
  - **目标**：实现「Encoder 预测未来 3D 骨骼坐标」的完整桥接网络，将 V-JEPA Encoder 的 token 输出，经过空间池化 + 时序对齐 + 特征融合后送入 `SiMLPeMotionMLP`。
  - **核心类**：`VisualMotionPredictor`
    - 依赖模块：
      - 从 `src.models.vision_transformer` 导入 `VisionTransformer` 作为视觉 Encoder（冻结）。
      - 从 `src.models.motion_mlp` 导入 `SiMLPeMotionMLP`。
    - 初始化主要组件：
      - `self.encoder`：`VisionTransformer` 或其子类实例，参数、权重加载与现有配置保持一致，设置为 `requires_grad=False`。
      - `self.visual_proj`：`nn.Linear(Dim_ViT, Dim_Motion)`，将视觉特征投影到与骨骼嵌入相同的维度（如 256）。
      - `self.motion_embed`：`nn.Linear(Joints * 3, Dim_Motion)`，将骨骼历史 `[Joints*3]` 投影到 `Dim_Motion`。
      - `self.fusion_layer`：`nn.Linear(2 * Dim_Motion, Hidden_Dim)`，对应文中的 Fusion Layer。
      - `self.motion_mlp`：`SiMLPeMotionMLP(hidden_dim=Hidden_Dim, num_joints=Joints, t_hist=T_hist, t_future=T_future, ...)`。
    - forward 接口（训练 / 推理统一）：
      - 输入：
        - `video`: `[B, 3, T_video, H, W]`，与 V-JEPA 预训练分辨率一致（如 `[B, 3, 16, 224, 224]`）。
        - `skeleton_hist`: `[B, T_hist, Joints, 3]`，为当前时刻之前的骨骼 3D 历史。
      - 输出：
        - `pred_future`: `[B, T_future, Joints, 3]`，未来骨骼 3D 序列预测。

### 6.2 Encoder → Fusion → siMLPe 的前向流程

下述步骤可直接对照编码（伪代码形式，省略具体变量名与误差处理）：

1. 视觉编码 (Encoder)
   - 调用冻结的 V-JEPA Encoder：
     - `x_enc = encoder(video)`，输出形状 `[B, N_tokens, Dim_ViT]`。
     - 其中 `N_tokens = (T_video / tubelet_size) * (H / patch_size) * (W / patch_size)`。

2. 空间池化与时间对齐 (Visual Adapter)
   - 将 token 还原为时空网格：
     - `x_grid = x_enc.view(B, T_feat, H_grid * W_grid, Dim_ViT)`，其中  
       - `T_feat = T_video / tubelet_size`，  
       - `H_grid = H / patch_size`，`W_grid = W / patch_size`。
   - 在空间维度上求平均：
     - `x_spatial_pooled = x_grid.mean(dim=2)`，得到 `[B, T_feat, Dim_ViT]`。
   - 时间对齐到骨骼历史长度 `T_hist`：
     - 若 `T_feat != T_hist`，使用 `nn.functional.interpolate` 沿时间维度线性插值：  
       - `x_visual = interpolate(x_spatial_pooled.transpose(1, 2), size=T_hist, mode="linear")`  
       - 再转置回来得到 `[B, T_hist, Dim_ViT]`。

3. 骨骼历史嵌入 (Motion Embedding)
   - 预处理后骨骼历史为 `[B, T_hist, Joints, 3]`，先在关节维度展平：
     - `x_motion_flat = skeleton_hist.view(B, T_hist, Joints * 3)`。
   - 通过线性嵌入层：
     - `x_motion = motion_embed(x_motion_flat)`，输出 `[B, T_hist, Dim_Motion]`。

4. 视觉特征投影与融合 (Feature Fusion)
   - 将视觉上下文映射到相同维度：
     - `x_visual_proj = visual_proj(x_visual)`，形状 `[B, T_hist, Dim_Motion]`。
   - 在通道维度拼接：
     - `x_cat = concat([x_motion, x_visual_proj], dim=-1)`，得到 `[B, T_hist, 2 * Dim_Motion]`。
   - 通过融合线性层：
     - `x_fused = fusion_layer(x_cat)`，输出 `[B, T_hist, Hidden_Dim]`。

5. siMLPe MLP 预测未来骨骼 (Prediction)
   - 将融合特征送入 `SiMLPeMotionMLP`：
     - `pred_future = motion_mlp(x_fused)`，输出 `[B, T_future, Joints, 3]`。

6. 损失计算 (训练时)
   - 给定 GT 序列 `gt_future`，形状 `[B, T_future, Joints, 3]`：
     - 位置损失：`L_pos` 按照前文 MPJPE 定义。
     - （可选）速度损失：`L_vel` 按照前文定义。
   - 总损失：`L = L_pos + λ * L_vel`，仅对 `visual_proj / motion_embed / fusion_layer / motion_mlp` 的参数回传梯度。

### 6.3 Ego4D 数据选取与切分方案

我们使用 Ego4D `ego_pose` 中的 3D body annotation 作为骨骼 GT，目录如下：

- 训练集标注目录：  
  - `/data3/lg2/human_wm/Ego4d/test/annotations/ego_pose/train/body/annotation`
- 验证集标注目录：  
  - `/data3/lg2/human_wm/Ego4d/test/annotations/ego_pose/val/body/annotation`

方案约定：

- 训练集 (train split)：使用 `train/body/annotation` 下全部 JSON 文件，每个文件对应一个 clip（clip_uid 即文件名）。
- 验证集 (val split)：使用 `val/body/annotation` 下全部 JSON 文件，不再从训练集中再切一部分做验证，保证与官方划分一致。
- 如需开发集，可在训练脚本中对 train 的 clip uid 再划分，例如 90% 训练 / 10% dev，但本方案默认不额外拆分。

### 6.4 Ego4D Pose 数据解析与样本构造

1）单个 JSON 文件结构理解（简化）

- 每个 `*.json` 文件包含多个时间索引键（例如 `"337"`, `"340"` 等），每个键对应一个时间步的标注：
  - `annotation3D`：世界坐标系下的 3D 关节，例如  
    `annotation3D[joint_name] = { "x": ..., "y": ..., "z": ..., "num_views_for_3d": ... }`
  - `annotation2D`：多个摄像机视角的 2D 关节（`cam01`~`cam04`, `aria02` 等），其中 `aria02` 为第一人称视角，可用于抽取视觉帧。
- 对于动作预测任务，我们主要使用 `annotation3D` 作为 3D GT，同时利用对应时间戳在 `aria02` 视角视频中取帧作为视觉输入。

2）骨骼关节集合与顺序

- 设定一个固定的关节顺序列表 `joint_names`（用于构造 `[Joints, 3]` 向量），例如：
  - `["nose", "left-eye", "right-eye", "left-ear", "right-ear", "left-shoulder", "right-shoulder", "left-elbow", "right-elbow", "left-wrist", "right-wrist", "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle", "right-ankle", ...]`
- 实现中：
  - 从 `annotation3D` 中按 `joint_names` 顺序读取 `(x, y, z)`；
  - 若某个关节缺失，可选择：用上一帧的值填充 / 用 0 填充 / 直接跳过该样本（推荐“若缺失比例过高则跳过该 window”），避免引入过多噪声。

3）时间序列与样本切片

- 对每个 JSON：
  - 将所有时间索引（字符串形式的帧号）排序，得到 `frame_ids = [f_1, f_2, ..., f_T_total]`。
  - 为了匹配方案中 `T_hist=50`, `T_future=25`，使用滑动窗口在时间轴上生成训练样本：
    - 对窗口起点 `s` 从 `0` 到 `T_total - (T_hist + T_future)`，步长 `stride`（例如 2 或 5）：
      - 历史帧：`frames_hist = frame_ids[s : s + T_hist]`
      - 未来帧：`frames_future = frame_ids[s + T_hist : s + T_hist + T_future]`
- 对每个窗口：
  - 骨骼历史：对每个 `t ∈ frames_hist`，从 `annotation3D` 取对应关节，组装成 `[T_hist, Joints, 3]`。
  - 未来 GT：对每个 `t ∈ frames_future`，同样构造 `[T_future, Joints, 3]`。

4）视觉帧对齐与采样

- 对应视觉输入 `video`，以 `aria02` 视角为主：
  - 假设已经有从 `clip_uid` 和 `frame_id` 到原始视频帧的映射（例如 Ego4D 官方工具或已有的解码代码）。
  - 对每个样本：
    - 在覆盖 `frames_hist ∪ frames_future` 的时间范围内，从 `aria02` 视频中以固定帧率（例如 4fps）采样 `T_video` 帧：
      - 若 `T_video >= T_hist`，可对齐到历史区间内（例如选择覆盖 `frames_hist` 的时间窗）。
      - 形成张量 `[3, T_video, H_raw, W_raw]`，再通过前文的数据预处理变为 `[3, T_video, H, W]`。
- 如视觉帧暂时无法完全对齐（工程实现阶段），可以在早期实验中：
  - 先用骨骼历史单模态训练 `SiMLPeMotionMLP`（视觉路径置空或使用零张量），待视频解码工具接入后再启用视觉条件。

### 6.5 Ego4D 数据集类与 DataLoader 设计

为方便后续实现，推荐在 `vjepa2/src/datasets`（或现有数据集目录）中新增一个数据集类（此处只在文档中定义接口，不强制具体文件路径）：

- 类名：`EgoPoseMotionDataset`
- 初始化参数：
  - `split`: `"train"` 或 `"val"`，决定使用哪一个 annotation 目录。
  - `ann_root`: `/data3/lg2/human_wm/Ego4d/test/annotations/ego_pose/{train,val}/body/annotation`。
  - `t_hist`: 历史帧数（默认 50）。
  - `t_future`: 未来帧数（默认 25）。
  - `joint_names`: 固定的关节名列表。
  - `use_video`: 是否加载视觉帧（布尔），方便调试时只用骨骼信号。
  - 其他：`stride`, `image_size`, `video_root` 等。
- `__getitem__` 返回内容（对应模型 forward 的输入/输出）：
  - `video`: `[3, T_video, H, W]` 或 `None`（若 `use_video=False`）。
  - `skeleton_hist`: `[T_hist, Joints, 3]`。
  - `skeleton_future`: `[T_future, Joints, 3]`。
  - `meta`: 包含 `clip_uid`, `frame_ids_hist`, `frame_ids_future` 等调试信息。

在训练脚本中：

- 构造：
  - `train_dataset = EgoPoseMotionDataset(split="train", ann_root=.../train/body/annotation, ...)`
  - `val_dataset   = EgoPoseMotionDataset(split="val",   ann_root=.../val/body/annotation,   ...)`
- 分别使用 `DataLoader`（带随机打乱 / 多进程），将 batch 整理成：
  - `video`: `[B, 3, T_video, H, W]`
  - `skeleton_hist`: `[B, T_hist, Joints, 3]`
  - `skeleton_future`: `[B, T_future, Joints, 3]`

### 6.6 与现有 V-JEPA 代码结构的衔接约束

- 编码器复用：直接复用 `src/models/vision_transformer.py` 中的 `VisionTransformer` 实现，按照现有配置加载 V-JEPA 预训练权重，并在训练动作预测时**完全冻结**其参数。
- 风格与接口：`VisualMotionPredictor` 的风格可参考同目录下的 `VisionTransformerPredictor` / `VisionTransformerPredictorAC`：
  - 保持 `nn.Module` 子类写法；
  - 使用 `__init__` 中定义好 Linear 层与 MLP Block；
  - forward 中只做当前任务必要的张量变换与预测。
- 改动范围控制：本方案只新增 `motion_mlp.py` 与 `motion_predictor.py` 两个文件，并在训练脚本中最小化接入，不改动已有 Encoder 与其他下游任务模型结构，避免对现有流程产生回归风险。
