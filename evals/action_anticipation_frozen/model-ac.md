### 1. 模块划分与整体目标（V‑JEPA 2‑AC / `app/vjepa_droid`）

- **场景**：在 DROID 机械臂轨迹上，对 **已预训练好的 ViT‑g 视频编码器** 进行 **action-conditioned 后训练**，得到一个可以在 latent 空间中做世界动力学预测的模型。
- **主要代码入口**
  - 训练脚本：`app/vjepa_droid/train.py`
  - 数据读入：`app/vjepa_droid/droid.py`（`DROIDVideoDataset` / `init_data`）
  - 数据增强：`app/vjepa_droid/transforms.py`（`VideoTransform`）
  - 模型构建：`app/vjepa_droid/utils.py:init_video_model`
  - AC 预测器：`src/models/ac_predictor.py`（`VisionTransformerPredictorAC` / `vit_ac_predictor`）
- **模块划分与职责**
  - 数据模块：从 DROID 轨迹中采样视频帧、机器人状态、动作、相机外参 → DataLoader 输出 `(clips, actions, states, extrinsics, indices)`。
  - 冻结目标编码器 `target_encoder`：以 ViT‑g 结构将视频 clip 编码为 patch 级 latent token（teacher 表征），不更新梯度。
  - AC 预测器 `predictor`：在给定过去 latent + 动作/状态（可选外参）条件下，预测未来 latent。
  - 训练循环与损失：用 teacher latent 作为 supervision，对 `predictor` 的 teacher-forcing 输出和自回归 rollout 输出做 L1 / L^p 拟合。

---

### 2. 数据与 DataLoader 输出（DROID 机器人轨迹）

以 `droid-256px-8f.yaml` 为例：

- **采样设置（来自 config）**
  - `dataset_fpcs: [8]` → 每个 clip 取 **T = 8 帧**
  - `fps: 4` → 按约 4 fps 采样
  - `crop_size: 256`, `patch_size: 16`, `tubelet_size: 2`

- **单个样本从磁盘读出来的内容**（`DROIDVideoDataset.loadvideo_decord`）：
  - 视频：从路径 `path/recordings/MP4/...` 中抽取一段，采样出 **8 帧**：
    - 初始 `buffer`: `[T=8, H_raw, W_raw, 3]`
  - 机械臂状态（笛卡尔 + 手爪开合）：  
    - `states_raw: [T_raw, 7]`，再按采样帧索引和 `frameskip=1` 对齐到 8 帧：
    - **`states: [T=8, 7]`**
  - 相机外参：
    - `extrinsics: [T=8, 7]`（后面可选地做坐标变换）
  - 动作（相邻状态差分）：`poses_to_diffs(states)`：
    - 差分后时间长度减 1：
    - **`actions: [T-1=7, 7]`**

- **加上 Transform 和 Batch 之后，DataLoader 给训练循环的张量**（`train.py` 里 `load_clips`）：
  - `VideoTransform` 会：
    - 随机裁剪 / resize 到 256×256，归一化
    - 把 `buffer: [T, H, W, C]` 转成 `[C, T, H, W]`
  - DataLoader 默认 collate 堆成 batch 后：
    - **`clips: [B, 3, 8, 256, 256]`**
    - **`actions: [B, 7, 7]`**   （7 个时间步，每步 7 维动作）
    - **`states: [B, 8, 7]`**
    - **`extrinsics: [B, 8, 7]`**

---

### 3. 冻结的目标编码器 `target_encoder`（ViT‑g）

在训练循环中，真正用于前向抽特征的是 **`target_encoder`**（冻结，仅做前向）：

- **输入预处理（`forward_target`）**
  - 先把 `clips: [B, 3, 8, 256, 256]` 重新组织成“每帧当一个样本”：
    - `c.permute(0, 2, 1, 3, 4)` → `[B, 8, 3, 256, 256]`
    - `.flatten(0, 1)` → `[B*8, 3, 256, 256]`
    - `.unsqueeze(2).repeat(1, 1, 2, 1, 1)` → **`[B*8, 3, 2, 256, 256]`**  
      （人为复制一帧变成 2 帧，使得 `tubelet_size=2` 时每个样本恰好 1 个时序 tubelet）

- **ViT‑g 编码（`vision_transformer.vit_giant_xformers`）**
  - 配置：`embed_dim = 1408`, `patch_size = 16`, `tubelet_size = 2`
  - 对输入 `[B*8, 3, 2, 256, 256]`：
    - 时间方向：`2 / 2 = 1` 个 tubelet
    - 空间：`256/16 = 16`，每帧 16×16 = **256** 个 patch
    - 每个样本 token 数：**`tokens_per_frame = 256`**
    - 输出：**`[B*8, 256, 1408]`**

- **还原成 “(batch, 时间×patch)” 的 teacher 特征**
  - `h.view(batch_size=B, max_num_frames=T=8, -1, D=1408)` → `[B, 8, 256, 1408]`
  - `.flatten(1, 2)`：
    - **`h (teacher latent): [B, 8*256, 1408] = [B, 2048, 1408]`**
  - 若 `normalize_reps = True`（config 里是 true），再做一次按通道 `LayerNorm`。

> **小结（encoder 输入 / 输出）**  
> - 输入：从 DataLoader 取出的 `clips: [B, 3, 8, 256, 256]`，在 `forward_target` 里重整为 `[B*8, 3, 2, 256, 256]` 再送入 ViT‑g。  
> - 编码后：ViT‑g 输出每帧 256 个 patch latent，展平成每个 clip 的 **8 帧 × 256 patch**，得到 **`h: [B, 2048, 1408]`** 作为 teacher latent。

---

### 4. Action‑Conditioned 预测器 `predictor`（`VisionTransformerPredictorAC`）

#### 4.1 基本设置

- 来自 `src/models/ac_predictor.py` 的 `vit_ac_predictor`：
  - `predictor_embed_dim = 1024`（config: `pred_embed_dim: 1024`）
  - `embed_dim` 与 encoder 对齐：**1408**
  - `grid_height = grid_width = 256/16 = 16`，所以 **`tokens_per_frame = 16×16 = 256`**
  - `depth = 24`, `num_heads = 16`
  - `action_embed_dim = 7`（与 `states` / `actions` 的 7 维对齐）
  - `use_extrinsics = False`（在该 config 中 extrinsics 不参与条件化）

- 线性层：
  - `predictor_embed: Linear(1408 → 1024)`
  - `action_encoder: Linear(7 → 1024)`
  - `state_encoder: Linear(7 → 1024)`
  - （`extrinsics_encoder` 这路在 `use_extrinsics=False` 下不会用）

#### 4.2 第一步：Teacher forcing 预测（`z_tf`）

训练里 `forward_predictions(h)` 的第一步：

- 取 **前 T-1 帧** 的 token 作为上下文：
  - 输入 `z = h: [B, 2048, 1408]`
  - `z[:, :-tokens_per_frame]` 去掉最后一帧的 256 个 token：
    - **`_z: [B, (8-1)*256, 1408] = [B, 1792, 1408]`**
  - `_a = actions: [B, 7, 7]`
  - `_s = states[:, :-1]: [B, 7, 7]`

- 在 AC predictor 里（简化写法）：
  - 映射到预测器维度：
    - `_z → predictor_embed → [B, 1792, 1024]`
  - 计算时间步：
    - `T_ctx = N_ctxt / tokens_per_frame = 1792 / 256 = 7`
  - reshape 成 `[B, T_ctx, H*W, 1024]`：
    - `[B, 7, 256, 1024]`
  - 编码动作和状态：
    - `a_enc: [B, 7, 1, 1024]`，`s_enc: [B, 7, 1, 1024]`
  - 在每个时间步前面插入 action/state token：
    - `x = cat([a_enc, s_enc, x], dim=2)` → `[B, 7, 256+2, 1024]`
    - 展平时间×token：`[B, 7*(256+2)=1806, 1024]`
  - 经过 24 层 ACBlock + 因果注意力：
    - 保持 shape `[B, 1806, 1024]`
  - 再拆回 `[B, 7, 2+256, 1024]`，去掉前两个 cond token，只保留 patch token：
    - `[B, 7, 256, 1024]` → 展平成 `[B, 7*256=1792, 1024]`
  - 归一化 & 回投到 encoder 维度：
    - `predictor_norm` + `predictor_proj(1024→1408)`：
    - **`z_tf: [B, 1792, 1408]`**

> **解释**：`z_tf` 对应 **7 个时间步（帧 1–7）的 256 个 patch latent**，是在给定 “历史 latent + 动作/状态” 下的 **一步 teacher forcing 预测**。

#### 4.3 第二步：多步自回归 rollout（`z_ar`）

接着 `forward_predictions` 做自回归 rollouts（config: `auto_steps: 2`）：

- 初始化：
  - 取 **首帧的真实 latent** 和 **teacher forcing 预测出的下一帧 latent**：
    - `z[:, :256]`：原始 `h` 的第 0 帧 token → `[B, 256, 1408]`
    - `z_tf[:, :256]`：预测的第 1 帧 token → `[B, 256, 1408]`
    - 拼成：
      - **`_z: [B, 512, 1408]`**（当前时间长度 `T_cur = 2`）

- 自回归循环（`n` 从 1 到 `auto_steps-1`，这里只跑一次）：
  - 当前使用前 `n+1=2` 个 step 的条件：
    - `_a = actions[:, :2] : [B, 2, 7]`
    - `_s = states[:, :2] : [B, 2, 7]`
  - `_step_predictor(_z, _a, _s, _e)`：
    - 内部逻辑同上，输出 `[B, 2*256, 1408] = [B, 512, 1408]`
    - 取最后一帧 tokens：`[:, -256:]` → `[B, 256, 1408]`，这是 **下一步预测帧**
  - 把新预测帧接到 `_z` 后面：
    - `_z` 从 `[B, 512, 1408]` 变成 `[B, 768, 1408]`（对应 3 帧）

最后丢掉最初那一帧真实 token，仅保留纯预测的部分：
- `z_ar = _z[:, 256:]`：
  - **`z_ar: [B, 512, 1408]`**  
    对应 **两帧自回归 rollout 的 2×256 个 patch latent**。

#### 4.4 `VisionTransformerPredictorAC.forward` 的输入 / 输出总结

- **输入（以 `forward_predictions` 内部调用为例）**
  - `x`：来自 encoder 的上下文 latent token，形如 `[B, T_ctx * tokens_per_frame, 1408]`  
    - 其中 `tokens_per_frame = (crop_size // patch_size)^2 = (256//16)^2 = 256`，`T_ctx` 是使用的上下文帧数（例如 teacher forcing 时 `T_ctx=7`）。
  - `actions`：对应这些上下文帧的动作序列，`[B, T_ctx, 7]`
  - `states`：对应的机器人状态，`[B, T_ctx, 7]`
  - `extrinsics`：相机外参，当前 config 下 `use_extrinsics=False`，传入但不参与条件编码。
- **内部特征维度流（概括版）**
  - `x: [B, T_ctx*256, 1408] → predictor_embed → [B, T_ctx*256, 1024]`
  - reshape：`[B, T_ctx, 256, 1024]`，与 `actions / states: [B, T_ctx, 7]` 对齐时间维
  - 条件 token 拼接：每个时间步前插入 2 个 cond token（动作 + 状态），得到 `[B, T_ctx*(256+2), 1024]`
  - 24 层 ACBlock：保持 `[B, T_ctx*(256+2), 1024]`
  - 去掉 cond token，只保留 patch token：`[B, T_ctx*256, 1024]`
  - `predictor_norm` + `predictor_proj(1024→1408)`：输出 `[B, T_ctx*256, 1408]`
- **输出**
  - 与输入 `x` 在时间 / patch 维度一一对齐的预测 latent，形状：  
    - **`z_pred: [B, T_ctx * tokens_per_frame, 1408]`**

---

### 5. 损失与训练输出

- **对齐 teacher latent 与预测 latent 计算 loss**（`loss_fn`）：
  - 统一用 `tokens_per_frame = 256` 做“时间对齐”：
  - 对任意 `z`（`z_tf` 或 `z_ar`），长度 `N_z = z.size(1)`：
    - 从 teacher latent `h: [B, 2048, 1408]` 中截取对齐时间范围：
      - `_h = h[:, 256 : N_z + 256]`  
        （即让预测对齐到 “从第 1 帧开始的若干帧”）
    - Loss:
      - **`loss(z, h) = mean(|z - _h|^loss_exp) / loss_exp`**  
        在 config 中 `loss_exp = 1.0`，就是标准 L1。

- **总损失与训练输出**：
  - `jloss = loss(z_tf, h)`：teacher forcing 误差
  - `sloss = loss(z_ar, h)`：自回归 rollout 误差
  - **`loss = jloss + sloss`**
  - 训练循环只记录这些 scalar（loss, jloss, sloss、lr 等）并写入日志/ckpt，**没有分类 logits 或显式状态输出**，模型学习的是 **latent 空间中的世界动力学**。

---

### 6. 总结成一条线（V‑JEPA 2‑AC / `vjepa_droid`）

- **模型输入（后训练阶段）**：
  - 视觉：`clips: [B, 3, 8, 256, 256]`
  - 动作：`actions: [B, 7, 7]`（相邻状态差分）
  - 状态：`states: [B, 8, 7]`
  - 外参：`extrinsics: [B, 8, 7]`（当前 config 未参与条件化）

- **特征与预测张量形状**（以 droid-256px-8f 为例）：
  - **目标 encoder（冻结 ViT‑g）**：  
    - `[B, 3, 8, 256, 256] → [B, 2048, 1408]`（8 帧 × 256 patch）
  - **AC predictor**：
    - Teacher forcing：`[B, 2048, 1408] → [B, 1792, 1408]`（7 帧 latent，`z_tf`）
    - 自回归 rollout（`auto_steps=2`）：→ `[B, 512, 1408]`（2 帧 latent，`z_ar`）
  - **训练目标**：让 `z_tf`、`z_ar` 在对应时间段内逼近 teacher latent `h`，用 L1/L^p 损失。
