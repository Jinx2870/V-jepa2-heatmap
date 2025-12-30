### 整体定位：AttentiveClassifier 在整网里的角色

在你现在的 `ego4d_keystep` 配置下：

- **`encoder + predictor`（来自 `vitl.pt`）**：负责把一段约 16 秒的视频，编码成“未来一帧的 256 个 patch 特征”  
  形状：`[B, 256, 1024]`
- **`AttentiveClassifier`**：只看这 `256×1024` 的特征，**学会“从未来这帧里读出 key-step 动作类别”**。

所以 `AttentiveClassifier` 是一个 **可训练的“读出头（readout head）”**，它自己不改 video 表征空间，只负责从固定的 ViT 表征里抽取能区分动作的信号。

---

### 1. 结构拆解：AttentiveClassifier 由哪些子模块组成？

定义位置：`evals/action_anticipation_frozen/models.py`

```python
class AttentiveClassifier(nn.Module):
    def __init__(..., embed_dim, num_heads, depth, ...):
        self.pooler = AttentivePooler(
            num_queries=1 if self.action_only else 3,
            embed_dim=embed_dim,          # 这里是 1024
            num_heads=num_heads,          # YAML 里是 8
            depth=depth,                  # YAML 里 num_probe_blocks=2
            use_activation_checkpointing=use_activation_checkpointing,
        )
        if not self.action_only:
            self.verb_classifier = nn.Linear(embed_dim, num_verb_classes)
            self.noun_classifier = nn.Linear(embed_dim, num_noun_classes)
        self.action_classifier = nn.Linear(embed_dim, num_action_classes)
```

对 **Ego4D KeyStep** 来说：

- 没有 verb/noun，`self.action_only = True`
- 所以：
  - **池化模块**：`AttentivePooler(num_queries=1, embed_dim=1024, num_heads=8, depth=2)`
  - **分类头**：`Linear(1024 → 602)`（602 个 key-step 类）

可以分三层理解：

1. **输入**：`[B, 256, 1024]`（未来一帧 256 个 patch 特征）
2. **AttentivePooler**：用 cross-attention 把 256 个 patch 聚成 1 个 1024 维的 clip-level 表征
3. **Linear 分类头**：把这 1024 维映射到 602 维动作 logits

---

### 2. AttentivePooler：怎么把 256 个 patch 聚成 1 个向量？

定义位置：`src/models/attentive_pooler.py`

#### 2.1 结构

```python
class AttentivePooler(nn.Module):
    def __init__(num_queries, embed_dim, num_heads, depth, ...):
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        if depth > 1:
            self.blocks = ModuleList([Block(dim=embed_dim, num_heads=num_heads, ...) for _ in range(depth-1)])
        self.cross_attention_block = CrossAttentionBlock(...)
```

对 KeyStep：

- `num_queries = 1`
- `embed_dim = 1024`
- `num_heads = 8`
- `depth = 2` → 有 `depth-1 = 1` 个自注意力 Block + 1 个 cross-attention Block。

#### 2.2 前向流程与维度

假设输入是 `x`：

- `x` 来自 predictor：
  \[
  x \in \mathbb{R}^{B \times 256 \times 1024}
  \]

1. **（可选）自注意力 Block（depth-1 层）**

```python
if self.blocks is not None:
    for blk in self.blocks:
        x = blk(x)  # 标准 Transformer Block
```

- 每个 `Block` 是一个 **self-attention + MLP** 的 Transformer 层：
  - 注意力维度：`dim=1024`, `num_heads=8`
- 形状保持不变：
  \[
  [B, 256, 1024] \rightarrow [B, 256, 1024]
  \]

直观理解：**在进入池化前，先在 patch 之间做一次“内部交流”，让信息在 256 个 patch 之间传播一下。**

2. **构造 query tokens**

```python
q = self.query_tokens.repeat(len(x), 1, 1)  # [1, 1, 1024] → [B, 1, 1024]
```

- `query_tokens` 是 **可学习的全局查询向量**：
  - 形状：`[1, 1, 1024]`
  - 类似于 ViT 里的 `[CLS]`，但这里是 **单独的模块**，不和 encoder 权重耦合；
- 对每个样本重复一份 → `q ∈ [B,1,1024]`。

3. **Cross-Attention 聚合**

```python
q = self.cross_attention_block(q, x)  # q: query, x: key & value
```

- `CrossAttentionBlock` 内部基本是：
  - `LN(q)`, `LN(x)`  
  - `Multi-Head Cross-Attention(q, k=x, v=x)`  
  - 残差 + MLP
- 维度变化：
  \[
  q: [B, 1, 1024],\quad x: [B, 256, 1024]
  \xrightarrow{\text{Cross-Attention}}
  q_{\text{out}}: [B, 1, 1024]
  \]

直观理解：**把 256 个 patch 当成一堆“记忆槽（memory slots）”，用一个 learnable query 去读取它们的加权和**：

- 某个时间步上，一个 query 头会对 256 个 patch 打分（softmax 权重）；
- 输出是这些 patch 的加权和；
- 多头 attention + 残差&MLP 后，得到对动作预测最有用的一个 1024 维向量。

4. **AttentivePooler 输出**

返回 `q`：

\[
\text{pooler}(x) \in \mathbb{R}^{B \times 1 \times 1024}
\]

---

### 3. 最后线性分类：从 `[B,1,1024]` 到 `[B,602]`

在 `AttentiveClassifier.forward` 里：

```python
x = self.pooler(x)          # [B, 1, 1024]
if not self.action_only:
    ...
else:
    x_action = x[:, 0, :]   # [B, 1024]
    x_action = self.action_classifier(x_action)  # Linear(1024 → 602)
    return dict(action=x_action)
```

对 Ego4D KeyStep：

1. **去掉 query 维度**
   - `x_action = x[:, 0, :]`
   - 形状：**`[B, 1, 1024] → [B, 1024]`**

2. **线性分类层**
   - `action_classifier: Linear(1024, num_action_classes)`
   - 这里 `num_action_classes = len(action_classes) = 602`；
   - 形状：**`[B, 1024] → [B, 602]`**

3. **输出字典**
   - `{"action": logits}`，参与交叉熵 loss 与 recall/accuracy 计算。

---

### 4. 对 Epic-Kitchens 的扩展（对比理解，帮助直觉）

虽然你现在跑的是 Ego4D（只有 action），但理解一下 **有 verb/noun 时的形态** 会更直观：

- Epic-K 时：
  - `num_queries = 3`，分别对应 verb / noun / action；
  - pooler 输出：`[B, 3, 1024]`；
  - 分别接三个 Linear：
    - `verb_classifier: [B,1024] → [B,#verb]`
    - `noun_classifier: [B,1024] → [B,#noun]`
    - `action_classifier: [B,1024] → [B,#action]`
- 这解释了 KeyStep 代码里 action-only 情况下的分支逻辑。

---

### 5. 训练视角下：AttentiveClassifier 学的是什么？

结合前面的冻结策略：

- **encoder + predictor 全部 frozen**（从 `vitl.pt` 加载，`requires_grad=False`）；
- **grad 只更新 AttentiveClassifier（pooler + Linear 头）**。

因此，AttentiveClassifier 本质上在学两件事：

1. **学一个“注意力读出方式”**：
   - query token + cross-attention 决定在未来帧的 256 个 patch 里，**更关注哪一块区域**（时空中的哪个 patch 更关键）；
   - 这相当于一个 **learned spatial attention** 专门为 key-step 动作服务。

2. **学一个“分类超平面”**：
   - 在线性层里，学习 602 个类别的决策边界；
   - 相当于在固定的 ViT 表征空间中，为 Ego4D KeyStep 找一个新的线性分类器。

因为 encoder/predictor 不动，你可以把 AttentiveClassifier 看成：

> “在 V-JEPA 已经学好的视觉表示空间上，  
>  用一个小型注意力池化 + 线性分类器，专门为 Ego4D key-step 做 probing / fine-tuning。”

---

如果你愿意，我可以再画一个更形式化的“小表格”，把 AttentiveClassifier 整体当成 3 行：

- 输入形状 → Pooler 内部（自注意力 + cross-attention）的形状 → 最终 Linear 的形状，方便你直接贴进 `model.md`。