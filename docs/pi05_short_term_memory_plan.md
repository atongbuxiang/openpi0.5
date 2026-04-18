# 基于 MEM 论文 C 部分的 `pi0.5` 短期记忆视频编码器改造方案

本文档基于论文《MEM: Multi-Scale Embodied Memory for Vision Language Action Models》中的短期记忆模块，也就是附录 C 的 `Video encoder with Space-Time separable attention`，结合本仓库当前 `pi0.5` 实现，给出一份可落地的模型改造方案。

本方案只覆盖短期记忆系统，即“优化的视频编码器”。不包含论文中的长时语言记忆 `m_t` 与高层策略 `\pi_{HL}`。

## 1. 目标

把当前 `pi0.5` 的“多相机单帧编码”升级为“多相机短视频编码”，让模型在不显著破坏实时性的前提下获得：

- 最近数秒的视觉记忆
- 对遮挡和自遮挡的鲁棒性
- 对抓取失败、错位等局部操控误差的快速在上下文内修正能力
- 比“直接把所有历史帧都拼进 LLM 前缀”更低的 token 开销

## 2. 当前 `pi0.5` 实现与论文设定的差距

### 2.1 当前实现

当前仓库里，`pi0.5` 的视觉前缀是“按相机逐帧编码，然后直接拼接”：

- [src/openpi/models/model.py](/e:/code/openpi0.5/src/openpi/models/model.py#L81) 中 `Observation.images` 的单个元素形状是 `[*b, h, w, c]`，没有时间维。
- [src/openpi/models/model.py](/e:/code/openpi0.5/src/openpi/models/model.py#L144) 的 `preprocess_observation()` 也是按单帧图像做 resize / augment。
- [src/openpi/models/pi0.py](/e:/code/openpi0.5/src/openpi/models/pi0.py#L106) 的 `embed_prefix()` 遍历每个相机。
- [src/openpi/models/pi0.py](/e:/code/openpi0.5/src/openpi/models/pi0.py#L114) 中每个相机图像直接送入 `self.PaliGemma.img(...)`。
- [src/openpi/models/siglip.py](/e:/code/openpi0.5/src/openpi/models/siglip.py#L177) 的 `SigLIP` 编码器本质上仍是单图 `ViT`，只输出当前帧 patch tokens。

因此当前前缀的结构更接近：

`[cam1 当前帧 patch tokens] + [cam2 当前帧 patch tokens] + [cam3 当前帧 patch tokens] + [prompt tokens]`

它没有显式建模时间，也没有在视觉编码阶段压缩历史信息。

### 2.2 直接堆历史帧为什么不合适

如果沿用当前设计，把过去 `K` 帧逐帧过一次 `SigLIP` 再全部拼给 `Gemma`，代价会线性甚至更糟地上涨：

- 每个相机 token 数乘以 `K`
- LLM 前缀长度显著增长
- prefix attention 与 KV cache 体积显著变大
- 推理延迟和显存占用会快速失控

这正是 MEM 论文 C 部分要解决的问题。

## 3. 论文 C 部分的核心理论

论文把单帧 ViT 扩展成视频编码器，但尽量复用原有 ViT 权重。

### 3.1 时间位置编码

对第 `l` 层、空间 patch `p`、时间步 `t` 的输入 token `z^{l-1}_{p,t}`，加入时间正弦位置编码：

`\\hat z^{l-1}_{p,t} = z^{l-1}_{p,t} + e(t)`

其中 `e(t)` 是只依赖时间步的 sinusoidal embedding，且满足：

`e(0) = 0`

这个边界条件非常关键，因为它让“当前帧”尽量与原始单帧编码行为对齐，从而更容易继承预训练 ViT 的能力。

### 3.2 复用原始 ViT 的 Q/K/V 投影

论文不重新定义一套视频注意力参数，而是继续复用原始 ViT 每层每头的投影矩阵：

`q^{l,a}_{p,t} = W^{l,a}_Q LN(\\hat z^{l-1}_{p,t})`

`k^{l,a}_{p,t} = W^{l,a}_K LN(\\hat z^{l-1}_{p,t})`

`v^{l,a}_{p,t} = W^{l,a}_V LN(\\hat z^{l-1}_{p,t})`

这意味着视频化主要发生在“attention 的连接方式”，不是彻底重训一个新 backbone。

### 3.3 空间-时间可分离注意力

论文先定义一般化注意力 `\\alpha(S, T)`，分别指定空间索引集合 `S` 和时间索引集合 `T`。

然后将视频注意力写成空间-时间可分离形式：

`\\alpha^{l,a}_{p,t}(S=\\{1,...,N\\}, T=\\emptyset) [ \\alpha^{l,a}_{p,t}(S=\\emptyset, T=\\{1,...,T\\}) [\\hat z] ]`

工程上可以理解为：

1. 先做时间方向注意力
2. 再做空间方向注意力

并且结合正文图示，设计意图是：

- 空间注意力：同一帧内双向
- 时间注意力：跨帧因果，只看当前及过去

这样既保留单帧 ViT 对空间结构的建模能力，又用较低代价引入历史信息。

### 3.4 token 压缩思想

论文正文图 4 还强调了一点：在较高层逐步丢弃过去时间步的 observation tokens，只保留更少、更抽象的历史 token 传给 VLM backbone。

这部分虽然附录公式没有展开，但它是让方法真正可用于实时机器人推理的关键。

## 4. 对 `pi0.5` 的结构映射

论文的短期记忆模块可以映射到当前 `pi0.5` 的视觉路径如下：

- 论文中的 image/video encoder
  对应本仓库 [src/openpi/models/siglip.py](/e:/code/openpi0.5/src/openpi/models/siglip.py)
- 论文中的 observation history `o_{t-K:t}`
  对应本仓库 `Observation.images`，但需要从单帧扩展为短序列
- 论文中的压缩后视频 token
  对应本仓库 [src/openpi/models/pi0.py](/e:/code/openpi0.5/src/openpi/models/pi0.py#L106) 里 `embed_prefix()` 返回的 image token 段

因此最自然的接入点不是改 `Gemma`，而是：

1. 扩展 `Observation` 的图像输入格式
2. 新增一个 `VideoSigLIP` 或 `SigLIPMemoryEncoder`
3. 在 `embed_prefix()` 中将每个相机的历史帧先压缩成较短 token 序列，再送入 `PaliGemma.llm`

## 5. 推荐改造方案

## 5.1 总体原则

优先采用“尽量不动 LLM、优先改视觉编码前缀”的路线：

- 保持 `Gemma` / action expert 基本不变
- 保持 suffix flow-matching 路径不变
- 仅替换 prefix 中每个相机的图像编码方式
- 先做短期记忆，再决定是否叠加长时语言记忆

这是对当前仓库侵入性最小、收益风险比最高的做法。

## 5.2 输入层改造

把每个相机从单帧输入改成短视频输入。

当前：

- `images[name]`: `[B, H, W, C]`
- `image_masks[name]`: `[B]`

建议新增兼容格式：

- `images[name]`: `[B, K, H, W, C]`
- `image_masks[name]`: `[B, K]`

其中：

- `K` 是短期记忆窗口，建议初始取 `4` 或 `6`
- 最后一个时间步固定为当前帧
- `t=0` 对应当前帧，`t<0` 对应过去帧

兼容策略：

- 如果输入仍是 4 维单帧，则自动扩成 `K=1`
- 这样可保证旧数据和旧推理接口不立即失效

对应需要修改的模块：

- [src/openpi/models/model.py](/e:/code/openpi0.5/src/openpi/models/model.py)
- 与数据变换相关的 `transforms` / dataset adapter
- 远端推理或机器人 runtime 的 observation 构造逻辑

## 5.3 新增视频编码器模块

建议新增文件：

- `src/openpi/models/video_siglip.py`

核心职责：

1. 复用当前 `SigLIP` patch embedding、空间 block 权重
2. 增加时间位置编码 `e(t)`，并满足 `e(0)=0`
3. 在每个 block 中插入空间-时间可分离注意力
4. 输出压缩后的历史视觉 tokens，供 `Pi0.embed_prefix()` 使用

建议接口：

```python
video_tokens, aux = self.PaliGemma.video_img(video_frames, frame_mask, train=train)
```

其中：

- `video_frames`: `[B, K, H, W, C]`
- `frame_mask`: `[B, K]`
- `video_tokens`: `[B, S_video, D]`

## 5.4 block 级实现方式

建议把当前 `siglip.py` 中的 `Encoder1DBlock` 扩展成两段：

1. `TemporalCausalBlock`
2. `SpatialBlock`

或者做成一个新 block：

- `SpaceTimeSeparableEncoderBlock`

推荐的数据排布：

- patchify 后得到 `[B, K, N, D]`
- `N` 是单帧 patch 数

每层做：

1. 加时间位置编码
2. 时间注意力：对每个 patch 位置 `p`，在 `K` 个时间步上做因果 attention
3. 空间注意力：对每个时间步 `t`，在 `N` 个 patch 上做双向 attention
4. MLP

实现上可以通过 reshape 复用现有 `nn.MultiHeadDotProductAttention`：

- temporal: `(B*N, K, D)`
- spatial: `(B*K, N, D)`

这样改造最直接，也最接近论文公式。

## 5.5 因果时间 mask

时间注意力必须是因果的，否则训练时会泄漏未来帧信息。

对时间步 `t`，只能看 `t' <= t`。

在本项目里，如果历史窗口组织为过去到现在：

- `[-K+1, ..., -1, 0]`

则 temporal mask 应为标准下三角 mask。

如果使用 padded history，则还要与 `frame_mask` 联合。

## 5.6 历史 token 压缩策略

这是方案成败的关键，推荐使用“分层丢弃旧帧 token”的保守实现。

### 方案 A：分阶段时间下采样

例如以 `K=6` 为例：

- 低层保留全部 6 帧
- 中层只保留 `3` 帧历史摘要 + 当前帧
- 高层只保留 `1` 个历史摘要帧 + 当前帧
- 最终只把 `当前帧 tokens + 少量历史摘要 tokens` 送给 LLM

优点：

- 最符合论文图示
- 推理速度收益明显

缺点：

- 需要设计 token merge / frame pooling 规则

### 方案 B：仅保留当前帧 full tokens，历史帧做 pooled memory tokens

例如：

- 当前帧保留全部 `N` 个 patch token
- 每个过去帧只保留 `M` 个 pooled tokens，`M << N`

可以通过：

- 平均池化
- learnable queries
- MAP pooling

得到过去帧摘要。

这是更适合 `pi0.5` 第一版上线的策略，我更推荐先做这个版本。

原因：

- 当前帧仍维持高分辨率空间细节
- 历史只提供“短期记忆补充”
- LLM prefix 长度更可控
- 更容易和现有单帧 checkpoint 对齐

## 5.7 前缀拼接策略

当前 `embed_prefix()` 是把每个相机的所有 patch token 全拼起来。

改造后建议为：

`[cam1 当前帧 tokens + cam1 history summary tokens] + [cam2 ...] + [cam3 ...] + [prompt tokens]`

并保持：

- image/video prefix tokens 与语言 prompt 之间仍是全可见
- suffix action tokens 仍沿用现有 `make_attn_mask()` 逻辑

这样可以不改动 `Gemma` 主体的 attention 机制。

## 6. 训练方案

## 6.1 第一阶段：视觉记忆编码器热启动

目标是尽量继承当前 `SigLIP` 单帧能力。

建议初始化：

- patch embedding、空间 attention、MLP 直接从现有 `SigLIP` 拷贝
- 新增 temporal attention 参数：
  - 若结构允许复用同一套 QKV，则直接共享/复制初始化
  - 若新增独立 temporal projection，则用空间 projection 拷贝初始化
- 时间位置编码初始化为固定 sin/cos，不训练或弱训练

训练策略：

- 冻结 LLM 和 action expert
- 只训练视频编码器新增部分 + 轻微解冻视觉高层

目的：

- 先让编码器学会“在不破坏当前帧语义的前提下融合历史”

## 6.2 第二阶段：与 `pi0.5` 联合微调

在现有 flow-matching 目标不变的前提下，联合训练：

- 视频编码器
- 视觉到语言的前缀路径
- 可选地微调部分 `Gemma` 层或 LoRA

损失函数可以先不改，仍使用当前：

- 动作流匹配损失

即：

- 保持 [src/openpi/models/pi0.py](/e:/code/openpi0.5/src/openpi/models/pi0.py#L188) 中 `compute_loss()` 主体不变
- 仅替换 `prefix_tokens = self.embed_prefix(observation)` 的视觉来源

这点很重要，因为短期记忆模块的核心收益本来就来自更好的条件输入，而不一定需要新增监督头。

## 6.3 可选辅助损失

若第一版发现训练不稳定，可以加两个轻量辅助目标：

1. 当前帧一致性损失
   约束 `K=1` 时视频编码器输出接近原始单帧 `SigLIP` 输出

2. 历史恢复损失
   用压缩后的历史 token 预测过去某些 patch 特征或过去帧全局 embedding

推荐先不加，只有在迁移不稳时再启用。

## 7. 数据要求

短期记忆视频编码器只有在训练数据中真正看到连续观测时才会起效。

因此训练数据侧至少需要满足：

- 每个样本能取到最近 `K` 帧，而不是只取当前帧
- 相机时间戳尽量对齐
- history 采样间隔固定，建议和控制频率一致或做轻微降采样

建议初始设置：

- base 相机: 最近 `4-6` 帧
- wrist 相机: 最近 `4-6` 帧
- 帧间隔: `1` 或 `2` control steps

如果显存压力太大，可以只先对 wrist camera 开历史，而 base camera 仍保留当前帧。

这是一个很值得尝试的低成本变体，因为许多遮挡和抓取修正主要依赖 wrist 视角。

## 8. 推理与延迟控制

论文 C 部分的价值在于把历史压缩留在视觉编码阶段，而不是把所有历史扔给 LLM。

对本项目，推理侧建议做以下优化：

### 8.1 历史帧滚动缓存

在 policy server 或 robot runtime 中维护每个相机的 ring buffer：

- 每步仅 append 当前帧
- 构造 `[B, K, H, W, C]` 输入

### 8.2 当前帧与历史帧分开算

如果实现允许，建议：

- 当前帧 full-resolution 编码每步更新
- 过去帧历史摘要在相邻步之间尽可能复用

这会比每步重算整段视频更高效。

### 8.3 prefix token 预算

推荐给 prefix 设硬预算，例如：

- 每相机 `<= 512` tokens
- 三相机总视觉 token `<= 1200-1500`

否则即使视频编码器本身高效，LLM prefix 也会重新成为瓶颈。

## 9. 分阶段实施建议

## Phase 1: 最小可用版本

目标：最小代价验证“历史视频 > 单帧”。

改动：

- `Observation.images` 支持 `[B, K, H, W, C]`
- 新增 `video_siglip.py`
- 只对 wrist 相机启用短期历史
- 历史帧只输出 pooled memory tokens
- `Pi0.embed_prefix()` 接入新视频 token

优点：

- 工程风险最低
- 最快能看到遮挡恢复和 regrasp 收益

## Phase 2: 全相机短期记忆

改动：

- base / left_wrist / right_wrist 全部接入短期历史
- 开始做 layer-wise history token dropping

目标：

- 提升全局场景追踪和多视角遮挡鲁棒性

## Phase 3: 更接近 MEM 论文完整版

改动：

- 与长时语言记忆模块联合
- 引入高层/低层策略拆分

这一步已经不只是“视频编码器优化”，而是向完整 MEM 系统演进。

## 10. 建议修改的代码点

核心文件建议如下：

- [src/openpi/models/model.py](/e:/code/openpi0.5/src/openpi/models/model.py)
  扩展 `Observation` 和 `preprocess_observation()` 以支持时间维

- `src/openpi/models/video_siglip.py`
  新增视频版视觉编码器，实现时间位置编码、时间因果注意力、空间注意力与历史压缩

- [src/openpi/models/pi0.py](/e:/code/openpi0.5/src/openpi/models/pi0.py)
  改造 `embed_prefix()`，把单帧图像编码改成视频记忆编码

- [src/openpi/models/pi0_config.py](/e:/code/openpi0.5/src/openpi/models/pi0_config.py)
  增加配置项，例如：
  - `memory_num_frames`
  - `memory_frame_stride`
  - `memory_enabled_cameras`
  - `memory_history_pool_tokens`
  - `memory_temporal_attention_layers`

- 数据变换与推理脚本
  负责构建 history buffer 和 frame mask

## 11. 配置建议

第一版推荐默认配置：

- `memory_num_frames = 4`
- `memory_frame_stride = 1`
- `memory_enabled_cameras = ("left_wrist_0_rgb", "right_wrist_0_rgb")`
- `memory_history_pool_tokens = 8`
- `memory_temporal_attention_layers = 4`
- `memory_keep_current_full_tokens = True`

原因：

- 先把收益集中在最依赖短时视觉反馈的 wrist 视角
- 当前帧保留 full patch tokens，尽量不损失空间精度
- 历史帧只保留少量摘要 token，控制 prefix 长度

## 12. 预期收益

如果按上面路线落地，`pi0.5` 应该更可能在以下场景受益：

- 夹爪或手臂遮挡目标后继续完成抓取
- 对物体滑落、错抓后的即时修正
- 需要短期计数或记住刚看到的局部信息
- 多步近场精细操作

相比之下，对“几分钟后的语义任务进度记忆”帮助会有限，因为那属于论文的长时语言记忆部分。

## 13. 风险与注意点

### 13.1 最大风险不是模型，而是 token 预算

如果历史 token 控制不好，LLM 前缀长度会爆炸，最终拖慢整机推理。

### 13.2 时间建模可能带来训练分布偏移

如果训练阶段大量使用 history，但推理阶段丢帧、缓存长度不足或相机不同步，性能可能反而下降。

因此必须保证：

- 训练和推理的 history 构造规则尽量一致
- mask 逻辑覆盖缺帧情况

### 13.3 不建议第一版就同时改视觉和动作侧

短期记忆模块本身已经是一个较大变量。建议保持 flow matching、action suffix、LLM attention 逻辑基本不动，先验证视觉记忆是否成立。

## 14. 最终建议

对当前 `pi0.5`，我最推荐的不是“把 MEM 的 C 部分完全照搬”，而是采用下面这条更稳的落地路径：

1. 先把 `Observation` 扩展成支持短视频输入
2. 新增一个兼容 `SigLIP` 权重的视频编码器
3. 采用“当前帧 full tokens + 历史帧 pooled memory tokens”的压缩策略
4. 先只给 wrist 相机开启短期记忆
5. 保持 `Gemma` 和 flow-matching action head 基本不变
6. 在验证遮挡恢复、regrasp、近场操控收益后，再推进到全相机和分层 token dropping

这条路线最符合论文的理论核心，也最适合当前仓库的结构。

## 15. 一句话结论

MEM 论文 C 部分最值得迁移到 `pi0.5` 的，不是“加更多历史帧”，而是“在视觉编码器内部用可分离的因果时间注意力压缩短期历史，再把少量高价值历史 token 交给 VLM”。

对本项目，最佳实施方式是把它做成 `SigLIP` 前面的最小侵入式视频记忆编码层，而不是直接改 `Gemma` 主干。
