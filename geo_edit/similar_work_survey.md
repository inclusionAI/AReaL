# 同期工作调研与差异化计划 — Agentic RL for Visual Reasoning

> 调研时间: 2026-03-30 | 覆盖范围: 2024-2026 arXiv, NeurIPS'25, ICLR'26, CVPR'25/26, ACL'25, EMNLP'25, AAAI'25/26
>
> **调研方法**: 6 个并行 librarian agent + 7 次 Exa web search, 覆盖 arXiv/OpenReview/GitHub/Semantic Scholar
>
> **引用说明**: 所有条目标注 arXiv ID 或会议信息; 标注"unverified"的条目缺乏可验证链接, 待后续确认
>
> **会议状态说明**: 标注为 "ICLR 2026" 的条目来自 OpenReview/arXiv 自标注, 部分可能为 under review 或已撤回; 使用前请在 OpenReview 上核实最终接收状态
>
> **差异化声明标准**: 本文中"未发现同期工作"指在上述检索范围内、满足"tool-augmented + RL training + public benchmark"三个条件下未找到匹配项, 不排除存在未公开或在投工作
>
> **文件约束**: 本文档为新建文件, 未修改任何已有文件

---

## 一、同期工作全景（按类别）

### A. 直接竞品：VLM + 视觉工具 + RL 训练

| # | 工作 | 会议/日期 | 核心贡献 | RL算法 | 工具类型 | 基座模型 | 与我们的重叠度 |
|---|------|----------|---------|--------|---------|---------|-------------|
| 1 | **VISTA-R1 / Mini-o3** | arXiv 2511.19773 | VISTA-Gym 统一训练环境 + 7 任务 13 数据集; GRPO 多轮训练 | GRPO | grounding, parsing (2-3个) | InternVL3-2B, Qwen2.5-VL | ⚠️ **高**: 同为 VLM + tool + GRPO |
| 2 | **VTool-R1** | ICLR 2026 | 首个训练 VLM 用图像工具做多模态 CoT 的框架; Python visual editing | GRPO | crop, rotate 等编辑工具 | 未公开 | ⚠️ **高**: 同为视觉工具 RL |
| 3 | **PyVision-RL** | arXiv 2602.20739 | 解决 "interaction collapse"（模型学会减少工具调用）; 累积工具奖励 | 自定义 RL | 视频帧采样, 图像理解 | 开源 VLM | ⚠️ **中高**: interaction collapse 是我们也会遇到的问题 |
| 4 | **OpenThinkIMG** | arXiv 2505.08617 | 首个开源端到端工具增强 LVLM 框架; 标准化工具接口 + 轨迹生成 + RL 训练 | RL | 标准化视觉工具 | LVLM | ⚠️ **高**: 框架定位接近 |
| 5 | **VisTA** (VisualToolAgent) | arXiv 2505.20289 | GRPO 训练 VLM 动态选择和组合工具 | GRPO | 多样工具库 | 未公开 | ⚠️ **中**: 侧重工具选择 |
| 6 | **ReVPT** | arXiv 2509.01656 | GRPO 训练 VLM 使用 4 个视觉工具; 3B/7B 模型 | GRPO变体 | 4 个固定工具 | 3B, 7B | ⚠️ **中**: 工具数量少 |
| 7 | **CodeV** | arXiv 2511.19661 | TAPO (Tool-Aware Policy Optimization): 过程级 RL + Python code 工具 | TAPO | Python 可执行工具 | 未公开 | ⚠️ **中**: 侧重代码工具 |
| 8 | **SpaceTools** | arXiv 2512.04069 | DIRL (Double Interactive RL): 深度/分割/姿态工具协调 | DIRL | 深度/分割/姿态 | 未公开 | ⚠️ **中**: 空间推理专用 |
| 9 | **Visual-ARFT** | arXiv 2505.14246 | 开源 LVLM 浏览网页+写代码处理图像; MAT 基准 | RFT | 网页浏览, 图像代码 | 开源 LVLM | ⚠️ **中低**: 侧重网页 |
| 10 | **Agent0-VL** | arXiv 2511.19900 | 自进化 VLM agent; Solver + Verifier 统一; 零外部奖励 | Self-evolving RL | 工具集成推理+验证 | 未公开 | ⚠️ **中**: 自进化方向 |

### B. 多轮工具 RL 框架（方法论竞品）

| # | 工作 | 会议/日期 | 核心贡献 | 与我们的关联 |
|---|------|----------|---------|------------|
| 11 | **ToolOrchestra** (NVIDIA) | arXiv [2511.21689](https://arxiv.org/abs/2511.21689), ICLR'26 under review | 端到端 RL 训练工具编排; 8B 模型管理多工具 | 框架参考 |
| 12 | **RLFactory** | arXiv 2509.06980 | 即插即用多轮工具 RL 训练框架 | 训练框架参考 |
| 13 | **Verl-Tool** | ICLR 2026 under review | 基于 verl 的全面 agentic RL + tool use | 框架参考 |
| 14 | **RC-GRPO** | arXiv 2602.03025 | 奖励条件化 GRPO; 离散奖励 token 控制探索 | 训练算法参考 |
| 15 | **ToRL** | arXiv 2503.23383 | 工具集成 RL scaling | 训练方法参考 |
| 16 | **ToolRL** | NeurIPS 2025 | 工具调用的奖励设计原则 | Reward 设计参考 |
| 17 | **Search-R1** | COLM 2025 | 搜索引擎 + 推理 RL; GRPO | 工具 RL 参考 |
| 18 | **AgentRL** | ICLR 2026 under review | 多轮多任务 agentic RL 扩展框架 | Scaling 参考 |

### C. 视觉推理 RL（无外部工具）

| # | 工作 | 会议/日期 | 核心贡献 | 与我们的关联 |
|---|------|----------|---------|------------|
| 19 | **One RL to See Them All** (MiniMax) | arXiv 2025 | 视觉三元统一 RL | 纯 RL baseline |
| 20 | **Pixel Reasoner** | NeurIPS 2025 | 好奇心驱动 RL 做像素级推理 | 像素级推理对比 |
| 21 | **OpenVLThinker** | ICLR 2026 under review | 迭代 SFT-RL 循环 | 训练方法参考 |
| 22 | **Vision-Zero** | ICLR 2026 | 自对弈游戏, 零标注 VLM 自改善 | 自改善方向 |
| 23 | **VALOR** | ICLR 2026 | 多模态验证器训练推理器, 无标签 | 验证器参考 |
| 24 | **R1-ShareVL** | arXiv 2025 | Share-GRPO 扩展问题空间 | GRPO 改进参考 |
| 25 | **Reason-RFT** (PKU) | NeurIPS 2025 | RL 微调视觉推理泛化 | 训练方法参考 |
| 26 | **PROPA** | arXiv 2025 | MCTS + GRPO 过程级密集奖励 | 奖励设计参考 |

### D. 交互式视觉思维 / Think-with-Images

| # | 工作 | 会议/日期 | 核心贡献 | 与我们的关联 |
|---|------|----------|---------|------------|
| 27 | **V-Thinker** | arXiv 2025 | 交互式图像思维 + Data Evolution Flywheel | 思维方式参考 |
| 28 | **ThinkMorph** | ICLR 2026 | 多模态交叉 CoT 的涌现特性 | 理论参考 |
| 29 | **Reinforcing Spatial Reasoning** | arXiv 2025 | 交织思考 + 视觉绘画 | 空间推理参考 |
| 30 | **VR-Thinker** | arXiv 2026 | 视频奖励模型 + 图像思维 | 视频方向参考 |

### E. 世界模型 / 世界推理

| # | 工作 | 会议/日期 | 核心贡献 | 与我们的关联 |
|---|------|----------|---------|------------|
| 31 | **VAGEN** (Stanford/Microsoft) | NeurIPS 2025 | RL 训练 VLM 构建世界模型; POMDP 建模; 世界建模奖励 | ⚠️ **高**: 同为 world reasoning |
| 32 | **Visual Generation Unlocks Reasoning** (THU/ByteDance) | arXiv 2026 | 视觉生成解锁推理能力 | 世界模型参考 |
| 33 | **WorldVQA** (Moonshot AI) | GitHub 2026 | 世界理解基准 | 评测参考 |
| 34 | **Think3D** | arXiv 2026 | 空间思维做空间推理 | 空间推理参考 |
| 35 | **SpatialThinker** (Oxford) | arXiv 2025 | 空间奖励强化 3D 推理 | 空间奖励参考 |
| 36 | **Embodied-R** (THU) | arXiv 2025 | 协作框架激活具身空间推理 | 空间推理参考 |

### F. 领域特定 VLM Agent

| # | 工作 | 会议/日期 | 核心贡献 | 与我们的关联 |
|---|------|----------|---------|------------|
| 37 | **ChartAgent** | arXiv 2510.04514 | 图表 QA 多模态 agent | 图表方向竞品 |
| 38 | **Chart-R1 / BigCharts-R1** | ICLR/COLM 2025 | RL + CoT 做图表推理 | 图表方向竞品 |
| 39 | **VProChart** | AAAI 2025 | 视觉感知 + 程序化推理做图表 QA | 图表方向竞品 |
| 40 | **ORCA** | arXiv 2026 | 多 agent 协作做文档 QA | 文档方向参考 |
| 41 | **MDocAgent** | ICLR 2026 | 多模态多 agent 文档理解 | 文档方向参考 |
| 42 | **MapAgent** | arXiv 2025 | 层级 agent + 动态地图工具 | 地图方向竞品 |
| 43 | **4KAgent** | NeurIPS 2025 | Agentic 超分辨率 | 图像处理参考 |
| 44 | **ARM-Thinker** | arXiv 2025 | Agentic 奖励模型 + 工具调用 | 奖励模型参考 |

### G. 中国实验室重点工作

| # | 团队 | 工作 | 核心贡献 |
|---|------|------|---------|
| 45 | ByteDance Seed | **UI-TARS-2** | 多轮 RL 训练 GUI agent |
| 46 | Tencent | **Insight-V** (CVPR 2025) | 长链视觉推理 |
| 47 | Tencent | **GTR** (ICCV 2025) | 防止 RL 训练中思维崩溃 |
| 48 | Tencent | **MindOmni** (NeurIPS 2025) | RGPO 增强推理生成 |
| 49 | SenseTime | **SenseNova-MARS** | Agentic 推理 + 搜索 via RL |
| 50 | PKU | **Reason-RFT** (NeurIPS 2025) | RL 微调视觉推理泛化 |
| 51 | ZJU | **SpatialLadder** (ICLR 2026) | 渐进式空间推理训练 |
| 52 | Baidu | **Perception Before Reasoning** | 两阶段 RL: 感知→推理 |

### H. 补充：重要遗漏工作

| # | 工作 | 会议/日期 | 核心贡献 | 与我们的关联 |
|---|------|----------|---------|------------|
| 53 | **SPORT** | NeurIPS 2025, arXiv [2504.21561](https://arxiv.org/abs/2504.21561) | 多模态 agent 通过步级偏好调优迭代探索工具使用策略; 无需预采集数据 | ⚠️ **中高**: 步级偏好优化与我们的 process-level tool reward 方向相近 |
| 54 | **ViGoRL** | arXiv [2505.23678](https://arxiv.org/abs/2505.23678) | 多轮 RL 训练 VLM 显式锚定每步推理到视觉坐标; 动态 zoom | ⚠️ **中**: grounded reasoning 与我们的工具调用有交叉 |
| 55 | **IMAgent** | arXiv [2512.08980](https://arxiv.org/abs/2512.08980) | 多图像视觉 agent; 视觉反思/确认工具; 动作-轨迹两级 mask (代码未开源) | ⚠️ **中**: 多图像 agent 方向 |
| 56 | **VG-Refiner** | arXiv [2512.06373](https://arxiv.org/abs/2512.06373) | 工具精炼参考推理; think-rethink 两阶段处理不可靠工具输出 | ⚠️ **中**: 工具可靠性问题相关 |
| 57 | **Argos** (Microsoft) | arXiv [2512.03438](https://arxiv.org/abs/2512.03438) | 多模态 RL + agentic verifier; 选择评分函数池 | ⚠️ **中**: 验证器设计参考 |
| 58 | **Look Less, Reason More** | arXiv [2510.01681](https://arxiv.org/abs/2510.01681) | 自适应像素空间推理; 仅 20.1% 工具使用率达 73.4% HR-Bench 4K | ⚠️ **中**: 自适应工具调用频率 |
| 59 | **GRIT** | NeurIPS 2025 | 教 MLLM "用图像思考"; 视觉推理中间步 | ⚠️ **中**: thinking-with-images 方向 |
| 60 | **VisionReasoner** | GitHub [dvlab-research](https://github.com/dvlab-research/VisionReasoner) | 统一推理集成视觉感知 via RL | ⚠️ **中**: 视觉推理+RL |
| 61 | **DeepEyes** | arXiv 2025 | 视觉搜索/放大; Mini-o3 的训练数据来源之一 | ⚠️ **低**: 工具而非训练方法 |

### I. 重要开源项目/框架（非论文但会被直接对比）

| # | 项目 | GitHub Stars | 说明 | 与我们的关联 |
|---|------|-------------|------|------------|
| 62 | **[VLM-R1](https://github.com/om-ai-lab/VLM-R1)** | 5.8k | R1-style VLM reasoning with RL; 社区最活跃的开源 VLM-RL 项目 | 开源 baseline 对比 |
| 63 | **[verl](https://github.com/verl-project/verl)** | 20k+ | 生产级 RL 训练框架, 支持 PPO/GRPO/DPO + tool calling | 我们用 AReaL, 但 verl 是行业标准 |
| 64 | **[Agent-R1](https://github.com/AgentR1/Agent-R1)** | 1.3k | 端到端 agent RL 训练 | 开源 agent RL baseline |
| 65 | **[Awesome Think-With-Images](https://github.com/zhaochen0110/Awesome_Think_With_Images)** | 1.4k | "Thinking with Images" 论文汇总列表 (非 OpenThinkIMG 代码仓库) | 追踪新工作 |
| 66 | **[ToolOrchestra](https://github.com/NVlabs/ToolOrchestra)** | 690 | NVIDIA 的端到端工具 RL 训练框架 | 框架对比 |
| 67 | **[Pixel-Reasoner](https://github.com/TIGER-AI-Lab/Pixel-Reasoner)** | 285 | 像素级推理 RL | 推理方法对比 |

---

## 二、重叠度分析 — 哪些方向已经很拥挤

### 🔴 高重叠（竞争激烈，需要差异化）

| 方向 | 竞品数量 | 代表工作 | 我们原计划中的对应 |
|------|---------|---------|------------------|
| **VLM + 视觉工具 + GRPO** | 10+ | VISTA-R1, VTool-R1, OpenThinkIMG, VisTA, ReVPT | 整个项目核心 |
| **图表 QA + RL/Agent** | 5+ | ChartAgent, Chart-R1, BigCharts-R1, VProChart | ChartQA 数据采集 |
| **SFT → RL 两阶段训练** | 8+ | Mini-o3, OpenVLThinker, Reason-RFT | SFT Cold Start → GRPO |
| **format + correctness reward** | 5+ | Mini-o3, PROPA, ToolRL | Reward 设计 |

### 🟡 中等重叠（有竞争但仍有空间）

| 方向 | 竞品数量 | 代表工作 | 我们的差异化空间 |
|------|---------|---------|---------------|
| **世界推理 RL** | 3 | VAGEN, WorldVQA | 我们有更丰富的工具辅助 world reasoning |
| **空间推理 RL** | 4 | SpatialThinker, SpaceTools, SpatialLadder | 我们有 SAM3+地图工具 |
| **多轮工具编排** | 3 | ToolOrchestra, RLFactory, Verl-Tool | 我们有 14+ 工具 vs 他们 3-4 个 |

### 🟢 低重叠（蓝海机会）

| 方向 | 竞品数量 | 说明 | 机会 |
|------|---------|------|------|
| **地图推理 + RL agent** | 1 (MapAgent) | MapAgent 仅推理无 RL 训练; 未检索到同时满足 tool-RL + map reasoning + public benchmark 的工作 | **我们有 5 个 cartomapqa 子任务 + 地图专用工具** |
| **视觉计数 + 工具 agent** | ~0 | 未检索到用 agentic tool-calling + RL 方法做精确计数的工作 (CountGD/LVLM-Count 是非 agentic 方法) | **SAM3 text prompt + count = 独特工具组合** |
| **文档理解 + RL 训练** | ~0 | 有文档 agent (ORCA/MDocAgent) 但均无 RL 训练 | 可考虑但非核心 |
| **Interaction collapse 防护** | 2-3 (PyVision-RL, SPORT, CodeV/TAPO 相邻) | PyVision-RL 直接研究 collapse; SPORT 的步级偏好和 CodeV 的 TAPO 过程级奖励是相邻工作 | **系统性分析 + 多工具场景下的防护方案** |
| **BabyVision 基础视觉推理** | ~0 | 未检索到用 tool-augmented RL 攻克 BabyVision 的工作 (BabyVLM 侧重训练效率非工具) | **独特难度和角度** |
| **多工具协调 (14+)** | ~0 | 检索到的竞品最大工具规模约 4-7 个; 14+ 异构工具(函数+GPU Agent)的 RL 训练尚未见报道 | **我们 14+ 工具 + Ray 分布式是最大差异** |

---

## 三、基于调研的升级版工作计划

### 定位调整

| 方向 | 定位 | 原因 |
|------|------|------|
| **Map Reasoning (地图推理)** | **🔥 核心贡献方向 1** | 蓝海; MapAgent 仅推理无 RL; 我们有 5 个子任务 + 地图专用工具 + Map-to-Graph 新工具 |
| **World Modeling & Reasoning** | **🔥 核心贡献方向 2** | VAGEN 仅用环境动作无视觉工具; 我们用外部工具做显式世界建模; BabyVision/VisWorld 无人攻克 |
| **ChartQA** | **保留做对比, 非核心贡献** | 5+ 竞品已覆盖, 但作为工具 RL 的标准评测任务, 需要保留结果以与 VISTA-R1/VTool-R1 等对比 |
| 基础 SFT→GRPO 两阶段训练 | 降为工程手段, 非贡献点 | Mini-o3, OpenVLThinker 等 8+ 工作已做 |
| format + correctness reward 设计 | 降为工程手段, 非贡献点 | Mini-o3, ToolRL 等已有成熟方案 |
| 简单的图文交替工具调用 | 降为 baseline, 非创新 | VTool-R1, OpenThinkIMG 已做 |

### 强化 / 新增的方向（核心贡献点）

> **研究主线**: 用外部视觉工具作为显式世界模型组件, 通过 RL 训练 VLM 学会调用这些工具, 解决地图推理和世界推理中 SOTA 模型尚不能解决的难题。

---

#### 🔥 贡献 1 (核心): Map Reasoning — 工具增强的地图推理 RL

**竞争格局**: 地图推理是蓝海。MapAgent 仅推理无 RL 训练; 无人在地图任务上做 tool-augmented RL。

**核心创新 — Map-to-Graph 工具**:

将视觉地图转化为可查询的结构化图，把"视觉空间推理"降维为"图算法"：
```
地图图像 → [Map-to-Graph Tool] → NetworkX 图 → shortest_path(A,B) / neighbors(X) / count_nodes()
```

**三类地图的图提取策略**:
| 地图类型 | 提取方法 | 图结构 | 对应任务 |
|---------|---------|--------|---------|
| 行政区划图 | 颜色分割 + OCR → 多边形邻接 | 区域邻接图 | CartoMapQA-Presence/Counting |
| 道路网络图 | Text spotting + 道路线检测 + 连通性 BFS | 路网图 (加权) | CartoMapQA-SRN, MapEval |
| 合成网络图 | SAM3 节点分割 + OCR + 线检测 | 带权图 | Shortest Path |

**其他地图推理工具**:
| 工具 | 功能 | 解决的子问题 |
|------|------|------------|
| SAM3 精确计数 | `count("all hospitals on the map")` → 分割 → count masks | STMF Counting (VLM 数不过 20) |
| 地图 OCR 定位 | text_spotting → 提取地名 + 坐标 | STMF Name Listing |
| 空间关系查询 | 基于图结构的邻接/距离/路径查询 | SRN 路线导航 |

**差异化行动**:
- [ ] 实现 **Map-to-Graph 工具** (VLM 辅助 + CV 验证的混合方案, 优先支持 3 类地图)
- [ ] 实现 **SAM3 精确计数工具**: text prompt 分割 → count
- [ ] **CartoMapQA 5 子任务全覆盖**: SRN + Counting + Presence + Name Listing + STMF (已有适配器)
- [ ] **MapEval Visual** + **Shortest Path** 评测
- [ ] 对比: 纯 VLM → VLM + 工具 (无 RL) → VLM + 工具 + RL 的阶梯提升
- [ ] 展示: Map-to-Graph 工具让地图推理从"视觉猜测"变为"确定性图算法", 准确率质变

---

#### 🔥 贡献 2 (核心): World Modeling & Reasoning — 外部工具作为显式世界模型

**竞争格局**:
- **VAGEN** (NeurIPS 2025) 是最接近的竞品, 但其关键局限: VAGEN 的 "工具" 是环境交互动作 (GUI 点击/游戏移动), **不是视觉处理工具**。它的世界模型完全靠 VLM 内部隐式推理, 没有外部工具辅助
- **现有 tool-RL 工作** (VISTA-R1, VTool-R1 等) 用视觉工具, 但不面向世界推理任务
- **空白地带**: 无人用 "视觉工具 + RL" 组合攻克 BabyVision/VisWorld-Eval

**核心思路**: VLM 的内部世界模型不可靠（BabyVision 上 Gemini 3 Pro 仅 49.7 vs 人类 94.1）。用 **外部工具作为显式世界模型组件**, 把隐式推理外化为可操作的结构：

| 世界建模能力 | VLM 内部推理的问题 | 外部工具的解决方案 |
|------------|------------------|-----------------|
| 物体识别与追踪 | 遗忘被遮挡物体 | SAM3 分割 + Object State Tracker |
| 空间结构理解 | 空间关系推理不可靠 (tunnel vision) | **Scene Graph Builder** → 显式空间关系图 |
| 状态变化检测 | 无法精确检测细微变化 | **Visual Diff** → 差异 mask + 状态变化描述 |
| 计数与度量 | 数不过 20, 距离估计差 | SAM3 count + 深度估计 |

**新工具设计** (超越现有工具):
| 新工具 | 功能 | 实现方案 | 覆盖任务 |
|--------|------|---------|---------|
| **Scene Graph Builder** | 图像 → `{entities, spatial_relations}` | Grounding DINO + 关系预测 / LLM 从分割结果提取 | BabyVision, VisWorld-Eval |
| **Visual Diff** | (img_before, img_after) → 差异描述 + mask | 图像差分 + SAM3 分割 + VLM 描述 | VisWorld-Eval 场景变化 |
| **Object State Tracker** | 多帧追踪物体状态变化 | SAM3 + 特征匹配 + 状态分类 | BabyVision 物体永恒性 |

**差异化行动**:
- [ ] 实现 **Scene Graph Builder**: 图像 → 结构化空间关系图 (可查询)
- [ ] 实现 **Visual Diff**: 两图对比 → 精确变化检测
- [ ] 引入 VAGEN 的 **World Modeling Reward** 思路 (State Estimation + Transition Modeling), 但扩展到视觉工具域:
  - R_state = 工具辅助的状态估计准确度 (Scene Graph 与 ground truth 的匹配)
  - R_transition = 状态转移预测的准确度 (Visual Diff 检测到的变化与预期是否一致)
- [ ] **BabyVision 评测**: 据我们检索, 无人用 tool-augmented RL 攻克此基准
- [ ] **VisWorld-Eval 评测**: 对比 纯推理 vs 工具辅助 vs 工具+RL
- [ ] 分析: 哪些世界推理子能力从外部工具获益最大 (物体永恒性 >> 因果推理?)

---

#### 贡献 3 (支撑): 大规模多工具协调 + Anti-Collapse

**定位**: 支撑贡献 1 和 2 的方法论贡献。解决 14+ 工具 RL 训练的独特挑战。

**研究问题**: 工具从 3 扩展到 14+ 时:
- 工具选择空间爆炸 → Tool Selection Reward (Jaccard on reference tool set)
- interaction collapse (模型学会偷懒不调工具) → Process-Level Tool Reward
- 异构工具协调 (图像返回 vs 文本返回 vs GPU Agent) → Heterogeneous Reward

**相邻工作**: PyVision-RL (累积工具奖励), CodeV/TAPO (过程级 RL, 仅 Python code), SPORT (步级偏好)

**差异化行动**:
- [ ] **Tool Scaling Law 实验**: Subset-3 (3 tools) vs Subset-7 (7 tools) vs Full-14+ 对比
- [ ] **Process-Level Tool Reward**: 针对异构工具 (图像/文本/GPU Agent) 的分类过程奖励
- [ ] **Collapse 分析**: 何时发生 (训练步数, 工具数量), 如何检测, 如何恢复

---

#### 贡献 4 (工程): AReaL VisionMultiTurnToolWorkflow

**定位**: 使上述实验可行的工程基础。

**差异化行动**:
- [ ] `VisionMultiTurnToolWorkflow`: 融合 VisionRLVR + MultiTurn + 工具执行
- [ ] 支持 Ray Actor GPU 工具 Agent (SAM3, Chart-R1 等)
- [ ] 支持 14+ 工具并行调度 + Map-to-Graph / Scene Graph Builder 等新工具

---

## 四、升级版时间线

### Phase 0: 核心工具开发 + 数据准备（1-2 周）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| **Map-to-Graph 工具实现** | P0 | 🔥 **贡献 1 核心**: 3 类地图的图提取 (行政区划/道路网络/合成网络) |
| **SAM3 text prompt 分割 + 计数工具** | P0 | 🔥 **贡献 1 核心**: 精确计数 |
| **Scene Graph Builder 工具** | P0 | 🔥 **贡献 2 核心**: 图像→空间关系图 |
| CartoMapQA 全子任务数据采集 | P0 | 地图推理轨迹 (Gemini 3 Pro, 多工具) |
| BabyVision/VisWorld-Eval 数据采集 | P0 | 世界推理轨迹 (Gemini 3 Pro) |
| ChartQA 数据采集 | P1 | 保留做对比 (2,500 条, Gemini 3 Flash) |
| Visual Diff 工具实现 | P1 | 贡献 2 补充: 状态变化检测 |
| ZwZ-RL-VQA 数据集成 | P1 | 74K pairs 精细感知训练 |

### Phase 1: 训练框架 + 首轮实验（2-4 周）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| VisionMultiTurnToolWorkflow | P0 | **贡献 4**: AReaL workflow 对接 |
| SFT Cold Start | P0 | 工程手段 |
| **首轮 GRPO: Map Reasoning** | P0 | 🔥 **贡献 1 验证**: CartoMapQA + MapEval + Shortest Path |
| **首轮 GRPO: World Reasoning** | P0 | 🔥 **贡献 2 验证**: BabyVision + VisWorld-Eval |
| Process-Level Tool Reward | P0 | **贡献 3**: 防 collapse |
| ChartQA GRPO 训练 | P1 | 对比实验用: 与 VISTA-R1/Chart-R1 对比 |

### Phase 2: 深度实验 + 论文（4-8 周）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| **Map Reasoning 消融实验** | P0 | 🔥 Map-to-Graph 工具的各组件贡献分析 |
| **World Reasoning 消融实验** | P0 | 🔥 哪些世界建模子能力从外部工具获益最大 |
| Tool Scaling Law 实验 | P0 | **贡献 3**: 3 vs 7 vs 14+ tools 对比 |
| Interaction Collapse 分析 | P0 | **贡献 3**: 系统性分析+解决方案 |
| Scaling up (32B, 更多轮数) | P1 | 扩展性验证 |
| Vision-Zero self-play 探索 | P2 | 长期方向 |

---

## 五、核心贡献总结（论文故事线）

> **标题方向**: "External Tools as Explicit World Models: Solving Unsolved Map Reasoning and World Modeling via Tool-Augmented RL"

### Story:

1. **观察**: 现有 VLM 在困难的视觉推理任务上表现极差:
   - 地图推理: VLM 看地图数不对、找不到路、辨不清空间关系
   - 世界建模: BabyVision 上 Gemini 3 Pro 仅 49.7 vs 人类 94.1, 物体永恒性/因果推理崩溃
   - 现有 tool-RL 工作 (VISTA-R1, VTool-R1) 用 3-7 个通用工具在标准 VQA 上提升, 但未触及这些 unsolved 难题

2. **关键 insight**: VLM 的失败根源是**缺乏可靠的世界模型** — 它不能精确感知空间结构、追踪状态变化、执行确定性空间推理。而 VAGEN 等世界模型工作只用 VLM 内部隐式推理, 同样不可靠。
   - **我们的方案**: 用**外部视觉工具作为显式世界模型组件** — Map-to-Graph 把地图转为图结构 (确定性空间推理), Scene Graph Builder 把场景转为关系图 (显式状态表示), Visual Diff 检测状态变化 (显式转移建模)

3. **方法**:
   - **Map-to-Graph**: 视觉地图 → 结构化图 → 图算法查询 (shortest_path, neighbors, count)
   - **Scene Graph Builder**: 图像 → 空间关系图 → 可查询的世界状态表示
   - **Visual Diff + State Tracker**: 状态变化检测, 物体追踪
   - **14+ 异构工具**的统一 RL 训练 + Process-Level Tool Reward 防 collapse
   - 基于 AReaL 的 VisionMultiTurnToolWorkflow

4. **实验**:
   - 🔥 **Map Reasoning**: CartoMapQA 5 子任务, MapEval, Shortest Path — Map-to-Graph 让准确率质变
   - 🔥 **World Reasoning**: BabyVision, VisWorld-Eval — 外部工具显式世界模型 vs VAGEN 风格隐式推理
   - **对比基准**: ChartQA — 与 VISTA-R1/VTool-R1/Chart-R1 对比, 确认在标准任务上不退化
   - **方法贡献**: Tool Scaling Law (3→7→14), Interaction Collapse 分析

5. **贡献**:
   - ✅ **Map-to-Graph**: 据我们检索, 将视觉地图推理转化为图算法的 tool-augmented RL 方法
   - ✅ **外部工具作为显式世界模型**: 与 VAGEN 的隐式世界模型形成互补, 在工具可介入的场景 (空间/计数/状态追踪) 上显著更强
   - ✅ 据我们检索, 在 BabyVision/CartoMapQA 上应用 tool-augmented RL 的开创性工作
   - ✅ Tool Scaling Law + Interaction Collapse 的系统性分析

---

## 六、关键竞品对比表（精简版, 适合论文 Related Work）

| 维度 | **Ours** | VISTA-R1 | VTool-R1 | VAGEN | PyVision-RL | MapAgent |
|------|------|----------|----------|-------|-------------|---------|
| **核心目标** | Map+World reasoning | 标准 VQA | 图表推理 | 世界模型推理 | 通用视觉理解 | 地图推理 |
| 工具数量 | **14+ 含新工具** | 3 | ~5 | 环境动作 (非视觉工具) | ~3 | ~4 |
| **Map-to-Graph** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Scene Graph Builder** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 世界模型类型 | **外部显式** (工具) | 无 | 无 | 内部隐式 (VLM) | 无 | 无 |
| RL 训练 | ✅ GRPO | ✅ GRPO | ✅ GRPO | ✅ 自定义 RL | ✅ 自定义 | ❌ 仅推理 |
| Anti-Collapse | ✅ 多级奖励 | ❌ | ❌ | ❌ | ✅ 累积奖励 | N/A |
| 地图推理 | ✅ **5 子任务** | ❌ | ❌ | ❌ | ❌ | ✅ 但无 RL |
| BabyVision | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| ChartQA | ✅ (对比用) | ✅ | ✅ | ❌ | ❌ | ❌ |
| 分布式工具 | ✅ Ray Actor | ❌ | ❌ | ❌ | ❌ | ❌ |

---

---

## 七、实验设计细节（使计划可执行）

### 核心实验 1: Map Reasoning 消融

| 实验 | 设置 | 评测 | 期望结果 |
|------|------|------|---------|
| **No-Tool baseline** | Qwen2.5-VL-7B 直接推理 | CartoMapQA 5 子任务 | 低 (VLM 空间推理不可靠) |
| **+ Map-to-Graph (无 RL)** | SFT 轨迹中包含 Map-to-Graph 调用 | 同上 | 中 (工具本身带来提升) |
| **+ Map-to-Graph + RL** | GRPO 训练 agent 学习何时/怎么用 | 同上 | 🔥 高 (RL 学会最优调用策略) |
| **Map-to-Graph 组件消融** | 分别去掉图提取/图查询/OCR 验证 | 同上 | 判断各组件贡献 |
| **Graph 类型消融** | 不同地图类型用不同/统一的提取策略 | 分 SRN/Counting/Presence | 判断类型特化 vs 通用方案 |

### 核心实验 2: World Reasoning 消融

| 实验 | 设置 | 评测 | 期望结果 |
|------|------|------|---------|
| **No-Tool baseline** | Qwen2.5-VL-7B 直接推理 | BabyVision (388), VisWorld-Eval | 低 (Gemini Pro 也仅 49.7) |
| **+ Scene Graph (无 RL)** | SFT 中使用 Scene Graph Builder | BabyVision 空间子类 | 中 |
| **+ Scene Graph + RL** | GRPO 训练 | 同上 | 🔥 高 |
| **+ Visual Diff (无 RL)** | SFT 中使用 Visual Diff | VisWorld-Eval 变化检测类 | 中 |
| **+ Visual Diff + RL** | GRPO 训练 | 同上 | 🔥 高 |
| **VAGEN 风格 baseline** | 隐式世界模型 (无外部工具, 纯 CoT) + RL | 同上 | 中低 (内部推理不够) |
| **子能力分析** | 按 BabyVision 22 子类分别评测 | 每个子类 | 判断哪些子能力从工具获益最大 |

### 对比实验: ChartQA (验证不退化)

| 实验 | 评测 | 对比对象 |
|------|------|---------|
| Ours (14+ tools) | ChartQA test-human, relaxed acc | Chart-R1, BigCharts-R1, VISTA-R1 |

### Tool Scaling Law 实验设计

| 工具子集 | 包含工具 | 工具数 | 说明 |
|---------|---------|-------|------|
| **Subset-3** | crop, ocr, general_vqa | 3 | 与 VISTA-R1 规模对齐的 baseline |
| **Subset-7** | + chart_r1, sam3_segment, grounding_dino, highlight | 7 | 中等规模 |
| **Subset-10** | + map_to_graph, scene_graph_builder, visual_diff | 10 | 加入新工具 |
| **Full-14+** | + depth_estimate, map_ocr, math_solver, g_llava 等 | 14+ | 完整工具库 |

**控制变量**: 相同基座模型 (Qwen2.5-VL-7B), 相同混合训练数据, 相同 GRPO 超参
**度量指标**: (1) 各基准准确率 (2) 工具调用次数/分布 (3) 训练稳定性 (4) collapse 时间点

### 评测基准

| 基准 | 角色 | Split | 样本数 | 评测方法 |
|------|------|-------|--------|---------|
| **CartoMapQA-SRN** | 🔥 核心 | test | ~200 | 路线结构匹配 |
| **CartoMapQA-Counting** | 🔥 核心 | test | ~200 | 数值精确匹配 |
| **CartoMapQA-Presence** | 🔥 核心 | test | ~200 | Yes/No 匹配 |
| **CartoMapQA-NameListing** | 🔥 核心 | test | ~200 | 列表匹配 |
| **BabyVision** | 🔥 核心 | full | 388 | 选择题/开放式 |
| **VisWorld-Eval** | 🔥 核心 | 多 split | 按 split | 选择题匹配 |
| **MapEval Visual** | 核心 | test | ~400 | 选择题匹配 |
| **Shortest Path** | 核心 | test | 500+ | 路径精确匹配 |
| ChartQA | 对比用 | test-human | ~6K | relaxed accuracy |
| ChartQAPro | 对比用 | test | ~2K | relaxed accuracy |

### Baselines 定义

| Baseline | 说明 |
|----------|------|
| **No-Tool** | Qwen2.5-VL-7B, 纯 CoT 推理 |
| **SFT-Only** | SFT on 正确轨迹 (含工具调用), 无 RL |
| **Outcome-Only RL** | GRPO + format + correctness reward (Mini-o3 配置) |
| **VISTA-R1 style** | 3 tools + GRPO |
| **VAGEN style** | 隐式世界模型 + RL (无外部视觉工具) |
| **Ours (Full)** | 14+ tools (含新工具) + GRPO + process-level reward |

---

*注: 本文档基于 2026-03-30 的调研结果, 领域发展迅速, 建议每 2-4 周更新。*
