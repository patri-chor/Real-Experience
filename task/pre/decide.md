# 研究决策与启示记录 (Research Decisions & Insights)

> **定位说明**：本文档主要用于记录研究过程中的启示性观点、技术构想与关键决策，为后续的代码构建、实验路线梳理及论文总结提供直接参考与追溯依据，非复杂的项目管理文档。

---

## 📌 记录规则（极简）

每次记录或更新决策时，仅需遵循以下核心要点：
1. **决策/修改时间**：注明具体的记录或修订时间（格式：`YYYY-MM-DD HH:MM`）。
2. **决策原因与依据（Why）**：阐述背后依据（文献启发、理论分析或痛点考量），说明为什么这么定。
3. **启示与代码/实验指导（Insights & How）**：提炼关键认知，直接指导后续代码模块的编写与实验路线设计。

---

## 📝 决策详录

### DEC-20260905-01: 参考 ExGRPO 制定验证策略与边界题筛选范式

- **决策/修改时间**：`2026-09-05 10:50:00 (UTC+8)`
- **涉及模块**：数据预处理 / 对照实验设计 / 指标构建

#### 1. 启示性观点 (Insights)
- **中等难度边界题（正确率 25% ~ 75%）是认知跃升的黄金区间（ExGRPO, ICLR 2026）**：
  - 完全掌握的题目（100% 正确）回放会导致信息冗余与计算浪费，完全无法理解的题目（0% 正确）回放会导致策略发散；
  - 筛选 500 道处于能力临界区（~50% 准确率）的题目，能够提供最高质量的优化梯度。
- **低动作熵轨迹可抑制错误雪崩效应（Snowball Effect）**：
  - 仅凭最终答案正确容易混入“侥幸蒙对但逻辑混乱”的高熵轨迹，回放此类经验会污染策略；
  - 经验入库与回放必须基于**低动作熵（高确定性逻辑链）**。
- **经验复用比例 50% 最佳**：兼顾实时探索（Exploration）与经验利用（Exploitation）。

#### 2. 决策原因与依据 (Why)
- **严密验证核心科学假说**：先导论文提出了“500 道边界题激发质变”和“越解越快的加速度奖励”。如果用全量数据盲跑，不仅算力不可控，也无法剥离关键因果。参考 ExGRPO 的边界分桶筛选法，能在消费级算力（¥5,000 预算内）精确验证假说。
- **确立差异化创新与基线对照**：ExGRPO 本质是隐式参数回放，本项目则走向“显式 Engram 经验库 + 加速度奖励”，必须先确立严谨的基准线（Base vs GRPO vs ExGRPO vs Ours）以支撑论文成果。

#### 3. 代码构建与实验路线指导 (Implementation Guide)
- **数据准备脚本（后续编写 `borderline_filter.py`）**：
  - 对题库（MATH / GSM8K）每题采样 $K=16$ 条，统计在线准确率；
  - 剔除 0% 与 100% 的题目，锁定 $[0.25, 0.75]$ 难度区间；
  - 以 0.5 为均值高斯加权采样出 500 道黄金边界题。
- **实验对照组设置（4 组）**：
  - `B0`: `Qwen2.5-Math-7B` Zero-shot
  - `B1`: 标准 GRPO (R1-Zero 纯规则强化)
  - `B2`: ExGRPO (隐式经验池回放)
  - `Ours`: 本项目完整方案（两阶段 + 加速度奖励 + Engram 结构化经验库）
- **核心评估维度**：
  - 准确率：MATH-500, GSM8K, AIME 的 Pass@1
  - **思考加速度（核心突破点）**：正确题目的平均 Token 消耗步数及降幅（预期降低 20% ~ 40%）
  - 经验库演化：经验命中率与动态遗忘淘汰表现

---

### DEC-20260905-02: 代码工程渐进实现准则（前置基线复现 + 三基础解耦）

- **决策/修改时间**：`2026-09-05 11:15:00 (UTC+8) | 最新修订: 2026-09-05 11:18:00 (UTC+8)`
- **涉及模块**：系统架构设计 / 前置基线复现 / 模块解耦 / 防偏离基准

#### 1. 决策原因与立项依据 (Why)
- **先复现基线效果与现象，再叠加创新模块（核心变更）**：
  学术与工程研究的铁律是“Replicate Baseline First”。在动手实现我们自己的“加速度奖励 + Engram 经验库”之前，必须先在本地极轻量环境下（如 `Qwen2.5-Math-1.5B` 上）跑通参考论文（ExGRPO）的核心现象（难度分桶与低熵优势）。
- **避免多模块耦合导致调试盲区**：
  完成基线现象复现后，再将我们的自进化认知系统解耦为三个独立可单测的基础，逐级开发验证，严防出现错误时因多变量混杂而无法归因。

#### 2. 核心架构设计与执行路线图 (Architecture & Roadmap)
- **【阶段 0】小规模基线现象复现**：本地单卡运行 `Qwen2.5-Math-1.5B`，对 50 道 GSM8K 题采样验真分桶与低熵现象。
- **【基础一】边界题规模化挖掘器 (`src/evaluator/`)**：产出 500 道黄金边界题。
- **【基础二】加速度奖励与自适应基线引擎 (`src/rewards/`)**：维护 EMA 动态基线并计算复合奖励。
- **【基础三】Engram 显式经验库与动态演化容器 (`src/memory/`)**：实现结构化条目查表与动态淘汰。

---

### DEC-20260905-03: 阶段一（边界题挖掘与轨迹熵评估器）流水线落地规范

- **决策/修改时间**：`2026-09-05 11:30:00 (UTC+8)`
- **涉及模块**：`src/evaluator/` 数据流水线
- **详尽规范文档**：参见专门制定的独立子文档 👉 [decide_stage1.md](file:///d:/AI%20Program/Real%20Exp/decide_stage1.md)
- **决策要点**：
  1. **确定性四步流水线**：
     - 步骤 1：`generate_rollouts`（多重采样 $K=16$，温度 0.7）；
     - 步骤 2：`verify_and_partition`（数学等价判题，锁定 $25\% \sim 75\%$ 边界题）；
     - 步骤 3：`select_lowest_entropy`（Top-P 截断动作熵计算，选取该题唯一下标最优低熵轨迹）；
     - 步骤 4：`gaussian_sample_and_export`（以 $\mu=0.5$ 高斯加权导出 500 题集）。
  2. **红线契约**：严禁用 NLL 冒充动作熵、采样数 $K$ 严禁低于 8、判题必须支持 LaTeX 等价正则。

---

### DEC-20260905-04: 阶段二（加速度奖励与自适应基线引擎）流水线落地规范

- **决策/修改时间**：`2026-09-05 13:45:00 (UTC+8) | 方案 A 重构修订: 2026-09-05 14:30:00 (UTC+8)`
- **涉及模块**：`src/rewards/` 奖励引擎
- **详尽规范文档**：参见专门制定的独立子文档 👉 [decide_stage2.md](file:///d:/AI%20Program/Real%20Exp/decide_stage2.md)
- **决策要点**：
  1. **方案 A 核心转向（从步数压缩转为认知熵减提速）**：
     - 废止追踪“历史 Token 步数（800 Token）”，全面转向追踪各学科细分类别的“历史认知动作熵 / 困惑度标尺 $B_c$”（冷启动先验设为 1.0 Nats）；
     - 加速度计算重构为相对降熵比：$\Delta = \frac{B_c - H_{\text{current}}}{B_c}$，动作熵降低则正向激励（上限截断为 0.99），犹豫或发散（$H \ge B_c$）硬截断为 0.0；
     - 物理下界调整：动作熵理论最小值为 0.0 Nats，下界截断从 150 Token 调整为 $\text{min\_clamp} = 0.0$。
  2. **双条件经验安全门控（Experience Gating）**：
     - 引入外部审查智能体（Review Agent）语义审查判定 `is_exp_used`；
     - 门控升级为双条件联合判断：$\text{Gating} = \mathbb{I}(\text{is\_correct} \land \text{is\_exp\_used})$。做错一票否决归零；做对但未有效利用经验降熵者，加速度奖励强制为 0.0（$R_{\text{total}} = 1.0$）。
  3. **确定性四步流水线**：
     - 步骤 1：`get_or_init_baseline`（按学科类别独立获取或初始化历史认知动作熵基线 $B_c$）；
     - 步骤 2：`compute_raw_acceleration`（计算相对降熵比例 $\Delta = \frac{B_c - H_{\text{current}}}{B_c}$，约束在 $[0.0, 0.99]$ 之间）；
     - 步骤 3：`composite_reward_gating`（双条件安全门控合成，仅当 `is_correct == True` 且 `is_exp_used == True` 时放行加速度奖励）；
     - 步骤 4：`update_category_baseline_ema`（高惯性 EMA 演进，$\alpha \ge 0.95$，仅在做对时平稳更新动作熵标尺，物理钳位 $\text{min\_clamp} \ge 0.0$）。
  4. **红线契约**：严禁在答错或未有效复用经验时结算加速度奖励、严禁 $\alpha < 0.90$、严禁抹平题型差异使用全局统一标尺。

---

### DEC-20260905-05: 架构与验证策略重大升级（PPL困惑度加速度 + R1专家蒸馏内化 +【单测+真实微训练】双轨制）

- **决策/修改时间**：`2026-09-05 14:55:00 (UTC+8)`
- **涉及模块**：全局系统架构 / 强化学习奖励函数 / 渐进工程实现与验证铁律
- **详尽规范文档**：参见阶段三独立子文档 👉 [decide_stage3.md](file:///d:/AI%20Program/Real%20Exp/decide_stage3.md)
- **决策原因与依据 (Why)**：
  1. **困惑度（PPL）替代 Token 数量的科学合理性**：
     - Token 长度作为速度代理指标极易引发奖励投机（Reward Hacking，诱导模型跳步、盲目省略推导甚至猜答案）；
     - 困惑度 $\text{PPL}$ 衡量的是生成推导过程中的“认知阻力与思维流畅度”。专家经验生效时，模型对解题逻辑的转移概率高度集中确信，PPL 显著降低。使用 PPL 相对降低幅度作为加速度奖励，彻底从数学机理上杜绝了偷懒猜答案的投机行为。
  2. **全面回归 DeepSeek-R1 正统范式（避免底层算子魔改深坑）**：
     - 放弃对底层 Transformer/Engram 硬件级 C++/CUDA 查表算子的修改，保持底层架构成熟稳定；
     - 学习 R1 思路：在思维链 `<think>` 层面显式呈现经验调用，使用 RL（GRPO + PPL 加速度奖励）在边界题上训练出高熟练度的 **“经验调用专家模型（Expert Model）”**；
     - 让专家模型在海量题库上大批量解题，通过拒绝采样筛选出低 PPL、高正确率的黄金思维链数据集；
     - 通过 SFT 蒸馏微调基座模型，将“低困惑度、直击要害、越解越快”的经验调用本能**彻底内化进神经网络权重（真正的大脑内化，部署时无需外部存储）**。
  3. **渐进实现铁律升级为【代码单测 + 真实微训练】双轨制**：
     - 严禁纸上谈兵。每个阶段除了编写纯逻辑的 Python 单元测试，**必须在本地单卡上挂载真实模型（`Qwen2.5-Math-1.5B`）执行轻量微训练或真实微采样实验（Micro-Training / Micro-Eval Run）**；
     - 亲眼验证实际模型在数据分桶、PPL 真实下降曲线、强化微调收敛性上的真实有效性，才允许进入下一模块。

#### 核心算法公式规范
1. **序列困惑度（PPL）计算**：
   $$\text{PPL}(o \mid q) = \exp\left( -\frac{1}{|o|} \sum_{t=1}^{|o|} \log \pi_\theta(w_t \mid q, w_{<t}) \right)$$
2. **基于 PPL 的相对加速度奖励**：
   $$R_{\text{accel(PPL)}} = \max\left(0, \frac{\text{PPL}_{\text{baseline}} - \text{PPL}_{\text{current}}}{\text{PPL}_{\text{baseline}}}\right)$$
3. **带严格安全门控的复合奖励**：
   $$R_{\text{total}} = R_{\text{correctness}} + \mathbb{I}(\text{is\_correct}) \cdot \lambda \cdot R_{\text{accel(PPL)}}, \quad \lambda \in [0.1, 0.3]$$
4. **类别 PPL 基线的 EMA 平滑更新（仅在答对时更新）**：
   $$\text{PPL}_{\text{baseline}}^{(t)} = \alpha \text{PPL}_{\text{baseline}}^{(t-1)} + (1 - \alpha) \text{PPL}_{\text{current}}, \quad \alpha \ge 0.95$$

#### 三大阶段双轨实现与微训练路线 (Roadmap with Micro-Training)
- **【模块一】边界题挖掘与低 PPL 经验原则库**：
  - 轨一（代码单测）：采样分桶、PPL 计算与高斯加权单测；
  - 轨二（真实微实验）：本地单卡加载 `Qwen2.5-Math-1.5B`，真实跑 50 道 GSM8K 题目，验证 $25\% \sim 75\%$ 边界分桶分布，验证正确解答的 PPL 是否显著低于错误解答，沉淀出真实可用的前 10~20 条低 PPL 黄金解法。
- **【模块二】经验调用专家孵化（基于 PPL 加速度的 RL 微训练）**：
  - 轨一（代码单测）：PPL 加速度计算、安全门控复合奖励函数单测；
  - 轨二（真实微训练）：抽取 20 道边界题，在本地单卡启动 30~50 steps 的 GRPO 微训练 Run，实证记录 Loss 曲线与 PPL 下降轨迹，验证加速度奖励真实起效。
- **【模块三】专家合成数据大批量蒸馏（R1 模式终极内化）**：
  - 轨一（代码单测）：拒绝采样过滤流与 SFT 数据生成单测；
  - 轨二（真实微训练）：拿专家生成的 50~100 条低 PPL 黄金解题链微调原始基座（Micro-SFT 1 个 Epoch），在无外部经验库辅助下测试基座模型的 Zero-shot 推理表现，验证认知本能已彻底内化。

---

### DEC-20260905-06: 实验临时数据生命周期与存储三层管理规范

- **决策/修改时间**：`2026-09-05 15:05:00 (UTC+8)`
- **涉及模块**：存储架构 / 实验运行目录 / 检查点滚动保存 / Git 防污染
- **决策原因与依据 (Why)**：
  1. **防御磁盘几何级数膨胀**：强化学习与多重采样会产生极高维的 Logits、多版本全量模型权重与中间采样轨迹。若无存储规范，数十 GB 空间将在几天内打满；
  2. **实验可追溯与可复现性**：临时实验产物若杂乱平铺，会导致不同超参下的中间数据严重混淆，无法准确定位黄金数据与实验日志的对应关系；
  3. **明确冷、温、热数据生命周期**：区分“只读原始输入”、“质检通过的黄金交付物”与“可随用随删的临时实验流”。
- **启示与工程落地规范 (Insights & Protocol)**：
  1. **三层物理隔离目录**：
     - **冷数据 (`data/raw/`)**：原始公开题库，脚本仅有只读权限；
     - **温数据 (`data/artifacts/`)**：必须经专门检查函数（`check_stage*.py`）核验通过后方可写入并版本化归档（如 `stage1_borderline_500_v1.jsonl`）；
     - **热/临时数据 (`runs/{stage}_{timestamp}_{tag}/`)**：每次实验生成独立子目录，包含 `checkpoints/`、`staging/`、`logs/`、`audit_report.json`。
  2. **权重滚动保存（Rolling Checkpoints）**：微训练只存 LoRA 适配层（~30MB），全生命周期最多保留 `best_adapter` 与 `last_adapter` 2 个权重，超限自动旋转清理；
  3. **张量零泄漏（Zero Logits Leakage）**：Logits 仅在显存内计算熵/PPL 标量，算毕即刻 `del`，严禁未压缩的巨幅 Tensor 存盘；
  4. **根目录 `.gitignore` 硬性拉黑**：将 `runs/`、`*.pt`、`*.bin`、`*.safetensors`、`*.log` 物理隔离，确保 Git 仓库绝对纯净。

---

### DEC-20260905-05: 经验复用的因果判定机制与消融实验基准 (Experience Reuse Causal Validation)

- **决策/修改时间**：`2026-09-05 19:06:39 (UTC+8)`
- **涉及模块**：Experimental Design / Experience Gating / Internalization Validation
- **定位与版本演进说明**：本决策针对原 `DEC-20260905-05` 中以困惑度（PPL）单标量衡量经验加速的科学局限进行关键因果修正与机制升级（工程与理论演化序列编号：`DEC-20260905-07`），确立因果推断（Causal Inference）与行为消融（Ablation Intervention）为核心评价基准。

#### 1. 启示与依据 (Insights & Why)
- **困惑度（PPL）仅度量确定性，非因果性“经验复用”的充分依据 (Certainty vs. Causality)**：
  - 困惑度下降仅反映模型推导当前上下文时的转移概率更集中、推导阻力更小（即确定性/流畅度），不能构成“模型因果性复用了特定检索经验”的充分证据；
  - 缺乏干预的 PPL 改善极易混杂模型固有的预训练记忆或偶然的模式匹配偏差（Confounding Bias）。必须采用更具因果效力的行为干预实验，剥离虚假相关。
- **学术主张的双层解耦与边界约束 (Separation of Claims)**：
  必须将“外部显式经验利用”与“参数权重内化”这两层不同阶段的学术声明严格分层，严防在模型可解释性（Interpretability）上过度声明（Overclaiming）：
  1. **外部经验阶段（External Experience Usage）**：可以通过严格的干预实验，因果性地证明检索到的经验条目 $e$ 对当前任务产生确定性的正向边际贡献（Causal Contribution）；
  2. **权重内化阶段（Weight Internalization）**：当经验经由 SFT 写入神经网络参数后，在大模型可解释性理论限制下，严格来说无法证明网络内部保留了某条物理可定位的特定经验记忆。此时只能科学声明：**“蒸馏后的基座模型在完全剥离外部经验时，在未见任务上依然稳定表现出与经验一致的高效解题行为模式与策略留存”**。两阶段结论必须严格隔离，不可混为一谈。

#### 2. 证据层级重塑与智能体角色降级 (Evidence Hierarchy & Role Redefinition)
在经验复用评测体系中，确立以客观因果为主的证据金字塔：
- **主证据（Primary Evidence）**：**行为干预实验（Behavioral Intervention）**。以解题最终表现、步骤压缩的客观因果变化为最高判据；
- **轨迹解释器（Trajectory Interpreter）**：**审查智能体（Review AI / LLM-as-a-judge）**。从原本具有一票否决权的主判官，**降级为轨迹解释器**，仅用于对思维链逻辑进行质性解释与对齐分析，不再作为因果判定的唯一充分依据；
- **过程证据（Process Evidence）**：**检索日志（Retrieval Logs）**。记录经验召回、相关度匹配与上下文注入链路，作为不可篡改的过程追溯支撑。

#### 3. 最小判定标准与量化公式 (Minimal Criteria & Mathematical Formulation)
经验条目 $e$ 被严格定义为**“行为验证复用（Causally Reused via Behavioral Validation）”**，必须同时满足以下四个连乘判定条件：
$$\text{retrieved}(e) \land \text{Score}(\text{with } e) > \text{Score}(\text{without } e) \land \text{Score}(\text{full memory}) > \text{Score}(\text{memory without } e) \land \text{gain\_on\_unseen\_structural\_tasks} > 0$$

- **四步验证闭环契约**：
  1. **检索条件成立**：系统在当前问题下确定性检索并返回了经验条目 $e$；
  2. **反事实有效（Counterfactual Gain）**：加入该经验后模型表现显著优于无经验基准（$\text{Score}(\text{with } e) > \text{Score}(\text{without } e)$）；
  3. **删除有效（Ablation Sensitivity）**：从完整经验库中消融剔除该经验后，相关任务性能发生显著回退（$\text{Score}(\text{full}) > \text{Score}(\text{full} \setminus \{e\})$）；
  4. **泛化成立（Structural Generalization）**：该经验带来的策略收益能够稳定迁移并体现在未见过的同构题型上（$\text{gain} > 0$）。

- **复合评分函数（Score Function）与硬门槛红线**：
  $$\text{Score} = \text{accuracy} - \lambda \cdot \text{normalized\_token\_cost}$$
  - **准确率硬门槛（Hard Gate）**：**准确率（accuracy）是一票否决的硬性前提**。只有解题完全正确时，降低推理步数节省的 Token 成本才计入正向收益；严禁通过生成短而错误的答案获取虚高评分，彻底杜绝奖励作弊（Reward Hacking）。

#### 4. 四组对照消融实验基准规范 (4-Group Ablation Benchmark)
无论是否运行专门的审查 AI，系统评测与验证必须建立以下 4 组因果对照矩阵：
1. **有经验组（With Exp, $+\{e\}$）**：挂载标准匹配经验，验证推理加速度与答题增益；
2. **无经验组（Without Exp, Baseline）**：纯基座 Zero-shot 状态，测量零知识冷启动基准；
3. **删除经验组（Deleted Exp, $\mathcal{M} \setminus \{e\}$）**：特定经验剔除消融组，验证该条目在经验集中的因果独占性与必要性；
4. **错误经验组（Wrong Exp, $+e_{\text{corrupted}}$）**：对抗与误导测试组，注入故意篡改或逻辑反向的虚假经验，检验模型抗噪鲁棒性及对经验的实际依赖程度。
