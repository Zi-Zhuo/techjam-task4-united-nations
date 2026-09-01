# 稳健性评测交接摘要

## 目标

在不继续拟合公开 200 条会话的前提下，完善当前 Agent 对未知私测 target 的泛化能力，重点修复“大型 intent-card 候选碰撞组”的召回缺陷。

## 已确认的官方事实

- 私测共 800 条，场景比例固定为 Buying / Browsing / Intent Override / Boundary = 40% / 40% / 15% / 5%。
- 私测沿用公开 evaluator 的确定性消息模板、`ask_attribute` 策略、停止规则和评分公式，不包含隐藏 paraphrase。
- target 来自 Amazon Reviews 2023 Clothing 5-core leave-last-out；intent card 只由同一份 50,000 商品目录生成。
- 因此主风险是未知 target 分布，而不是语言模板变化。

官方说明：

- <https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/final_evaluation_faq.md>
- <https://github.com/TechJam2026/techjam-conversational-search>

## 当前产物

- 评测构造器：`scripts/evaluate_robustness.py`
- 冻结评测结果：`robustness_results.json`
- 完整分析：`docs/robustness_evaluation.md`
- 单元测试：`tests/test_robustness_evaluation.py`
- 复现命令：`pixi run evaluate-robustness`

构造器严格排除公开 200 个 target ASIN，但这些商品仍保留在检索目录中。每个套件为 20 条，场景数固定为 8 / 8 / 3 / 1；本轮总计仅运行 40 条。

## 核心结果

| 套件 | N | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同分布伪私测 | 20 | 1.000 | 1.000 | 2.400 | **0.972** |
| 高碰撞压力组（12 matched + 8 collision） | 20 | 0.700 | 0.567 | 5.700 | **0.626** |

同分布套件匹配了 target 的大类、热度、评分、价格、元数据完整度、目录 cohort、约束类型和 card 碰撞规模。20 个新 target 全部 rank 1 命中，说明当前高分并非简单记忆公开 ASIN，且官方同模板策略具备迁移性。

8 个 `full-card candidate count > 50` 的显式压力样本结果：

- HitRate@10 = 0.25；TechnicalScore = 0.143；
- 6 个完全 miss；
- 仅 2 个命中，均在第 10 轮，rank 分别为 7 和 5；
- 同一进程中的另外 12 个普通 matched target 全部命中。

## 根因定位

公开集的完整 card 候选数最大为 47，完全没有覆盖 `>50`；完整目录中约有 700 个商品处于该区域。

相关代码：

- `starter/agent.py::_card_evidence_level`：只有候选数 `<=50` 才可能判为 strong；
- `starter/agent.py::_recommendation_count`：medium 状态前 9 轮通常只返回 2 个，第 10 轮返回 10 个；
- 总覆盖约 28 个不同商品，无法覆盖 51–604 个同卡候选；
- medium 候选没有完整复用 strong 路径中的结构化 slot/LCS 排序，更多依赖 soft boost 和 popularity tie-break。

## 建议解答 agent 优先处理

1. 让 medium / 大碰撞集合也采用 deadline-aware 扩容；候选明显超过剩余覆盖能力时，尽早返回 `top_k=10`。
2. 对 medium 的 exact-card rows 使用与 strong 相同的结构化排序，再结合 category-normalized popularity、评分及稳定 tie-break。
3. 明确区分“最终分数优化”和“长尾鲁棒性”：真实 target 极度偏热门，不能为了 8 个刻意低热度压力样本破坏主分布 MRR。
4. 添加候选覆盖测试：`51`、`100`、`264`、`604` 四档，验证累计唯一推荐数、目标首次出现轮次和前排精度。
5. 不要修改 evaluator、公开标签或评测 target manifest。

## 验收方式

- 先运行 `pixi run test`，当前基线为 49 tests passed。
- 使用新的 seed 生成评测集，不在本轮冻结的 8 个 collision target 上调参。
- 新 matched 套件不得比当前主结果下降超过 0.02；若样本较小，至少保持全部命中且 MRR 不低于 0.95。
- 新 collision slice 应显著高于当前 HitRate 0.25，并报告 `>50` 子组结果，不能混入私测主估计。
- 修改后保留新的结果 JSON、seed、脚本 SHA-256 和运行耗时。

## 重要口径

- `robustness_results.json` 是本轮冻结审计集，不应继续用于选择超参数。
- 0.972 仅是 20 条条件匹配样本的结果，不是私测置信区间或最终分数保证。
- 现有两个 40 条“holdout”此前均参与过公开集策略实验，只能作为实现回归测试。
- 模板 paraphrase 可作为真实性 OOD 附录，但官方已明确不会用于 final，不应占用当前主要实验预算。
