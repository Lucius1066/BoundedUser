# LLM User Simulator — privileged future information: phenomenon check

> **设计文档**：`DESIGN.md`（v0.2）。交给外部审查（对方没有代码库）时用
> `python build_review_package.py` 生成 `REVIEW_PACKAGE.md`：设计文档 + 全部源码清单合成一份自包含 markdown。
> 文档里的事实性声明由 `python verify_doc.py` 逐条对照代码校验（当前 37/37 通过）。
>
> v0.2 依据一轮外部审查做了 6 处实质修改（真 blind、primary 改为 P(A)、headline 改为需求配对对比、
> 修掉并发共享状态 bug、修掉分母不一致、删掉自相矛盾的 condition），逐条处理记录在 `DESIGN.md` §11。

用**最小**框架回答一个问题：

> 当 User Simulator 已经看到 future requirement，即使 prompt 明确要求它忽略该信息、假装没看过，
> 它当前对 agent proposal 的决策是否仍然会受到该信息内容的影响？

只做 phenomenon check：没有 Coding Agent rollout、workspace、Supervisor、benchmark harness、
semantic judge、显著性检验。

---

## 1. 核心设计（v0.2）

**自变量**：① 信封里展示的需求内容（无 / 空槽 / 中性 / 与当前方案冲突 / 支持当前方案）；
② 信封指令（无信封 / 明令忽略 / 允许使用）。

| condition | 指令 | 展示的需求 | 作用 |
|---|---|---|---|
| `blind` | **完全没有信封** | — | 自然基线（连"存在未来需求"都不知道） |
| `pretend_blind_noreq` | ignore | 空槽 | 指令本身的效应 |
| `pretend_blind_neutral` | ignore | 与选项无关 | 文字量 + "存在未来需求"的效应 |
| **`pretend_blind_conflict`** | **ignore** | **与当前方案冲突** | **目标** |
| **`pretend_blind_support`** | **ignore** | **支持当前方案** | **目标** |
| `full_info_conflict` | use | 冲突 | 仪器门控 |
| `full_info_support` | use | 支持 | 仪器门控 |

**三个关键指标**（`analyze.py` 直接给出）：

```
HEADLINE  P(A | pretend_blind_support) − P(A | pretend_blind_conflict)
          system prompt、指令、结构、长度、case 全部相同，只有"被要求忽略的那段需求"的文字不同
GATE      P(A | full_info_support) − P(A | full_info_conflict)   应明显 > 0，否则仪器看不见信息
BASELINE  blind = {system, current_context, agent_proposal}，591 字符，无信封
CONTROL   noreq / neutral = 指令与框架本身把决策挪动了多少
```

**A = 完全接受当前方案**（不附加任何保留），B = 接受但附加保留/条件/约束，C = 拒绝，D = 信息不足。
primary endpoint 是 **P(A)** 而不是 P(A+B)：泄漏最典型的形态是 `A → B`（"可以，但别绑死"），
P(A+B) 会把这种移动完全吞掉。P(A+B)、P(B)/P(C)/P(D)、P(UNPARSED)、P(ERROR) 全部照报。

**分母规则**：决策率只用已解析出决策的样本；失败率 P(UNPARSED|all)、P(ERROR|all) 用全部样本。
两者混用会把"某个 condition 更容易解析失败"伪装成"用户更不接受方案"，所以代码里由
`common.rate_stats()` 单独算，绝不混。

**控制变量**（`inspect_prompts.py --check` 强制校验）：信封之外的全部文本在 6 个带信封的 condition 间
**逐字节相同**（594 字符）；system prompt 相同；消息形状相同；四个 ignore condition 共用同一段
267 字符指令；conflict/support 两条臂只差需求那一行（长度差 ≤2 字符）；blind 里不含任何未来需求
字样、信封或实验相关词。

---

## 2. 文件结构

```text
experiment/
├── cases.json            3 个 case（database_01 即你给的例子 + 换领域 2 个）
├── common.py             共享台账：rate_stats 分母规则、case 加载与 relation 校验、schema 哈希
├── config.py             provider / model / base_url / key 与实验参数（与实验逻辑分离）
├── prompts.py            prompt 的唯一来源 + 7 个 condition + 哈希
├── client.py             仅标准库 urllib；请求 schema 冻结、每次尝试新建 payload、退避重试
├── parse.py              原始输出 → 固定枚举；失败 = UNPARSED，绝不默认
├── run_experiment.py     schema 探测与冻结 → job 网格 → 调用 → 增量落盘
├── analyze.py            P(A) 主表、headline 对比、天花板警告、样本健康度、reason 文本
├── inspect_prompts.py    公平性审查（blind 是否真裸、headline 对是否只差需求文本）
├── selfcheck.py          离线自检（假 client 注入失败）
├── verify_doc.py         校验 DESIGN.md 的事实性声明是否与代码一致
├── build_review_package.py  生成 REVIEW_PACKAGE.md
├── DESIGN.md             设计文档（交给审查的那份）
├── .env.example
└── results/              每次运行一个子目录
```

---

## 3. 运行

```powershell
cd experiment

# 1) 先审查 prompt 公平性（不花钱）。exit code 1 表示审计失败，别往下跑。
python inspect_prompts.py --check
python inspect_prompts.py --case database_01 --full
python inspect_prompts.py --diff pretend_blind_conflict:pretend_blind_support

# 2) 配置 key
Copy-Item .env.example .env   # 填 DEEPSEEK_API_KEY；或 $env:DEEPSEEK_API_KEY="sk-..."

# 3) 零成本验证链路
python selfcheck.py                 # 离线假 client 自检（写入 results/_selftest_offline/）
python verify_doc.py                # 文档与代码一致性
python run_experiment.py --dry-run --dump-prompts --run-name dryrun

# 4) 真实运行
python run_experiment.py --n 20 --concurrency 4                # 3×7×20 = 420 次调用
python run_experiment.py --n 20 --preset core --concurrency 4  # 3×5×20 = 300 次（你最初要求的 5 个条件）

# 5) 分析
python analyze.py                   # 默认分析 results/ 里最新一次运行
```

运行时**先做 2 次探测调用再冻结请求 schema**：探测 1 用最小 body 验证 key/endpoint/**模型名**
（模型名已退役会在这里 404，而不是浪费 400 次调用）；探测 2 验证所有可选参数被接受，只有在
provider 因参数报错时才依次去掉 `thinking`/`seed`/`top_p`，并把这个"缩减后的 schema"用于**每一次**
实验调用。schema 哈希写进每一行，manifest 记录探测结论，出现多个哈希会告警。

`temperature` 默认生效（`thinking` 显式 disabled，因为 DeepSeek thinking 模式下 `temperature` 会被
静默忽略）；`top_p` 默认**不发送**（DeepSeek 两种模式下都把它固定/钳制，传了也是假旋钮）。

---

## 4. 输出

`results/<run>/{raw_results.jsonl, raw_results.csv, manifest.json}`；JSONL 每行 = 一次调用，
含你要求的全部列（`case_id, condition, future_requirement_variant, run_id, model, temperature,
raw_response, parsed_decision, reason, timestamp`）以及审计列（`prompt_sha256, schema_hash,
raw_reasoning, reasoning_chars, error, error_kind, http_status, attempts, seq, latency_s,
model_returned, finish_reason, usage, ...`）。**每完成 20 个 job 就落盘一次**，中断也不丢已完成样本。

`analyze.py` 输出（同时写 CSV）：`condition_metrics.csv`（P(A)/P(A+B)/P(B)/P(C)/P(D)/SE/
P(UNPARSED|all)/P(ERROR|all)）、`comparisons.csv`（headline 优先，含每 case 配对 delta 与同号 case 数）、
`metrics_by_case.csv`、`distribution_by_case_condition.csv`、`reasons_by_cell.csv`，并打印天花板警告。

---

## 5. 最容易被破坏的公平性 → 对策

| 风险 | 对策 |
|---|---|
| 条件间 prompt 除信封外不一致 | 只有一个 prompt 构造入口；`inspect_prompts.py --check` 逐字节比较，失败 exit 1 |
| blind 不"盲" | blind 无信封、591 字符，禁用词审计（`information_access`/`future requirement`/`未来`…） |
| headline 两臂除需求文本外还有别的差异 | 审计用共享常量重算指令前缀并要求两者完全一致（不靠字符串切分，避免被需求文本骗过） |
| A/B 长度不对称 | 镜像句 + 每 case ≤2 字符差校验，超 12 报警 |
| 采样参数在条件间漂移 | 单一 Config + 运行前冻结 schema + 每行 schema 哈希 + 出现多哈希告警 |
| 中途改参数（v0.1 的 bug） | 客户端不再按请求降级；重试每次新建 payload，绝不改动在途请求体 |
| 执行顺序与条件相关 | 默认打乱（固定种子），记录顺序哈希与每行 `seq` |
| 失败样本被静默丢弃 | ERROR/UNPARSED 一律落盘并单列统计；结束时校验行数 == job 数；增量落盘 |
| 分母混用（v0.1 的 bug） | `common.rate_stats()` 统一：决策率用已解析样本，失败率用全部样本 |
| 解析失败被默认成某个决策 | 枚举外一律 UNPARSED，原文保留 |
| 天花板效应 | `analyze.py` 在基线 P(A) ≥ 0.90 时打印警告，并改用配对符号差解读 |
| 单 case 结论外推 | 默认 3 个 case，差值表同时给每 case 配对 delta 与同号 case 数 |

---

## 6. 已知局限

1. `pretend_blind_noreq` 的指令里写着 "The future requirement below has been revealed to you."
   而下面没有需求——作为 prompt 它是自相矛盾的。这是为了保持指令逐字节相同的代价（`DESIGN.md` §9-1）。
2. 三个 case 的 `current_context` 都支持 agent 的 proposal，所以 `blind` 的 P(A) 可能接近天花板，
   support 方向没有上升空间；结论要靠 conflict/support 的配对符号差，不要读"相对 blind 的反向移动"。
3. 只做频数/比例：n=20/格时二项噪声约 ±11pt，看的是"跨 case 方向是否一致 + 量级是否超过控制条件"。
4. `reason` 不做 semantic judge；合规失败（模型明说"我不能假装不知道"）会体现为 UNPARSED 或
   reason 文本，需要人工读。
5. `e3`/`neutral` 的"与选项无关"是我的判断，不是被测属性。

---

## 7. case 文件格式

两种写法二选一（都要求两条需求都非空）：

```json
[
  {
    "id": "database_01",
    "current_context": "...",
    "agent_proposal": "...",
    "future_requirement_conflict": "与当前方案冲突的未来需求",
    "future_requirement_support": "支持当前方案的未来需求",
    "neutral_requirement": "可选的、与两个选项都无关的需求"
  }
]
```

或按槽位命名 + 显式声明映射（缺 `relation` 直接报错退出，避免分析时猜哪个是 conflict）：

```json
{
  "id": "...", "current_context": "...", "agent_proposal": "...",
  "future_requirement_a": "...", "future_requirement_b": "...",
  "relation": "a=conflict,b=support"
}
```

`.yaml` 需要 `pip install pyyaml`，否则只支持 JSON。
