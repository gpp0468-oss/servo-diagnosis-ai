# 伺服排故系统 · 一个诚实的复盘

> 混合检索（BM25 + 中文嵌入向量）+ 243 条规则库 + 可选的 DeepSeek 生成：辅助排查站住了，判断对错没站住。

[![data](https://img.shields.io/badge/rules-243条-blue)]() [![codes](https://img.shields.io/badge/codes-122个-blue)]() [![eval](https://img.shields.io/badge/eval-33条-green)]() [![L0](https://img.shields.io/badge/L0命中-93%25-brightgreen)]()

---

## 这个仓库想说什么

现在的 AI 在工业诊断里能做**辅助决策和故障梳理**——把客户的口语翻译成术语、把一条报警码下的所有可能原因列全、让专家第一次介入就拿到全部有效信息。这一层它做得很好。

但它做不了 **Assurance**（保证结论正确、可追责、可验证）。根本原因是：

> **它在推理"关于问题的文字"，而不是在推理"问题本身"。**

这套系统最值钱的产出，是**两个负结果**：

1. 成本敏感信息增益排序，用数据证明它只值 **1%**（[docs/07](docs/07-排序实验.md)）
2. 检索分数**无法区分对错**——答错的中位分 0.9242，答对的中位分 0.9258，所以过度自信率是 **100%**，且任何阈值都拦不住（[docs/09](docs/09-置信度审计.md)）

第二个结果直接推翻了「加个分数阈值就能弃权」这个方案，也说明了为什么必须建**证据门**而不是分数门。

完整推演整理成了一篇长文：[《我给伺服排故做了一套 RAG 系统：辅助排查站住了，判断对错没站住》](https://zhuanlan.zhihu.com/p/2102843627479343706)；**本仓库只保留可复现的代码、数据与短文档**，不含长文正文。

### 两个组件，两套可复现性

| 组件 | 入口 | 有生成式模型？ | 可复现性 |
|---|---|---|---|
| **评测核心**（本仓库主体） | `eval_run.py` / `diag.py` / `rag_query.py` | ❌ 只有 bge-small-zh 嵌入模型（给检索打分） | **完全可复现**：本地跑、不联网、不要 API key |
| **Streamlit 界面**（[`ui/app.py`](ui/app.py)） | `streamlit run ui/app.py` | ✅ DeepSeek `deepseek-chat` / `deepseek-reasoner` + ToT 假设生成 | **需自备 API key**，不花钱跑不起来 |

**这个区别很重要**：所有对外公布的数字（93% / 67% / 83% / 70% / 1%）**全部来自评测核心，一个模型调用都没有**。UI 里那段 LLM 生成**没有任何指标支撑**——见下「UI 的已知局限」。

---

## 核心数据

### 评测基线

> **这组数字只在 Release 的 `rag-v1.0` 索引上成立**（见「B 档 复现」）。
> 本地用 `rag_build.py` 重建出来的索引是另一份产物，指标会变——实测无 PDF 重建后 Top-1 从 67% 掉到 56%。

33 条用例 = 20 条种子题 + 13 条现场题（其中 10 条带 `source=真实工单` 标注，3 条未标注）。

| 指标 | 结果 |
|---|---|
| L0 规则命中（带报警码） | 14/15 = **93%** |
| 检索 Top-1（纯症状描述） | 12/18 = **67%** |
| 检索 Top-3 | 15/18 = **83%** |
| 故障大类准确率 | 23/33 = **70%** |
| 平均延迟 | **~8 ms（本机实测）** |
| 评测集 vs 知识库重叠 | **0**（防泄漏） |

### 置信度审计：分数是不是"知道自己错了"的信号？

系统**没有弃权通道**——任何输入都输出 Top-1，答错也看不出来。所以：

> **过度自信率 = 100%（定义上必然）**

而检索分数也救不了：

| | 答对的 12 条 | 答错的 6 条 |
|---|---:|---:|
| Top-1 分数 · 中位 | 0.9258 | **0.9242** |
| Top-1 分数 · 区间 | 0.873–0.946 | 0.912–0.939 |

**两个分布几乎完全重叠**，所以从 0.30 到 0.80 的任何阈值，错案拦截率都是 **0/6**。

> 结论：检索分数反映的是「文本有多像」，不是「答案对不对」。
> **想修置信度，不能靠分数阈值，只能靠证据引用、可证伪条件、物理约束。**

详见 [docs/09-置信度审计.md](docs/09-置信度审计.md)。

### 排序实验（全文最核心的负结果）

| 收敛阈值 | 旧策略（按成本） | 新策略（信息增益） | 降幅 |
|---|---:|---:|---:|
| 0.70 | 2.79 步 | 2.76 步 | 1.2% |
| 0.75 | 3.04 步 | 3.00 步 | 1.3% |
| 0.80 | 3.20 步 | 3.19 步 | 0.3% |
| 0.85 | 3.32 步 | 3.32 步 | 0.0% |
| 0.90 | 3.52 步 | 3.52 步 | 0.0% |

**结论：信息增益排序在当前知识库上只值 0.0–1.3%。** 原因不是算法，是判据区分度不够（26 个码判据为空）。

详见 [docs/07-排序实验.md](docs/07-排序实验.md)。

---

## 知识库体检：五类脏数据

243 条规则逐条体检后挖出：

| 类型 | 数量 | 后果 |
|---|---|---|
| 串码（同一错误码混多套原因组） | — | 不该并列的原因被并列 |
| 未拆分（原因字段是占位符） | **25 个码** | 系统只能说"按手册 16.6 排查" |
| 判据为空 | **26 个码** | 无法判断"先查哪个" |
| 字段语义混淆（verify vs check_type） | 全库风险 | 推荐最贵的检查还理直气壮 |
| 非法枚举（`verify=未分类`） | **49 行** | 静默 fallback 到最便宜档 |

详见 [docs/02-数据体检.md](docs/02-数据体检.md)。

---

## 项目结构

```
.
├── data/
│   ├── rules.csv          # 243 条故障规则：原因→判据→处置→成本
│   ├── cases.csv          # 17 条现场案例
│   ├── eval.csv           # 33 条评测集（与知识库物理隔离）
│   └── feedback.csv       # 3 条«归档改善»种子数据（脱敏现场经验）
├── ui/
│   └── app.py             # Streamlit 界面（需 API key，见「Streamlit 界面」）
├── docs/
│   ├── 01-开发文档.md      # 系统架构与设计决策（v1 早期设计稿）
│   ├── 02-数据体检.md      # 五类脏数据与审计盲区
│   ├── 03-FTA故障树.md     # 故障树分析（`fta_build.py` 产物，仅供阅读，无代码消费）
│   ├── 04-案例覆盖度.md    # 知识库覆盖度对照
│   ├── 05-调试复盘.md      # 一次完整的排障复盘
│   ├── 06-评测方法.md      # 指标设计与评测自身的盲区
│   ├── 07-排序实验.md      # 信息增益排序：从 0% 到 1.3%
│   ├── 08-架构边界.md      # 为什么 AI 做不了 Assurance
│   ├── 09-置信度审计.md    # 检索分数不区分对错（过度自信率 100%）
│   └── notes/00-伺服原理笔记.md   # ⚠ 运行依赖：rag_build.py:147 会直接 open 它
├── *.py                   # 脚本平铺于根目录（角色见下「脚本角色」）
├── requirements.txt       # 评测核心依赖（无需 API key）
├── requirements-ui.txt    # 仅 Streamlit 界面需要（streamlit/openai/dotenv）
└── eval_report.md         # 最近一次评测报告（运行 eval_run.py 会覆盖）
```

`docs/01-开发文档.md` 是 v1 早期设计稿，其目录结构（`demo_starter/`、`启动.bat`）已不适用，实际结构以本 README 为准。

### 脚本角色

21 个根目录脚本 + 1 个 UI 脚本，**真正跑在评测链路里的只有 8 个**，其余是评测、迁移与文档工具：

| 类别 | 脚本 | 是否在运行时链路 |
|---|---|---|
| 运行时核心 | `csvio` `kb_category` `diag` `kb_dedupe` `rag_query` | ✅ 被产品代码 import |
| 索引构建 | `rag_build` `embed_build` `tokenizer_util` | ✅ 仅 C 档重建时 |
| 界面 | `ui/app.py` | ⭕ 可选；**调 DeepSeek，需 API key** |
| 评测与审计 | `eval_run` `eval_diagnose` `confidence_audit` `diag_rank` `diag_rank_show` `audit_rules` `check_leak` | ❌ 手动运行；**本仓库所有指标由它们产出** |
| 一次性迁移 | `fix_categories` `fix_type` `reclassify` `mark_leaked` `kb_apply` | ❌ 数据来路，保留供追溯 |
| 文档生成 | `fta_build` | ❌ 产出 `docs/03-FTA故障树.md`，无代码消费 |

`docs/` 下 10 篇文档中，**只有 `notes/00-伺服原理笔记.md` 被代码读取**（C 档重建），其余 9 篇均为人工阅读材料。

---

## 快速开始

**环境：Python 3.12 / 3.13 实测可用。** 先装依赖：

```bash
pip install -r requirements.txt
```

> **以下所有命令必须在仓库根目录执行（`cd servo-diagnosis-ai`）。**
> 部分脚本用 cwd 相对路径找 `data/`，在别的目录执行会「一半成功一半失败」。

### A 档 · 轻量（不需要模型，30 秒）

```bash
python audit_rules.py      # 数据体检：串码 / 未拆分
python kb_dedupe.py        # cases 查重与冲突（正常应报 0 冲突）
python check_leak.py       # 评测集防泄漏
python diag_rank.py        # 信息增益排序实验
```

预期输出：

| 命令 | 期望 |
|---|---|
| `audit_rules.py` | `审计完成。` |
| `kb_dedupe.py` | `重复 0 / 疑似重复 0 / 冲突 0` + 多原因结构说明 |
| `check_leak.py` | `eval: 33 \| cases: 17 \| overlap: 0` |
| `diag_rank.py` | `48 个故障码`，`降幅区间 0.0%~1.3%` |

### B 档 · 复现（推荐，可复现上面的全部指标）

`rag/` 是指向量的检索索引（35.7 MB，含 `rag/model/` 里的 ONNX 模型），不入 git，**放在 GitHub Release**：

1. 从 Release 下载 **`rag-v1.0.zip`**：

   ```
   https://github.com/gpp0468-oss/servo-diagnosis-ai/releases/tag/rag-v1.0
   ```

2. 校验（SHA256 必须一致；不一致说明下载损坏，不要继续）：

   ```powershell
   Get-FileHash .\rag-v1.0.zip -Algorithm SHA256
   # 期望 BE63317CC9DB7CB1C919F688113BF4FDE9499CA4F56E1DEB5F84F8F9B57D6494
   ```

3. 解压到**仓库根目录**，得到 `rag/`。注意是**两级目录**——顶层 4 个文件，`rag/model/` 下还有 2 个：

   ```
   rag/chunks.jsonl          rag/bm25.pkl
   rag/vectors.npy           rag/README.md
   rag/model/model.onnx      rag/model/tokenizer.json
   ```

4. 跑评测：

   ```bash
   python eval_run.py         # 期望 14/15、12/18、15/18、23/33、~8ms
   python confidence_audit.py # 期望 12 对 / 6 错，中位 0.9258 / 0.9242
   python eval_diagnose.py    # 失败用例归因，写 eval_diagnosis.md
   ```

只有走这条路，才叫"复现"。README 里的每一个指标都是在 `rag-v1.0` 上测出来的。

### C 档 · 重建索引（可选，**不等于复现**）

想把系统换到别的数据/手册上时用。需要两样东西：`pypdf`（已在 requirements 里）和一份厂商手册 PDF（**不随仓库分发**）。

```bash
# 有 PDF：含 manual 的完整重建
python rag_build.py --pdf "D:\path\to\SV660N系列伺服用户手册-CN-D01.PDF"
# 或用环境变量
SV660N_PDF="D:\path\to\手册.PDF" python rag_build.py

# 无 PDF：只用 笔记 + rules + cases 构建（本地调试用）
python rag_build.py

# 重算向量（必须！否则向量与切片错位）
python embed_build.py
```

**三条硬性边界：**

1. **重建产物与 Release 的 `rag-v1.0` 不是同一份索引，不可用于复现 README 指标。** 无 PDF 时脚本会打印 `[WARN]` 跳过 manual 部分；有 PDF 时也会打印「本次为重建索引」。实测无 PDF 重建后 Top-1 从 67% 掉到 56%。
2. **`rag_build.py` 只覆盖 `chunks.jsonl` 与 `bm25.pkl`，不覆盖 `vectors.npy`。** 如果你之前解压过 Release 的 `rag/`，重建完**必须立刻重跑 `embed_build.py`**；否则是「新切片 + 旧向量」，检索按错位下标取结果，数字全错但不报错。
3. 为了防止第 2 条被忽略，`Rag` 初始化时会硬校验行数：`chunks` 与 `vectors` 数量不一致就抛 `RagIndexMismatch`，`eval_run` / `confidence_audit` / `eval_diagnose` 都会以 exit 2 退出，**不会打印错误指标**。

---

## Streamlit 界面（可选）

`ui/app.py` 是一个四模式的网页原型。**它不是评测核心，也不产出任何指标**——只是把 `rag_query` / `diag` / `eval_run` 包成可点的界面。

```powershell
pip install -r requirements.txt -r requirements-ui.txt
streamlit run ui/app.py
```

左侧填 **DeepSeek API Key**（`https://platform.deepseek.com` 申请，或设环境变量 `DEEPSEEK_API_KEY`），模型可选 `deepseek-chat` / `deepseek-reasoner`。

| 模式 | 做什么 | 用不用模型 |
|---|---|---|
| 单轮问答 | 现象 → 规则/检索 → DeepSeek 出结论 | ✅ |
| 分步诊断 | 生成互斥假设表 → 逐条标记 → 自动剪枝 → 给下一步 | ✅ |
| 评测 | 跑 `eval_run`，界面里看 33 条结果并下载报告 | ❌ 纯本地 |
| 归档改善 | 现场经验入 `data/feedback.csv` → 审核 → 合并进 `cases.csv` 并重建索引 | ❌ 纯本地 |

### UI 的已知局限（重要）

- **那段 LLM 生成没有任何评测支撑。** `SYSTEM_PROMPT` 要求模型输出「3 个互斥假设 + 支持/反驳证据 + 剪枝理由 + JSON」，格式好看，但**100% 的公开指标（93% / 67% / 83% / 70% / 1%）都不包含它**——它们全部来自不调模型的评测核心。
- **不填 key 就跑不了。** 所以别人 clone 下来能免费复现的是评测核心，不是 UI 的效果。
- 归档改善会**改写 `data/cases.csv` 并重算 `rag/` 索引**，跑完指标会变（等价于 C 档重建，不再等于 `rag-v1.0`）。想恢复：重下 Release 的 `rag-v1.0.zip`，`git checkout data/cases.csv`。

---

## 设计决策

| 决策 | 选择 | 代价 |
|---|---|---|
| 分类维度 | 按**故障源**，不按现象 | 树必须互斥，同一病根的多表现要靠第二维 |
| 数据结构 | `category` 单值 + `phenomenon_tags` 多值 | 多值标签**未接进检索**，仅用于评测判分放宽（实测只多对 1 行） |
| 推理路径 | 评测核心：带码走规则直出，无码走检索+模板（**无 LLM**）；UI：一律可调 DeepSeek | 无码时效果完全依赖检索；UI 那条无评测 |
| 人机边界 | AI 只摆清事实，不做决定 | 无法自动化闭环 |

---

## 移植到其他设备

规则库基于某一个品牌的伺服（SV660N）建库。换品牌后故障码、功能码、参数名全变。

**唯一能跨品牌迁移的是「参数在控制环里干什么」这一层理解** —— 见 [docs/notes/00-伺服原理笔记.md](docs/notes/00-伺服原理笔记.md)。

移植步骤：

1. 读原理笔记，搞清楚目标设备的控制环与参数体系
2. 从目标设备手册提取故障码 → 按 `data/rules.csv` 的字段结构重建
3. 按新设备重写 `data/eval.csv`（至少 30 条，含真实工单）
4. 重跑 `audit_rules.py` —— **先体检数据，再谈模型**
5. 走 C 档重建索引；接受"这是新索引，指标不能与本文对比"

---

## 已知局限

这个项目**没有真实用户**，所有指标来自自建评测集与模拟，仅供验证技术路线。

- 评测覆盖"找没找到"，不覆盖"下一步查哪个"（后者无标准答案）
- **UI 里 DeepSeek 生成的诊断建议，一条题都没有测过**——33 条评测集测的是检索和分类，不是生成质量
- 排序层效果只能靠模拟，而模拟依赖手工设定的似然参数
- 判据质量是瓶颈：26 个码判据为空，其余多为"查看/测量……"
- `eval_report.md` 是 2026-11-05 的历史结果，任何一次 `eval_run.py` 都会覆盖它

---

## 许可

- 代码：MIT
- 数据与文档：CC BY 4.0（可转载改编，需署名）
- 文中故障码与手册内容均来自公开产品手册
- Release 内 `rag/model/` 是 bge-small-zh 的 ONNX 构建产物，版权归原项目（BAAI）；仅用于复现本仓库指标

---

*作者：gpp0468-oss · 4 年伺服/PLC 现场排故*
