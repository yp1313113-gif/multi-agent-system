# 研发费用智能管理 Agent 平台（multi-agent-system）

> 基于 **LangGraph Supervisor-Worker 多 Agent + 混合检索 RAG** 的研发费用智能管理助手：政策问答、费用归集、风险扫描、工时填报，四位一体。
> 回答可靠（grounding 防幻觉 + 引用来源）、过程可控（敏感查询人工审核 HITL）、能力可标准化接入（MCP）。
>
> 📖 [技术选型理由（面试速查版）](./docs/tech_rationale.md) ｜ 📋 [需求规格说明书](./docs/需求规格说明书_研发费用智能管理系统.md)
>
> 🗓️ 开发周期 2026.04 — 2026.08：4-7 月完成核心功能开发与迭代（早期在本地维护），8 月完成需求自洽整改（补需求规格/清理旧知识库/依赖迁移）与并发控制后统一开源上线。

---

## 架构定位：Agent Loop + Harness + MCP

- **Agent Loop**：Supervisor-Worker 循环（路由 → Worker 执行 → 回收 → 再路由），LangGraph StateGraph 驱动；防死循环用 recursion_limit + AgentLoopGuard 双保险。
- **Agent Harness**（harness.py）：工具执行环境与约束层——统一超时 / 有限重试 / 日志 / 耗时统计（guarded_tool 包装所有 Worker 工具），保证"执行可靠、有界、可观测"。
- **MCP 接入**（mcp_server.py）：把费用归集 / 风险扫描 / 工时填报 / 费用条目暴露为标准 MCP 工具（stdio 传输），外部系统或其他 Agent 可通过 MCP 协议调用——"Agent 能力标准化接入"。

---

## 功能特性

| 能力 | 实现说明 |
|------|----------|
| 🤖 多 Agent 编排 | **LangGraph StateGraph 实现的 Supervisor-Worker**：Supervisor 用 LLM 做路由决策，分派给 4 个专职 Worker，协作回路由 `Command` 控制 |
| 🛡️ 防幻觉（grounding） | 检索内容与问题相关性门控（`_is_relevant`），不相关一律拒答"知识库未收录"，绝不编造；回答附引用来源 |
| 🔄 CRAG 重搜 | 检索不足时改写查询重搜一次（`_rewrite_query`），提高召回 |
| 💾 多轮对话记忆 | `AsyncSqliteSaver` 持久化 checkpointer，服务重启后对话历史不丢失；支持 /reset |
| 🔍 混合检索 RAG | BM25（jieba 分词）+ Chroma 向量（bge-small-zh-v1.5）→ RRF 融合 → bge-reranker-v2-m3 重排，Recall@3 **88%** |
| ⚡ Redis 缓存 | 高频问题命中缓存毫秒级返回（`_cacheable` 准入策略：仅带引用来源的答案入缓存）；Redis 不可用时自动降级 |
| 🛡️ HITL 人工审核 | 敏感词命中（我的/个人/工资/薪资/薪酬）触发审核队列，需 `python hitl.py` 批准后才能查询，审核状态持久化 |
| ⏱️ Agent Harness | 工具执行统一包装：超时（10s）/ 重试（2 次）/ 日志 / 耗时（guarded_tool 接入全部 Worker 工具） |
| 🔀 并发控制 | asyncio.Semaphore 限流：并发满排队、超时返回 503 保护下游；single-flight 缓存防击穿：热点 key 并发 miss 只重建一次，其余等待共享结果 |
| 🔌 MCP 接入 | mcp_server.py 暴露 4 个工具为标准 MCP 工具（stdio），外部 Agent 可协议调用 |
| 📊 全链路追踪 | Langfuse（可选，未配置自动降级）+ request_id，可视化监控 Token 消耗与响应延迟 |
| 🐳 容器化部署 | docker-compose 一键启动（FastAPI 服务 + Chroma + Redis） |

---

## 核心亮点

- **4 个专职 Worker，各管一段真实业务**：
  - **policy_agent**：研发费用政策问答（加计扣除 100%、8 类费用口径、高企认定、辅助账、留存备查、申报流程）——grounding 门控，知识库没有就明说"未收录"，不编造；
  - **expense_agent**：费用归集——把自然语言描述归类到 8 类口径（人员人工/直接投入/折旧/无形资产摊销/新产品设计费/装配调试/其他相关费用/委托研发）；
  - **risk_agent**：风险扫描——对标金四 6 个比例指标（其他费用≤10%、研发人员≥5%、直接投入≤50%、委托 80%、高企 3/4/5%、波动）输出绿/黄/红预警；
  - **fill_agent**：工时填报——自然语言转工时单，信息不全先确认，不补造数据。
- **可靠性工程**：Agent Harness（超时/重试/日志） + 死循环双保险 + AST 白名单计算工具（防代码注入）；
- **生产级素养**：HITL 人工审核、缓存准入、Redis 降级、上下文 token 预算（检索文档截断 4000 字符）、全链路可观测、CI 自动测试。
- **并发控制（高并发素养）**：API 层 asyncio.Semaphore 限流（并发满排队超时返回 503，保护 LLM API 与数据库）+ single-flight 缓存防击穿（热点问题缓存过期瞬间只重建一次，其余等待共享结果）；/health 进程内压测 **50 并发 QPS≈452、零错误**。
- **真 token 级流式**：`astream_events(v2)` 按 `langgraph_node` 只转发 Worker 的最终回答 token；工具调用轮的前缀缓冲整段丢弃（避免 thinking 模型把"预答文本"推给用户造成重复输出）；流式异常时回读 checkpointer 兜底，保证**一定有回答**。实测热态首字延迟 4.9s、生成阶段 1.0s（数百个 token 块）。
- **两级缓存（优雅降级）**：Redis + 进程内 LRU。Redis 未部署/抖动时自动降级，缓存能力不归零 —— 不是"要么全有要么全无"；`/health` 直接暴露当前生效的缓存后端，降级状态一眼可见。
- **服务预热（消灭首个请求的冷启动）**：启动时预加载向量模型 / 交叉编码器重排模型 / BM25 索引 / 编排图。实测**首个请求 26.6s → 8.2s**（预热本身耗时 17.9s，只在部署时付一次，不再由第一个用户买单）。预热是尽力而为的：任何一步失败只告警、不阻断启动。
- **长期记忆 + 准入判断**：按 session 记住「偏好 / 身份 / 常用项目」，写库前做三重准入（白名单 key、拒绝问句与时效性表述、长度上限）防**记忆污染**；只注入 Worker、不注入 Supervisor（记忆里的词会干扰路由令牌匹配）。
- **技能库（Skill Registry）**：`skills/<name>/SKILL.md` 声明式能力单元（名称 / 版本 / 说明 / 何时使用 / 归属 Worker / 自带评测用例），启动扫描 → 动态加载 → 按归属自动装配到对应 Worker。新增能力 = 新增一个目录，**主流程零改动**；单个技能包写坏只跳过并告警，不影响服务启动。
- **Token 成本埋点**：按「会话 / 节点」记账，能看出钱花在 Supervisor 路由还是某个 Worker；支持单会话费用阈值告警（为降级策略留接口）。

---

## 系统架构

```mermaid
flowchart TD
    U["用户问题"] --> API["FastAPI /chat (SSE 流式)"]
    API --> SUP["Supervisor 调度中枢<br/>(LLM 路由决策)"]

    SUP -->|"policy_agent"| PA["政策问答 Worker<br/>(加计扣除/口径/高企)"]
    SUP -->|"expense_agent"| EA["费用归集 Worker<br/>(8 类口径)"]
    SUP -->|"risk_agent"| RA["风险扫描 Worker<br/>(金四 6 指标红黄绿)"]
    SUP -->|"fill_agent"| FA["工时填报 Worker<br/>(信息不全先确认)"]

    PA --> SUP
    EA --> SUP
    RA --> SUP
    FA --> SUP

    PA --> RAG["query_my_documents<br/>(混合检索 + grounding 门控 + CRAG)"]
    RAG --> HITL{"HITL 敏感词检测"}
    HITL -->|"敏感"| APP["人工审核队列 (SQLite)<br/>需 python hitl.py 批准"]
    HITL -->|"普通"| CACHE{"Redis 缓存命中?"}
    CACHE -->|"命中"| ANS["返回缓存答案"]
    CACHE -->|"未命中"| HY["混合检索 HybridRetriever"]

    HY --> BM25["BM25 稀疏检索 (jieba 分词)"]
    HY --> VEC["向量稠密检索 (Chroma)"]
    BM25 --> RRF["RRF 融合 (k=60)"]
    VEC --> RRF
    RRF --> RR["bge-reranker-v2-m3 重排"]
    RR --> GEN["LLM 生成 + 引用来源"]
    GEN --> ANS

    EA --> EXP["classify_expense / list_rd_expenses"]
    RA --> RSK["scan_rd_risk / list_risk_indicators"]
    FA --> FILL["fill_timesheet"]

    subgraph Harness["Agent Harness (guarded_tool)"]
        EXP
        RSK
        FILL
        RAG
    end

    ANS --> U
    SUP -.->|"Langfuse 追踪"| LF["Langfuse 可观测"]
```

---

## 技术栈

- **Agent 框架**：LangChain 1.3 + LangGraph 1.2（StateGraph Supervisor-Worker 多 Agent）
- **LLM**：DeepSeek API（OpenAI 兼容接口）
- **向量检索**：ChromaDB + HuggingFace `bge-small-zh-v1.5`
- **重排**：`bge-reranker-v2-m3` CrossEncoder
- **缓存**：Redis（带降级）
- **记忆**：AsyncSqliteSaver（checkpoints.db）
- **可观测**：Langfuse（可选降级）
- **服务**：FastAPI（SSE 流式输出）
- **部署**：Docker + docker-compose

---

## 项目结构

```
multi-agent-system/
├── run.py              # CLI 程序入口（日志 / Langfuse / 流式对话）
├── api.py              # FastAPI 接口（SSE 流式输出，/chat + 聊天前端）
├── supervisor.py       # Supervisor-Worker 多 Agent 编排（路由令牌 + 兜底路由 + Harness）
├── harness.py          # Agent Harness：guarded_tool / with_timeout / with_retry / AgentLoopGuard
├── concurrency.py      # 并发控制：AsyncLimiter 限流 + SyncSingleFlight 缓存防击穿
├── mcp_server.py       # MCP 服务器：4 个工具暴露为标准 MCP 工具（stdio）
├── config.py           # 统一配置管理（多源 DATA_SOURCES / .env 热更新）
├── context.py          # 请求级上下文（contextvars，并发隔离；request_id + 记忆提示块）
├── memory_store.py     # 长期记忆：三重准入判断 + SQLite 持久化 + LLM 事实抽取
├── cost_tracker.py     # Token 成本埋点：按会话/节点记账 + 预算告警
├── skills/             # 技能库：声明式能力单元（新增能力只加目录，主流程零改动）
│   ├── registry.py     #   扫描 SKILL.md → 解析元数据 → 动态加载 handler → 按 Worker 装配
│   └── <skill>/SKILL.md#   每个技能：名称/版本/说明/何时使用/归属 Worker/评测用例
├── exceptions.py       # 自定义异常分类
├── hitl.py             # 人工审核模块（Human-in-the-loop，SQLite 持久化）
├── cache.py            # 两级缓存：Redis + 进程内 LRU（Redis 不可用时自动降级）
├── ingest.py           # 知识库构建（多文件 + 结构切分 + 向量化 + 入库）
├── docker-compose.yml  # Docker 部署编排
├── Dockerfile          # 镜像构建
├── requirements.txt    # 依赖清单
├── data/               # 知识库文档（研发费用政策库 / 归集FAQ / 风险指标库）
├── docs/               # 项目文档与图表（tech_rationale.md 面试速查版）
├── eval/               # 评测脚本（检索召回率对比 + ragas + 流式性能基准 + 端到端验证）
├── tests/              # pytest 单测（工具 / 编排 / 并发 / 长期记忆 / 技能库 / 成本埋点）
├── .github/workflows/  # CI（push/PR 跑 pytest）
└── tools/              # 工具集
    ├── rag_tool.py       # 政策检索（混合检索 + grounding 门控 + CRAG + HITL + 缓存准入）
    ├── expense_tool.py   # 费用归集（8 类口径）+ 费用条目
    ├── risk_scan_tool.py # 风险扫描（金四 6 指标红黄绿）
    ├── fill_timesheet_tool.py # 工时填报（信息不全先确认）
    ├── list_sources_tool.py   # 数据源列表
    ├── math_tool.py      # 通用计算工具（AST 白名单安全防护，供扩展）
    ├── data_clean.py     # 知识库文本清洗
    └── doc_parser.py     # 章节解析（结构切分）
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env（参考 config.py，填入 DEEPSEEK_API_KEY 等）
#    创建 .env：DEEPSEEK_API_KEY=sk-xxx （模型/Redis/阈值等均可在 config.py 调整）

# 3. 构建知识库（data/ 下 txt → Chroma）
python ingest.py

# 4. 启动服务（启动时会预热模型，约 18s；可用 WARMUP_ENABLED=false 关闭）
python api.py          # http://127.0.0.1:8001 聊天前端
# 生产部署：uvicorn api:app --host 0.0.0.0 --port 8000
# 或 CLI 交互
python run.py

# 5. 启动 MCP 服务器（标准 MCP 工具，stdio）
python mcp_server.py

# 6. 跑测试
pytest
```

---

## 测试与评估

- **单元测试**：79 passed 1 skipped（工具数据流、缓存准入、Supervisor 路由回环、并发控制限流/防击穿、长期记忆准入判断、技能库装配与容错、成本记账与解析）
- **流式性能基准**：`python eval/stream_bench.py` —— 实测冷启动 TTFT 26.6s / 热态 4.9s，生成阶段 ~1.0s（数百个 token 块证明是真流式而非分块补发）
- **端到端验证**：`python eval/verify_all.py` —— 一轮说身份偏好 → 落长期记忆 → 下轮 Worker 收到记忆注入 → 全程 token 自动记账
- **服务实测**：`python api.py` 启动 → 预热 17.9s → 首个请求 8.2s、热态 5.0s（含 `✅ 缓存命中`）；`/health` 返回预热耗时与缓存后端
- **检索评估**：eval/retrieval_eval.py 对比纯向量 / BM25 / 混合检索召回率（Recall@3 88%，相对纯向量 72% 提升）
- **CI**：.github/workflows 在 push/PR 时自动跑 pytest

---

## 面试亮点速记

1. **为什么多 Agent？** 单一 Agent 工具越多路由越乱；按业务拆 4 个专职 Worker，职责单一、可独立演进；
2. **怎么防幻觉？** grounding 相关性门控 + CRAG 重搜 + 引用来源 + 无来源拒答，四层防线；
3. **怎么保证可靠？** Agent Harness：统一超时/重试/日志/耗时，防死循环双保险；
4. **怎么对外开放能力？** MCP 标准协议，外部系统可协议调用；
5. **生产级素养**：HITL 审核、缓存准入、降级、token 预算、可观测、CI。
6. **流式怎么做才是真的？** `astream_events` 按节点过滤 + 工具调用轮缓冲丢弃；不是 `ainvoke` 之后再分块假装流式；
7. **记忆怎么防止污染？** 写库前三重准入（白名单 key / 拒绝问句与时效性表述 / 长度上限）——宁可少写，不可写错；
8. **怎么让能力可插拔？** 技能库把「工具 + 提示词 + 评测」打包成带版本的目录，声明式装配，主流程零改动；
9. **成本怎么控？** 按节点记账找出钱花在哪（Supervisor 路由 vs Worker），再谈优化；本项目的确定性路由本身就是省成本设计；
10. **冷启动怎么优化？** 模型加载/索引构建这类一次性成本从"首个用户请求"挪到"服务启动"——首个请求 26.6s → 8.2s；
11. **请求上下文怎么隔离？** 用 `contextvars` 而不是模块级全局变量：10 并发下全局变量会让 A 请求的日志打上 B 的 request_id，线程池还要显式 `copy_context()` 才能把上下文带进工具线程。
