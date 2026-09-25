# supervisor.py
"""
Supervisor-Worker 多 Agent 协作架构（基于 LangGraph StateGraph）

协作模式：
  - Supervisor（调度中枢）：接收用户消息，用 LLM 做路由决策，
    将任务分派给最匹配的专职 Worker，或判断已经可以结束。
  - Worker（专职执行）：每个 Worker 是一个独立的 Agent（create_agent），
    只负责一类任务（政策/知识库问答 / 天气 / 计算）。
  - 协作回路：Worker 完成处理后通过 Command 回到 Supervisor，
    Supervisor 根据最新上下文决定是否继续分派或结束。

对话记忆由最外层 StateGraph 统一通过 AsyncSqliteSaver 持久化，
各 Worker 不自带 checkpointer，避免多 Agent 共享 thread 冲突。
"""
import operator
import aiosqlite
from typing import Annotated, Literal
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from config import config
import context
from harness import guarded_tool
from tools.list_sources_tool import list_data_sources
from tools.rag_tool import query_my_documents
from tools.expense_tool import classify_expense, list_rd_expenses
from tools.risk_scan_tool import scan_rd_risk, list_risk_indicators
from tools.fill_timesheet_tool import fill_timesheet
from ledger import generate_rd_ledger


# 专职 Worker 节点名
WORKER_NAMES = ["policy_agent", "expense_agent", "risk_agent", "fill_agent"]


class RDState(MessagesState):
    """在 messages 之外，加一块「业务黑板」。

    ━━━ 为什么要加 ━━━
    原实现的 State 就是 MessagesState（只有一个 messages 列表），而且
    _make_worker_node 只把「最后一条用户消息」传给 Worker ——
    于是 4 个 Worker 各自回答、互不可见，连上下文都看不到。

    但业务上它们本应是一条链：
        工时填报 → 人员人工按工时占比分摊 → 费用归集 → 风险指标扫描
    这条链在旧实现里是断的（工时填完什么都没存，风险扫描读的是另一套数据）。

    ━━━ 字段怎么读 ━━━
    · 带 Annotated[list, operator.add] 的：多个节点写同一个字段是【追加】不是覆盖
      （messages 本身就是这个机制，这里只是把同一套用到业务数据上）
    · 不带 reducer 的（dict）：后写覆盖前写，适合「最新状态」类数据
    · ★ 读取一律用 state.get(key)，因为没写过的 channel 是空的（读取会 KeyError）
    """

    # ① 项目立项
    projects: Annotated[list, operator.add]
    # ② 工时记录（fill_agent 写）
    timesheets: Annotated[list, operator.add]
    # ③ 人员人工分摊（expense_agent 读工时后算，写回）
    allocations: dict
    # ④ 归集后的费用条目（expense_agent 写）
    expenses: Annotated[list, operator.add]
    # ⑤ 风险指标（risk_agent 读 expenses 后算，写回）
    risk: dict
    # ⑥ 辅助账 / ⑦ 留存备查清单
    ledger: dict
    evidence: Annotated[list, operator.add]
    # ★ 待人工复核（合规复核闭环的入口）
    pending: Annotated[list, operator.add]


def _format_board(state) -> str:
    """把业务黑板渲染成一段给 Worker 看的中文摘要。

    旧实现 Worker 只拿到最后一句用户消息，等于每次都从零开始；
    注入黑板之后，它能看到「本次会话已经产生了哪些业务数据」。
    """
    timesheets = state.get("timesheets") or []
    allocations = state.get("allocations") or {}
    expenses = state.get("expenses") or []
    risk = state.get("risk") or {}
    pending = state.get("pending") or []
    projects = state.get("projects") or []

    if not any((projects, timesheets, allocations, expenses, risk, pending)):
        return ""

    lines = ["【当前业务数据】（本会话已产生的状态，回答时请据此判断，不要凭空猜测）"]

    if projects:
        names = "、".join(str(p.get("name", p)) for p in projects[:5])
        lines.append(f"· 研发项目：{names}")

    if timesheets:
        lines.append(f"· 工时记录 {len(timesheets)} 条：")
        for t in timesheets[-5:]:
            lines.append(f"    - {t.get('employee', '?')} {t.get('date', '?')} "
                         f"「{t.get('project', '?')}」{t.get('hours', '?')}h")
        if len(timesheets) > 5:
            lines.append(f"    …（共 {len(timesheets)} 条）")

    if allocations:
        lines.append("· 人员人工分摊：" + "；".join(f"{k} ¥{v:,.0f}"
                     for k, v in allocations.items() if isinstance(v, (int, float))))

    if expenses:
        total = sum(float(e.get("amount", 0) or 0) for e in expenses)
        lines.append(f"· 已归集费用 {len(expenses)} 笔，合计 ¥{total:,.0f}：")
        by_cat: dict = {}
        for e in expenses:
            c = e.get("category", "未知")
            by_cat[c] = by_cat.get(c, 0) + float(e.get("amount", 0) or 0)
        for c, v in sorted(by_cat.items(), key=lambda x: -x[1])[:8]:
            lines.append(f"    - {c}：¥{v:,.0f}")

    if risk:
        if isinstance(risk, dict):
            lines.append("· 风险扫描：" + "；".join(f"{k}={v}" for k, v in list(risk.items())[:8]))
        else:
            lines.append(f"· 风险扫描：{risk}")

    if pending:
        lines.append(f"· ⚠️ 待人工复核 {len(pending)} 条：")
        for p in pending[:3]:
            lines.append(f"    - {p.get('reason', p) if isinstance(p, dict) else p}")

    return "\n".join(lines)

# 所有 Worker 共用的输出格式约束。
# 起因：模型喜欢输出 markdown 表格，但聊天气泡较窄，表格在窄容器里会挤成一坨、
# 甚至出现"表格没有换行"的畸形输出（实测遇到过 |---|---||值| 连在一起的情况）。
# 统一要求用分点列表，读起来更清楚也更稳。
FORMAT_RULES = (
    "\n【输出格式】用简洁的中文分点列表作答（每点一行，以「- 」开头），"
    "不要使用 markdown 表格。涉及比例/条件对照时，用「- 情形：对应比例」的形式逐条列出。"
)

# 路由令牌（Supervisor 只输出其中之一，避免依赖结构化输出 / response_format，
# 因为 deepseek-chat 等模型不支持 response_format 参数）
ROUTE_TOKENS = ["policy_agent", "expense_agent", "risk_agent", "fill_agent", "__end__"]


ROUTER_SYSTEM = """你是一个多 Agent 协作系统的 Supervisor（调度中枢）。
你的唯一职责：根据用户最新的消息，判断应该由哪个专职 Worker 处理，或判断已经可以结束。

可用 Worker：
- policy_agent：研发费用政策问答（加计扣除 100%、六大费用口径、高企认定条件、申报流程、辅助账、留存备查资料等），
  也负责「列出有哪些知识库/数据源」这类问题（它会调用 query_my_documents / list_data_sources 工具）。
  重要：若用户的问题既不属于 expense_agent / risk_agent / fill_agent，也不是结束语/寒暄，
  一律路由给 policy_agent——它会查询知识库，知识库没有的内容会明确告知"未收录"，不会编造。
- expense_agent：研发费用数据归集（把费用条目归类到 8 类费用口径：人员人工/直接投入/折旧/无形资产摊销/新产品设计费/装配调试/其他相关费用/委托研发），
  也负责「列出当前研发费用条目」。
- risk_agent：研发费用风险扫描（对标金四指标，输出绿/黄/红预警），也负责「列出风险指标」。
- fill_agent：研发工时填报（把自然语言描述转成工时单）。

决策规则：
1. 只依据「最新一条 user 消息」选择 Worker，不要管更早的历史。
2. 只有以下情况才输出 __end__：
   - 最新消息是结束语、寒暄或无实质任务；
   - 最新消息已经被当前 Worker 充分回答过（你看到的是 Worker 刚返回的标记消息）。
3. 一次只路由给「一个」Worker。
4. 如果某个问题可以由多个 Worker 处理，选择最匹配的那个。

【输出格式要求】你只能回复以下五个令牌中的【唯一一个】，不要输出任何解释、标点或多余文字：
policy_agent
expense_agent
risk_agent
fill_agent
__end__"""


# 当 LLM 不听话、没有输出可识别令牌时的兜底路由表
_ROUTE_FALLBACKS = [
    ("fill_agent", ["填报", "工时", "填写", "记录工时", "报工时"]),
    ("risk_agent", ["风险", "扫描", "预警", "指标", "对标", "金四"]),
    ("expense_agent", ["归集", "费用归类", "归类", "8类费用", "费用条目", "直接投入", "折旧", "摊销", "委托研发"]),
    ("policy_agent", ["加计扣除", "高企", "高新技术企业", "口径", "政策", "制度", "辅助账", "备查", "申报", "研发活动", "研发人员", "工资", "薪酬", "知识库", "数据源"]),
]


def _fallback_route(last_user_text: str) -> str:
    """当 LLM 没有按约定输出令牌时，用关键词兜底路由。"""
    if not last_user_text:
        return "__end__"
    text = last_user_text.lower()
    for target, keywords in _ROUTE_FALLBACKS:
        for kw in keywords:
            if kw.lower() in text:
                return target
    return "__end__"


def _make_llm():
    """构造 LLM。
    优先使用带 reasoning_content 兼容修复的 ChatDeepSeek（解决 DeepSeek thinking
    模式下 function calling 回传工具结果时的 400 报错）；
    若未安装 langchain-deepseek，则回退到通用 ChatOpenAI。"""
    try:
        from tools.deepseek_fix import ChatDeepSeekFixReasoningContent
        return ChatDeepSeekFixReasoningContent(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=config.TEMPERATURE,
        )
    except Exception:
        return ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=config.TEMPERATURE,
        )


def _tools_for(worker: str, fallback: list) -> list:
    """优先从技能库装配该 Worker 的工具；技能库异常时回退到硬编码列表。

    回退不是多余：技能库的价值是"可插拔"，但如果一个技能包写错就让整个服务起不来，
    这个抽象就是负收益。所以降级路径必须有，而且要告警（不静默）。
    """
    try:
        from skills.registry import tools_for_worker
        pairs = tools_for_worker(worker)          # [(Skill, guarded_tool), ...]
        if pairs:
            logger.info(f"[skills] {worker} 装配 {len(pairs)} 个技能: {[s.name for s, _ in pairs]}")
            return [t for _, t in pairs]
        logger.warning(f"[skills] {worker} 未匹配到任何技能，回退硬编码工具列表")
    except Exception as e:
        logger.warning(f"[skills] 技能库装配失败（回退硬编码）: {e}")
    return [guarded_tool(t) for t in fallback]


def _build_worker(tools, system_prompt: str):
    """构建一个专职 Worker（create_agent）。
    不自带 checkpointer：对话记忆由外层 StateGraph 统一通过 AsyncSqliteSaver 管理。"""
    return create_agent(model=_make_llm(), tools=tools, system_prompt=system_prompt)


async def _supervisor(state: RDState) -> Command[
    Literal["policy_agent", "expense_agent", "risk_agent", "fill_agent", "__end__"]
]:
    # 判断是否为「Worker 回答之后的二次路由」：
    # - 若最后一条消息是带 _worker_reply 标记的 AIMessage，说明本回合已由 Worker
    #   处理完毕，直接结束（省一次调用，也规避 DeepSeek 推理模型要求回传
    #   reasoning_content 而 langchain 已丢弃导致的 400 报错）。
    # - 用户发来新消息后，最后一条会是 HumanMessage（而非 AI），此时正常路由，
    #   避免把上一轮的 AI 历史误判为"本轮已处理"（这是多轮对话串台的关键修复）。
    last = state["messages"][-1] if state["messages"] else None
    if last is not None and getattr(last, "type", "") == "ai":
        ak = getattr(last, "additional_kwargs", {}) or {}
        if ak.get("_worker_reply"):
            return Command(goto=END)

    # 首次路由：使用纯文本输出 + 解析路由令牌（deepseek-chat 不支持 response_format/结构化输出）
    router = _make_llm()
    resp = await router.ainvoke([SystemMessage(content=ROUTER_SYSTEM)] + state["messages"])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    target = None
    for token in ROUTE_TOKENS:
        if token in text:
            target = token
            break
    # 兜底：若 LLM 没有输出可识别的路由令牌（thinking 模型常见），按关键词强制路由
    if target is None:
        last_user_text = ""
        for m in reversed(state["messages"]):
            if getattr(m, "type", "") == "human":
                last_user_text = str(getattr(m, "content", "") or "")
                break
        target = _fallback_route(last_user_text)
        logger.warning(f"Supervisor 路由未识别令牌(text={text[:80]!r})，关键词兜底: {target}")
    if target == "__end__":
        return Command(goto=END)
    return Command(goto=target)


def _make_worker_node(agent, name: str):
    async def node(state: RDState) -> Command[Literal["supervisor"]]:
        # 关键修复：只把「当前轮次的最后一条用户消息」传给 Agent，
        # 不传递任何历史 AI/Tool/早期 Human 消息。这样每个 Worker 只回答
        # 当前问题，避免被上一轮的 HITL 提示、错误回复或未回答问题干扰。
        last_human = None
        for m in reversed(state["messages"]):
            if getattr(m, "type", "") == "human":
                last_human = m
                break
        # 长期记忆注入：把「关于该用户的事实」作为前缀拼进本轮用户消息。
        # 注意只注入到 Worker，不注入 Supervisor —— 路由只依据用户原话，
        # 记忆里的词（如"偏好"/"项目"）会干扰令牌匹配，把路由带偏。
        memory = context.get_memory_prompt()
        if last_human is not None and memory:
            from langchain_core.messages import HumanMessage
            last_human = HumanMessage(
                content=f"{memory}\n\n{getattr(last_human, 'content', '')}"
            )
            logger.info(f"[memory] 已向 {name} 注入长期记忆（{len(memory)} 字符）")
        # ★ 业务黑板注入：把「本会话已产生的业务数据」拼进 Worker 的输入。
        # 旧实现只传最后一句用户消息 —— 工时/归集/风险之间没有任何数据流。
        board_text = _format_board(state)
        if last_human is not None and board_text:
            from langchain_core.messages import HumanMessage
            last_human = HumanMessage(
                content=f"{getattr(last_human, 'content', '')}\n\n{board_text}"
            )
            logger.info(f"[board] 已向 {name} 注入业务数据摘要（{len(board_text)} 字符）")

        filtered_state = {"messages": [last_human]} if last_human else state

        # ★ 开一块黑板草稿：本 Worker 的工具往里写，节点结束后整块收口到 State
        token = context.new_board()
        try:
            result = await agent.ainvoke(filtered_state)
        except Exception:
            context.reset_board(token)
            raise
        # 取 Worker 的最终回答（最后一条消息）
        reply = result["messages"][-1]

        # 关键兜底：thinking 模型（如 deepseek-v4-flash）在工具调用后可能生成空
        # content，而工具返回的 Observation（如 HITL 审核提示）才是真正的回答。
        # 此时从 result["messages"] 中倒序找到最近一条非空 ToolMessage，把内容补回。
        if getattr(reply, "type", "") == "ai" and not getattr(reply, "content", "").strip():
            for m in reversed(result["messages"]):
                if getattr(m, "type", "") == "tool" and getattr(m, "content", "").strip():
                    reply.content = m.content
                    break

        # 打上 _worker_reply 标记，供 supervisor 判断"本轮已处理"、
        # 避免与上一轮历史 AI 消息混淆（多轮对话串台的关键修复）。
        if hasattr(reply, "additional_kwargs"):
            reply.additional_kwargs["_worker_reply"] = True

        # ★ 黑板收口：把工具执行期间写进草稿的数据，落到 State 的业务字段。
        # 必须在 reset 之前取出来。
        draft = context.board_draft()
        context.reset_board(token)
        update: dict = {"messages": [reply]}
        if draft:
            update.update(draft)
            logger.info(f"[board] {name} 写入黑板字段: {list(draft)}")
        return Command(goto="supervisor", update=update)
    node.__name__ = name
    return node


class SupervisorWrapper:
    def __init__(self):
        self.conn = None
        self.supervisor = None
        self._agents = {}

    async def initialize(self):
        print("🧠 构建 Supervisor-Worker 多 Agent 协作系统...")
        print(f"  · 专职 Worker（{len(WORKER_NAMES)} 个）: {WORKER_NAMES}")

        self._agents = {
            "policy_agent": _build_worker(
                _tools_for("policy_agent", [query_my_documents, list_data_sources]),
                "你是研发费用政策问答 Worker。你的任务：仅针对对话历史中最后一条 user 消息回答，不要理睬历史里尚未回答的问题。"
                "【强制规则】只要用户问题涉及研发费用加计扣除、六大费用口径、高企认定、申报流程、辅助账、留存备查资料等政策内容，"
                "你必须调用 query_my_documents 工具检索知识库，绝不能跳过。"
                "当用户问「有哪些知识库/你能查什么」时，用 list_data_sources 工具。"
                "只基于检索到的上下文作答并引用来源，不编造；知识库没有的就说不知道。" + FORMAT_RULES,
            ),
            "expense_agent": _build_worker(
                _tools_for("expense_agent", [classify_expense, list_rd_expenses, generate_rd_ledger]),
                "你是研发费用数据归集 Worker。任务：仅针对最后一条 user 消息回答。"
                "当用户要求把费用归类/归集到 8 类口径时，用 classify_expense 工具（需先确定类别、金额、说明）。"
                "当用户问「列出研发费用条目」时，用 list_rd_expenses。只返回工具结果，不编造金额。"
                "当用户要求生成辅助账 / 汇总表 / 留存备查资料清单时，用 generate_rd_ledger；"
                "若本次会话还没有归集任何费用，工具会如实告知，不要凭空生成金额。" + FORMAT_RULES,
            ),
            "risk_agent": _build_worker(
                _tools_for("risk_agent", [scan_rd_risk, list_risk_indicators]),
                "你是研发费用风险扫描 Worker。任务：仅针对最后一条 user 消息回答。"
                "当用户要求扫描/检查研发费用风险时，用 scan_rd_risk 工具。"
                "当用户问「有哪些风险指标」时，用 list_risk_indicators。" + FORMAT_RULES,
            ),
            "fill_agent": _build_worker(
                _tools_for("fill_agent", [fill_timesheet]),
                "你是研发工时填报 Worker。任务：仅针对最后一条 user 消息回答。"
                "当用户描述「某人某天在某项目干了多少小时」时，用 fill_timesheet 工具（需提取：人员、项目、日期、工时、任务）。"
                "信息不全时先向用户确认缺失字段。" + FORMAT_RULES,
            ),
        }

        builder = StateGraph(RDState)
        builder.add_node("supervisor", _supervisor)
        for name, agent in self._agents.items():
            builder.add_node(name, _make_worker_node(agent, name))
        builder.add_edge(START, "supervisor")

        self.conn = await aiosqlite.connect("checkpoints.db")
        memory = AsyncSqliteSaver(self.conn)

        self.supervisor = builder.compile(checkpointer=memory)
        print("✅ Supervisor-Worker 多 Agent 系统构建完成")
        return self.supervisor

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None


async def create_supervisor():
    wrapper = SupervisorWrapper()
    await wrapper.initialize()
    return wrapper
