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
import aiosqlite
from typing import Literal
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from config import config
from tools.list_sources_tool import list_data_sources
from tools.rag_tool import query_my_documents
from tools.weather_tool import get_weather
from tools.math_tool import calculate


# 专职 Worker 节点名
WORKER_NAMES = ["rag_agent", "weather_agent", "math_agent"]

# 路由令牌（Supervisor 只输出其中之一，避免依赖结构化输出 / response_format，
# 因为 deepseek-chat 等模型不支持 response_format 参数）
ROUTE_TOKENS = ["rag_agent", "weather_agent", "math_agent", "__end__"]


ROUTER_SYSTEM = """你是一个多 Agent 协作系统的 Supervisor（调度中枢）。
你的唯一职责：根据用户最新的消息，判断应该由哪个专职 Worker 处理，或判断已经可以结束。

可用 Worker：
- rag_agent：企业行政管理政策/制度问答（年假、报销、考勤、出差、社保、公积金、工资/薪酬制度等），
  也负责「列出有哪些知识库/数据源」这类问题（它会调用 list_data_sources 工具）。
  注意：员工个人工资/薪酬查询（如"我的工资是多少"）也属于 rag_agent 范围，它会触发人工审核流程。
- weather_agent：查询任意城市的实时天气
- math_agent：数学计算（支持 + - * / 以及括号、幂运算）

决策规则：
1. 只依据「最新一条 user 消息」选择 Worker，不要管更早的历史。
2. 只有以下情况才输出 __end__：
   - 最新消息是结束语、寒暄或无实质任务；
   - 最新消息已经被当前 Worker 充分回答过（你看到的是 Worker 刚返回的标记消息）。
3. 一次只路由给「一个」Worker。
4. 如果某个问题可以由多个 Worker 处理，选择最匹配的那个。

【输出格式要求】你只能回复以下四个令牌中的【唯一一个】，不要输出任何解释、标点或多余文字：
rag_agent
weather_agent
math_agent
__end__"""


# 当 LLM 不听话、没有输出可识别令牌时的兜底路由表
_ROUTE_FALLBACKS = [
    # 注意：裸「算」太贪婪（"年假怎么算" 会被误判为数学题），
    # 只用明确的数学表达短语 + 运算符，避免政策问答被错误路由到 math_agent
    ("math_agent", ["计算", "算一下", "算一算", "算算", "帮我算", "+", "-", "*", "/", "=", "^", "次方", "平方", "立方", "等于", "多少", "结果"]),
    ("weather_agent", ["天气", "温度", "几度", "下雨", "下雪", "晴天", "阴天", "多云", "风", "气温", "预报"]),
    ("rag_agent", ["工资", "薪酬", "薪资", "年假", "假期", "请假", "病假", "事假", "婚假", "产假", "陪产假", "丧假", "考勤", "迟到", "早退", "加班", "出差", "报销", "社保", "公积金", "五险一金", "制度", "政策", "规定", "流程", "手册", "知识库", "数据源"]),
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


def _build_worker(tools, system_prompt: str):
    """构建一个专职 Worker（create_agent）。
    不自带 checkpointer：对话记忆由外层 StateGraph 统一通过 AsyncSqliteSaver 管理。"""
    return create_agent(model=_make_llm(), tools=tools, system_prompt=system_prompt)


async def _supervisor(state: MessagesState) -> Command[
    Literal["rag_agent", "weather_agent", "math_agent", "__end__"]
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
    async def node(state: MessagesState) -> Command[Literal["supervisor"]]:
        # 关键修复：只把「当前轮次的最后一条用户消息」传给 Agent，
        # 不传递任何历史 AI/Tool/早期 Human 消息。这样每个 Worker 只回答
        # 当前问题，避免被上一轮的 HITL 提示、错误回复或未回答问题干扰。
        last_human = None
        for m in reversed(state["messages"]):
            if getattr(m, "type", "") == "human":
                last_human = m
                break
        filtered_state = {"messages": [last_human]} if last_human else state
        result = await agent.ainvoke(filtered_state)
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
        return Command(goto="supervisor", update={"messages": [reply]})
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
            "rag_agent": _build_worker(
                [query_my_documents, list_data_sources],
                "你是企业行政管理政策/制度问答 Worker，同时也负责回答「有哪些知识库/数据源」。"
                "你的任务：仅针对对话历史中最后一条 user 消息进行回答，不要理睬历史里任何尚未回答的其他问题，也不要在回答中主动提及它们。"
                "【强制规则】只要用户问题涉及公司制度、政策、工资、薪酬、年假、报销、考勤、社保、公积金等内容，"
                "你必须调用 query_my_documents 工具检索知识库，绝不能因为历史对话中曾出现过类似问题或审核提示而跳过工具调用。"
                "当用户问「有哪些知识库/你能查什么」时，请使用 list_data_sources 工具列出可用数据源。"
                "只基于检索到的上下文作答，简洁准确，不要编造。",
            ),
            "weather_agent": _build_worker(
                [get_weather],
                "你是天气查询 Worker。"
                "你的任务：仅针对对话历史中最后一条 user 消息进行回答，不要理睬历史里任何尚未回答的其他问题，也不要在回答中主动提及它们。"
                "当用户询问某个城市的天气时，请使用 get_weather 工具查询并回答。",
            ),
            "math_agent": _build_worker(
                [calculate],
                "你是数学计算 Worker。"
                "你的任务：仅针对对话历史中最后一条 user 消息进行回答，不要理睬历史里任何尚未回答的其他问题，也不要在回答中主动提及它们。"
                "当用户提出数学运算需求时，请使用 calculate 工具计算并给出结果。",
            ),
        }

        builder = StateGraph(MessagesState)
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
