# tests/test_supervisor_agents.py
"""
Supervisor-Worker 多 Agent 编排层测试：四 Worker 的 happy path / 拒答 / 跨轮。

背景：真实 LLM 行为（检索质量、工具调用、拒答话术）依赖 API Key 与模型权重，
在 CI / 离线环境不可复现。因此本模块把「模型层」替换为确定性替身：

  - FakeRouter  按关键词返回路由令牌（复用 supervisor._fallback_route 的规则），
                替代 LLM 路由决策；
  - FakeWorker  记录收到的消息并返回固定答复，替代 create_agent 构建的专职 Worker。

被测试的是「编排层」自身的确定性逻辑——这正是 README 中
「pytest 覆盖四 Worker 的 happy path / 拒答 / 跨轮」所指：
  1. happy path：三个 Worker 各自领域的问题被正确路由并返回答复；
  2. 拒答      ：范围外/寒暄输入由 Supervisor 直接终止（__end__），不硬答；
                 路由令牌无法识别时走关键词兜底，不抛错、不串台；
  3. 跨轮      ：多轮对话下 Worker 只收到「最后一条 human 消息」（防串台修复），
                 且 AsyncSqliteSaver 按 thread_id 持久化记忆。

运行：python -m pytest tests/test_supervisor_agents.py -q
"""
import asyncio
import os
import sys
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

# 把项目根目录加入 path，便于直接运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import supervisor  # noqa: E402  (需在 sys.path 设置之后导入)


# ---------------------------------------------------------------------------
# 确定性替身：假 Router（替代 LLM 路由决策）+ 假 Worker（替代 create_agent）
# ---------------------------------------------------------------------------

# 各 Worker 的固定答复（假 Worker 直接返回，不触发真实工具）
REPLIES = {
    "policy_agent": "已检索知识库：根据《研发费用政策库》，加计扣除比例 100%……",
    "expense_agent": "已归集：研发服务器电费 → 直接投入费用 ¥3,200。",
    "risk_agent": "🟢 未发现明显风险，指标均在合规范围内",
    "fill_agent": "✅ 工时单已生成（FakeWorker 固定答复）",
}


def _worker_name(tools) -> str:
    """按工具集合识别 Worker 身份（与 supervisor.initialize 的注册顺序无关）。"""
    fnames = {
        getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools
    }
    if "query_my_documents" in fnames or "list_data_sources" in fnames:
        return "policy_agent"
    if "classify_expense" in fnames or "list_rd_expenses" in fnames:
        return "expense_agent"
    if "scan_rd_risk" in fnames or "list_risk_indicators" in fnames:
        return "risk_agent"
    if "fill_timesheet" in fnames:
        return "fill_agent"
    return "unknown"


class FakeRouter:
    """假 LLM：按关键词返回路由令牌（规则与 supervisor._fallback_route 一致）。"""

    def __init__(self, registry):
        self.registry = registry

    async def ainvoke(self, messages):
        self.registry["router_calls"].append(len(messages))
        text = ""
        for m in reversed(messages):
            if getattr(m, "type", None) == "human":
                text = str(getattr(m, "content", "") or "")
                break
        return SimpleNamespace(content=supervisor._fallback_route(text))


class GarbageRouter(FakeRouter):
    """假 LLM：返回无法识别的令牌，强制触发关键词兜底路径。"""

    async def ainvoke(self, messages):
        return SimpleNamespace(content="？？？无法识别的路由输出？？？")


class FakeWorker:
    """假专职 Worker：记录收到的消息并返回固定答复（替代 create_agent）。"""

    def __init__(self, name, reply):
        self.name = name
        self.reply = reply
        self.received = []  # 每次被调用时收到的 state["messages"]（(type, content) 列表）
        self.calls = 0

    async def ainvoke(self, state):
        self.calls += 1
        self.received.append(
            [(m.type, str(m.content)) for m in state["messages"]]
        )
        return {
            "messages": [
                AIMessage(content=self.reply, additional_kwargs={"_worker_reply": True})
            ]
        }


# ---------------------------------------------------------------------------
# fixtures 与事件循环管理
# （注意：AsyncSqliteSaver 内部的 asyncio.Lock 绑定创建时的事件循环，
#   因此「构建 + 全部 invoke + 关闭」必须在同一个事件循环内完成，
#   不能为每次调用单独 asyncio.run()。）
# ---------------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch):
    """把 supervisor 的模型层替换为确定性替身，返回可断言的注册表。"""
    registry = {"workers": {}, "router_calls": []}

    def fake_router():
        return FakeRouter(registry)

    def fake_build_worker(tools, system_prompt):
        name = _worker_name(tools)
        worker = FakeWorker(name, REPLIES.get(name, "（未知 Worker）"))
        registry["workers"][name] = worker
        return worker

    monkeypatch.setattr(supervisor, "_make_llm", fake_router)
    monkeypatch.setattr(supervisor, "_build_worker", fake_build_worker)
    return registry


def _run(coro):
    """在单一事件循环中执行异步测试体。"""
    return asyncio.run(coro)


async def _invoke(wrapper, message: str, thread_id: str):
    """以指定 thread_id 发送一条用户消息，返回最终状态（须在 wrapper 的同一事件循环内 await）。"""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "request_id": f"req_{uuid.uuid4().hex[:8]}",
        },
        "recursion_limit": 10,
    }
    return await wrapper.supervisor.ainvoke(
        {"messages": [{"role": "user", "content": message}]}, config=config
    )


def _final_ai(result) -> str:
    """取最终状态里最后一条 AI 消息的文本。"""
    for m in reversed(result["messages"]):
        if m.type == "ai":
            return str(m.content)
    return ""


def _case(fn):
    """把 async 测试体包成同步 pytest 用例（单事件循环内完成 build/invoke/close）。"""

    def wrapper(env):
        async def body():
            wrapper_obj = await supervisor.create_supervisor()
            try:
                return await fn(env, wrapper_obj)
            finally:
                await wrapper_obj.close()

        return _run(body())

    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# 1. happy path：三个 Worker 各自领域的问题被正确路由并返回答复
# ---------------------------------------------------------------------------

@_case
async def test_happy_path_rag(env, wrapper):
    result = await _invoke(wrapper, "研发费用加计扣除怎么算？", "t-happy-rag")
    assert REPLIES["policy_agent"] in _final_ai(result)
    assert env["workers"]["policy_agent"].calls == 1
    assert env["workers"]["expense_agent"].calls == 0
    assert env["workers"]["risk_agent"].calls == 0


@_case
async def test_happy_path_weather(env, wrapper):
    result = await _invoke(wrapper, "帮我把研发服务器电费归集到直接投入费用", "t-happy-weather")
    assert REPLIES["expense_agent"] in _final_ai(result)
    assert env["workers"]["expense_agent"].calls == 1
    assert env["workers"]["risk_agent"].calls == 0


@_case
async def test_happy_path_math(env, wrapper):
    result = await _invoke(wrapper, "扫描一下研发费用风险：其他相关费用占比 15%", "t-happy-math")
    assert REPLIES["risk_agent"] in _final_ai(result)
    assert env["workers"]["risk_agent"].calls == 1
    assert env["workers"]["policy_agent"].calls == 0


# ---------------------------------------------------------------------------
# 2. 拒答：范围外/寒暄输入由 Supervisor 直接终止，不硬答、不误路由
# ---------------------------------------------------------------------------

@_case
async def test_refuse_out_of_scope_ends_without_worker(env, wrapper):
    """寒暄/无实质任务 → supervisor 路由 __end__，任何 Worker 都不被调用。"""
    result = await _invoke(wrapper, "你好，在吗？", "t-refuse-1")
    # 最终状态只有用户消息，没有 AI 答复
    assert len(result["messages"]) == 1
    assert result["messages"][0].type == "human"
    for name, worker in env["workers"].items():
        assert worker.calls == 0, f"{name} 不应被调用"


@_case
async def test_refuse_after_answered_conversation(env, wrapper):
    """先正常问答，再发结束语 → 不再追加 Worker 调用，不硬答。"""
    await _invoke(wrapper, "扫描一下研发费用风险：其他相关费用占比 15%", "t-refuse-2")
    before = env["workers"]["risk_agent"].calls
    assert before == 1
    result = await _invoke(wrapper, "谢谢，没有其他问题了", "t-refuse-2")
    # 第 2 轮只追加了 human 消息，没有新的 AI 答复
    assert env["workers"]["risk_agent"].calls == before
    assert len(result["messages"]) == 3  # H1, A1, H2
    assert result["messages"][-1].type == "human"


@_case
async def test_router_unrecognized_token_falls_back(env, wrapper):
    """路由令牌无法识别 → 关键词兜底路由，不抛错、仍能正确分派。"""
    # GarbageRouter 通过 fixture 之外的 monkeypatch 注入：
    # 这里直接覆写 supervisor._make_llm 指向垃圾路由
    import supervisor as _sup
    _sup._make_llm = lambda: GarbageRouter(env)
    result = await _invoke(wrapper, "帮我把研发服务器电费归集到直接投入费用", "t-refuse-3")
    assert REPLIES["expense_agent"] in _final_ai(result)
    assert env["workers"]["expense_agent"].calls == 1


# ---------------------------------------------------------------------------
# 3. 跨轮：多轮记忆 + Worker 只收到最后一条 human 消息（防串台修复）
# ---------------------------------------------------------------------------

@_case
async def test_cross_turn_memory_and_no_bleed(env, wrapper):
    """同一 thread 两轮问答：记忆累计、答案不串台。"""
    r1 = await _invoke(wrapper, "扫描一下研发费用风险：其他相关费用占比 15%", "t-cross-1")
    assert REPLIES["risk_agent"] in _final_ai(r1)

    r2 = await _invoke(wrapper, "帮我把研发服务器电费归集到直接投入费用", "t-cross-1")
    # 最终答复是天气，不是上一轮的数学答案（无跨轮串扰）
    assert REPLIES["expense_agent"] in _final_ai(r2)
    assert REPLIES["risk_agent"] not in _final_ai(r2)
    # 记忆按 thread 累计：H1, A1, H2, A2 共 4 条
    assert len(r2["messages"]) == 4


@_case
async def test_cross_turn_worker_only_sees_last_human(env, wrapper):
    """防串台核心修复：Worker 收到的输入被压缩为「最后一条 human 消息」。"""
    await _invoke(wrapper, "扫描一下研发费用风险：其他相关费用占比 15%", "t-cross-2")
    await _invoke(wrapper, "帮我把研发服务器电费归集到直接投入费用", "t-cross-2")
    weather = env["workers"]["expense_agent"]
    # 天气 Worker 被调用时，只收到本轮 human 消息
    assert weather.calls == 1
    assert weather.received[-1] == [("human", "帮我把研发服务器电费归集到直接投入费用")]


# ---------------------------------------------------------------------------
# 4. 关键词兜底路由规则（纯函数单测，覆盖拒答判定）
# ---------------------------------------------------------------------------

def test_fallback_route_rules():
    assert supervisor._fallback_route("扫描一下研发费用风险") == "risk_agent"
    assert supervisor._fallback_route("把电费归集到直接投入费用") == "expense_agent"
    assert supervisor._fallback_route("研发费用加计扣除怎么算") == "policy_agent"
    assert supervisor._fallback_route("帮王建国填报今天在恒泰项目的6小时工时") == "fill_agent"
    assert supervisor._fallback_route("你好") == "__end__"
    assert supervisor._fallback_route("") == "__end__"
    assert supervisor._fallback_route("随便聊聊") == "__end__"
