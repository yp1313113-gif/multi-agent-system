# tests/test_streaming.py
"""流式输出契约测试（不需要 API Key / 模型权重）。

被测的是 agent.stream_chat 的「事件 → 输出 token」判定逻辑：
  1. Supervisor 的路由决策不能推给用户；
  2. 「去调工具」那一轮的前言文本不能留在最终输出里
     （前言可能很长，光靠缓冲阈值拦不住，必须靠 on_chat_model_end 的重置信号）；
  3. 最终回答必须完整输出；
  4. 整条链路一个 token 都没有时，要回读 checkpointer 兜底，不能静默。

用假的 astream_events 事件流驱动，覆盖真实踩过的 bug：
  "前言 38 字符 > 缓冲阈值 16 字符 → 前言被推出去 → 发现是工具轮 → 必须发重置信号"
"""
import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import agent  # noqa: E402
import memory_store  # noqa: E402


def chunk(content=None, tool_call=False):
    return SimpleNamespace(
        content=content,
        tool_calls=[{"name": "x"}] if tool_call else [],
        tool_call_chunks=[{"index": 0}] if tool_call else [],
    )


def ev_stream(content, run_id, node="policy_agent"):
    return {
        "event": "on_chat_model_stream",
        "run_id": run_id,
        "metadata": {"langgraph_node": node},
        "data": {"chunk": chunk(content=content)},
    }


def ev_tool_chunk(run_id, node="policy_agent"):
    return {
        "event": "on_chat_model_stream",
        "run_id": run_id,
        "metadata": {"langgraph_node": node},
        "data": {"chunk": chunk(tool_call=True)},
    }


def ev_end(run_id, has_tool_calls, node="policy_agent"):
    msg = SimpleNamespace(tool_calls=[{"name": "x"}] if has_tool_calls else [])
    return {
        "event": "on_chat_model_end",
        "run_id": run_id,
        "metadata": {"langgraph_node": node},
        "data": {"output": msg},
    }


class FakeSupervisor:
    def __init__(self, events, final_answer=""):
        self._events = events
        self._final_answer = final_answer

    async def astream_events(self, inputs, config=None, version=None):
        for e in self._events:
            yield e

    async def aget_state(self, config):
        msgs = []
        if self._final_answer:
            msgs = [SimpleNamespace(type="ai", content=self._final_answer,
                                    additional_kwargs={"_worker_reply": True})]
        return SimpleNamespace(values={"messages": msgs})


class FakeWrapper:
    def __init__(self, supervisor):
        self.supervisor = supervisor


@pytest.fixture
def patched(monkeypatch):
    """屏蔽长期记忆抽取（避免测试里真的去调模型）。"""
    async def noop(*a, **k):
        return {}
    monkeypatch.setattr(memory_store, "extract_and_remember", noop)
    monkeypatch.setattr(memory_store, "build_memory_prompt", lambda s: "")

    def install(events, final_answer=""):
        wrapper = FakeWrapper(FakeSupervisor(events, final_answer))
        async def fake_get_wrapper():
            return wrapper
        monkeypatch.setattr(agent, "_get_wrapper", fake_get_wrapper)
    return install


def collect(session="t-stream"):
    """跑一次 stream_chat，收集全部输出 token（同步包装，方便断言）。"""
    async def run():
        out = []
        async for tok in agent.stream_chat("测试问题", session=session):
            out.append(tok)
        return out
    return asyncio.run(run())


def test_preamble_before_tool_call_is_reset(patched):
    """前言比缓冲阈值长时，必须发重置信号把它撤回来（真实踩过的 bug）。"""
    long_preamble = "好的，我先去知识库检索一下相关政策和口径，请稍等"   # 远超 16 字
    events = [
        ev_stream(long_preamble, "r1"),
        ev_tool_chunk("r1"),
        ev_end("r1", has_tool_calls=True),
        ev_stream("最终答案：加计扣除比例是 100%。", "r2"),
        ev_end("r2", has_tool_calls=False),
    ]
    patched(events)
    out = collect()

    assert agent.STREAM_RESET in out, "前言已推送却没有发出重置信号"
    # 重置之后的内容必须是最终答案，且不含前言
    after_reset = "".join(out[out.index(agent.STREAM_RESET) + 1:])
    assert "最终答案" in after_reset
    assert "我先去知识库" not in after_reset


def test_supervisor_node_is_filtered_out(patched):
    """Supervisor 的路由令牌不该推给用户。"""
    events = [
        ev_stream("policy_agent", "r0", node="supervisor"),
        ev_end("r0", has_tool_calls=False, node="supervisor"),
        ev_stream("答案内容", "r1"),
        ev_end("r1", has_tool_calls=False),
    ]
    patched(events)
    out = collect()
    assert "policy_agent" not in "".join(out)
    assert "答案内容" in "".join(out)


def test_tool_internal_llm_call_is_filtered(patched):
    """工具内部的模型调用不能推给用户（真实踩过的重复输出 bug）。

    RAG 工具里自己会调一次模型生成答案，它在 create_agent 的 "tools" 节点里执行。
    不过滤的话，用户会先看到一遍扁平版答案、再看到 Worker 的结构化答案。
    """
    tool_event = {
        "event": "on_chat_model_stream",
        "run_id": "rt",
        "metadata": {
            "langgraph_node": "tools",
            "langgraph_checkpoint_ns": "policy_agent:aaa|tools:bbb",
        },
        "data": {"chunk": chunk(content="工具内部生成的扁平版答案")},
    }
    events = [
        tool_event,
        {
            "event": "on_chat_model_end",
            "run_id": "rt",
            "metadata": {"langgraph_node": "tools",
                         "langgraph_checkpoint_ns": "policy_agent:aaa|tools:bbb"},
            "data": {"output": SimpleNamespace(tool_calls=[])},
        },
        ev_stream("Worker 的结构化答案", "r2"),
        ev_end("r2", has_tool_calls=False),
    ]
    patched(events)
    out = "".join(collect())
    assert "工具内部生成的扁平版答案" not in out
    assert "Worker 的结构化答案" in out


def test_direct_answer_without_tools_is_flushed(patched):
    """没有工具调用的一轮：内容要在 on_chat_model_end 放行，不能丢。"""
    events = [
        ev_stream("直接回答", "r1"),
        ev_end("r1", has_tool_calls=False),
    ]
    patched(events)
    out = collect()
    assert "直接回答" in "".join(out)
    assert agent.STREAM_RESET not in out


def test_short_preamble_is_discarded_without_reset(patched):
    """前言很短（未超阈值）时，应在缓冲阶段被丢弃，不必发重置信号。"""
    events = [
        ev_stream("查一下", "r1"),          # 3 字，未满 16
        ev_tool_chunk("r1"),
        ev_end("r1", has_tool_calls=True),
        ev_stream("最终答案", "r2"),
        ev_end("r2", has_tool_calls=False),
    ]
    patched(events)
    out = collect()
    assert agent.STREAM_RESET not in out
    assert "最终答案" in "".join(out)
    assert "查一下" not in "".join(out)


def test_fallback_reads_checkpointer_when_nothing_streamed(patched):
    """流式一个 token 都没出时，要回读状态兜底，保证一定有回答。"""
    patched([], final_answer="兜底答案")
    out = collect()
    assert "兜底答案" in "".join(out)


def test_fallback_greeting_when_state_empty(patched):
    """连状态里也没有答案时（寒暄被路由到 __end__），给能力清单而不是报错。"""
    patched([])
    out = collect()
    text = "".join(out)
    assert "研发费用" in text and "你可以问我" in text
