# tests/test_cost.py
"""Token 成本埋点测试：记账正确性 + 阈值告警 + 解析健壮性。"""
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cost_tracker  # noqa: E402


@pytest.fixture
def cost(tmp_path, monkeypatch):
    """指向临时库，避免污染项目根目录的 cost.db。"""
    monkeypatch.setattr(cost_tracker, "DB_PATH", str(tmp_path / "cost_test.db"))
    return cost_tracker


# ---------------------------------------------------------------------------
# 1. 记账与汇总
# ---------------------------------------------------------------------------

def test_record_and_session_summary(cost):
    cost.record_usage("s1", "deepseek", "supervisor", 100, 10)
    cost.record_usage("s1", "deepseek", "policy_agent", 500, 200)

    s = cost.session_summary("s1")
    assert s["calls"] == 2
    assert s["prompt_tokens"] == 600
    assert s["completion_tokens"] == 210
    assert s["total_tokens"] == 810
    # 按节点拆分：能看出钱花在哪个节点上（这才是埋点的意义）
    assert set(s["by_node"]) == {"supervisor", "policy_agent"}
    assert s["by_node"]["policy_agent"]["completion_tokens"] == 200


def test_sessions_are_isolated(cost):
    cost.record_usage("s1", "m", "n", 100, 100)
    cost.record_usage("s2", "m", "n", 5, 5)
    assert cost.session_summary("s1")["total_tokens"] == 200
    assert cost.session_summary("s2")["total_tokens"] == 10
    assert cost.session_summary("nobody")["calls"] == 0


def test_global_summary(cost):
    cost.record_usage("s1", "m", "n", 100, 0)
    cost.record_usage("s2", "m", "n", 50, 0)
    g = cost.global_summary()
    assert g["calls"] == 2
    assert g["sessions"] == 2
    assert g["prompt_tokens"] == 150


# ---------------------------------------------------------------------------
# 2. 费用计算（单价未配置时应只统计 token、不算错钱）
# ---------------------------------------------------------------------------

def test_cost_is_zero_when_price_not_configured(cost, monkeypatch):
    from config import config
    monkeypatch.setattr(config, "PRICE_INPUT_PER_M", 0.0, raising=False)
    monkeypatch.setattr(config, "PRICE_OUTPUT_PER_M", 0.0, raising=False)
    assert cost.compute_cost(1_000_000, 1_000_000) == 0.0


def test_cost_uses_configured_price(cost, monkeypatch):
    from config import config
    monkeypatch.setattr(config, "PRICE_INPUT_PER_M", 1.0, raising=False)   # 1 元/百万
    monkeypatch.setattr(config, "PRICE_OUTPUT_PER_M", 2.0, raising=False)  # 2 元/百万
    assert cost.compute_cost(1_000_000, 1_000_000) == 3.0
    assert cost.compute_cost(500_000, 0) == 0.5


# ---------------------------------------------------------------------------
# 3. 预算告警（成本失控的发现机制）
# ---------------------------------------------------------------------------

def test_budget_alert_triggers_over_threshold(cost, monkeypatch):
    from config import config
    monkeypatch.setattr(config, "PRICE_OUTPUT_PER_M", 1_000_000.0, raising=False)  # 放大便于触发
    monkeypatch.setattr(config, "COST_ALERT_PER_SESSION", 1.0, raising=False)

    cost.record_usage("s1", "m", "n", 0, 2)          # 2 token * 1e6/1e6 = 2 元
    assert cost.check_budget("s1") is True           # 2 > 阈值 1 → 告警（边界相等不做超限）


def test_no_alert_when_threshold_disabled(cost, monkeypatch):
    from config import config
    monkeypatch.setattr(config, "COST_ALERT_PER_SESSION", 0.0, raising=False)
    cost.record_usage("s1", "m", "n", 9_999_999, 9_999_999)
    assert cost.check_budget("s1") is False


# ---------------------------------------------------------------------------
# 4. usage 解析：不同 langchain 版本返回结构不同，都要能取到
# ---------------------------------------------------------------------------

def test_extract_usage_from_llm_output(cost):
    resp = SimpleNamespace(llm_output={"token_usage": {"prompt_tokens": 12, "completion_tokens": 34},
                                       "model_name": "deepseek"})
    assert cost._extract_usage(resp) == (12, 34, "deepseek")


def test_extract_usage_from_usage_metadata(cost):
    msg = SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 8},
                          response_metadata={"model_name": "m2"})
    resp = SimpleNamespace(llm_output={}, generations=[[SimpleNamespace(message=msg)]])
    assert cost._extract_usage(resp) == (7, 8, "m2")


def test_extract_usage_from_chat_result_flat(cost):
    """ChatResult 是扁平结构（generations[0].message），astream_events 走这条路径。

    只兼容 LLMResult 的嵌套结构会导致 on_chat_model_end 静默记不到账 —— 这个坑踩过。
    """
    msg = SimpleNamespace(usage_metadata={"input_tokens": 11, "output_tokens": 22},
                          response_metadata={"model_name": "m3"})
    resp = SimpleNamespace(llm_output={}, generations=[SimpleNamespace(message=msg)])
    assert cost._extract_usage(resp) == (11, 22, "m3")


def test_extract_usage_from_response_metadata_token_usage(cost):
    """DeepSeek/OpenAI 兼容接口有时只把用量放在 response_metadata.token_usage 里。"""
    msg = SimpleNamespace(usage_metadata={},
                          response_metadata={"token_usage": {"prompt_tokens": 33, "completion_tokens": 44}})
    resp = SimpleNamespace(llm_output={}, generations=[SimpleNamespace(message=msg)])
    assert cost._extract_usage(resp) == (33, 44, "")


def test_extract_usage_from_bare_aimessage(cost):
    """astream_events 的 on_chat_model_end 直接给 AIMessage（本机实测结构）。

    按 LLMResult 解析会静默返回 (0,0,'')，成本就悄悄丢了 —— 这个坑踩过。
    """
    msg = SimpleNamespace(usage_metadata={"input_tokens": 88, "output_tokens": 99},
                          response_metadata={"model_name": "deepseek-v4-flash"})
    assert cost._extract_usage(msg) == (88, 99, "deepseek-v4-flash")


def test_record_from_event_attributes_node(cost, monkeypatch):
    """事件记账要能定位到具体节点（Supervisor 路由 vs 某个 Worker）。"""
    called = {}
    monkeypatch.setattr(cost, "record_usage",
                        lambda s, m, n, p, c: called.update(session=s, model=m, node=n, pt=p, ct=c))
    msg = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 20},
                          response_metadata={"model_name": "m"})
    event = {"event": "on_chat_model_end",
             "data": {"output": msg},
             "metadata": {"langgraph_node": "supervisor"}}
    cost.record_from_event(event, "s1")
    assert called["node"] == "supervisor"
    assert called["pt"] == 10 and called["ct"] == 20


def test_record_from_event_resolves_worker_from_checkpoint_ns(cost, monkeypatch):
    """子图内部的 langgraph_node 会退化成 "model"，要用 checkpoint_ns 最外层还原 Worker。"""
    called = {}
    monkeypatch.setattr(cost, "record_usage",
                        lambda s, m, n, p, c: called.update(node=n))
    msg = SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 2},
                          response_metadata={})
    event = {"event": "on_chat_model_end",
             "data": {"output": msg},
             "metadata": {"langgraph_node": "model",
                          "langgraph_checkpoint_ns": "policy_agent:abc|model:def"}}
    cost.record_from_event(event, "s1")
    assert called["node"] == "policy_agent"


def test_extract_usage_returns_zero_on_unknown_shape(cost):
    assert cost._extract_usage(SimpleNamespace(llm_output={}, generations=[])) == (0, 0, "")


def test_callback_never_raises_on_bad_response(cost):
    """回调解析失败绝不能把主流程带崩 —— 记账是旁路，不是主路。"""
    cb = cost.CostTrackerCallback("s1")
    cb.on_llm_end(SimpleNamespace(llm_output=None))       # 不抛异常即通过
    cb.on_llm_end(object())
