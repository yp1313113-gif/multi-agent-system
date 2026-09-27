# tests/test_middleware.py
"""中间件链的测试。

重点同样是**不变量与失败路径**：

  1. 洋葱顺序必须正确（重试在外层 → **每次重试都有独立超时**）；
  2. 超时必须「真的超时」，不能等任务跑完才抛；
  3. contextvars 必须跨线程传播（否则工具线程读不到 request_id / 发起人）；
  4. 中间件抛异常不能带崩主流程，但**必须留痕**（不静默降级）；
  5. 原始消息列表不能被就地修改；
  6. scope 过滤必须生效（记忆不能注入路由）；
  7. 审批守卫缺失时是 no-op，已批准时消费一次后放行。
"""
import asyncio
import contextvars
import os
import time

import pytest

import middleware as M


# ============================================================
# 底层原语
# ============================================================

def test_run_with_timeout_returns_value():
    assert M.run_with_timeout(lambda: 42, 5) == 42


def test_run_with_timeout_actually_times_out():
    """超时必须**真的**超时。

    回归测试：曾经写成 `with ThreadPoolExecutor(...) as ex:`，
    `with` 退出时 shutdown(wait=True) 会等任务跑完，
    于是 2 秒超时在 6 秒任务上要 6 秒才抛 —— 超时形同虚设。
    """
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        M.run_with_timeout(lambda: time.sleep(3), 0.3)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5, f"超时没有立即生效（耗时 {elapsed:.2f}s）"


def test_run_with_timeout_propagates_contextvars():
    """线程池不自动继承 contextvars —— copy_context 必须显式传。"""
    var = contextvars.ContextVar("probe", default="default")
    var.set("set-in-caller")
    assert M.run_with_timeout(lambda: var.get(), 5) == "set-in-caller"


def test_run_with_retry_counts_total_attempts():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert M.run_with_retry(flaky, max_retries=5, delay=0) == "ok"
    assert calls["n"] == 3


def test_run_with_retry_raises_after_exhausted():
    def always_fail():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        M.run_with_retry(always_fail, max_retries=2, delay=0)


# ============================================================
# 洋葱顺序
# ============================================================

class _Recorder(M.Middleware):
    def __init__(self, tag, order, trace):
        self.name = tag
        self.order = order
        self.trace = trace

    def call(self, call, next_fn):
        self.trace.append(f"{self.name}:in")
        r = next_fn()
        self.trace.append(f"{self.name}:out")
        return r


def test_tool_chain_onion_order():
    """order 小的在外层：先进后出。"""
    trace = []
    chain = M.MiddlewareChain()
    chain.add_tool(_Recorder("outer", 10, trace))
    chain.add_tool(_Recorder("inner", 20, trace))
    chain.run_tool(lambda: "done", "t")
    assert trace == ["outer:in", "inner:in", "inner:out", "outer:out"]


def test_retry_is_outer_so_each_attempt_has_own_timeout():
    """重试在外层 → 每次重试都有独立的超时（而不是整个重试共享一次超时）。"""
    attempts = []

    def flaky():
        attempts.append(time.perf_counter())
        raise ValueError("always")

    chain = M.MiddlewareChain()
    chain.add_tool(M.RetryMiddleware())
    chain.add_tool(M.TimeoutMiddleware())
    # 直连执行（pytest 里 config 的默认重试次数可能是 2）
    with pytest.raises(Exception):
        chain.run_tool(flaky, "flaky", __max_retries__=3, __timeout__=5)
    assert len(attempts) == 3          # 重试确实发生了 3 次
    # 每次尝试都进了内层（TimeoutMiddleware）—— 证明重试在外层
    assert [m["name"] for m in chain.describe()["tool"]] == ["retry", "timeout"]


# ============================================================
# 工具边界中间件
# ============================================================

def test_loop_guard_blocks_after_limit():
    token = M.new_loop_guard(max_turns=10, max_tool_calls=2)
    try:
        chain = M.MiddlewareChain()
        chain.add_tool(M.LoopGuardMiddleware())
        assert chain.run_tool(lambda: "ok", "rag") == "ok"
        assert chain.run_tool(lambda: "ok", "rag") == "ok"
        third = chain.run_tool(lambda: "SHOULD_NOT_RUN", "rag")
        assert "超限" in third          # 被拦截，返回提示而不是结果
    finally:
        M.reset_loop_guard(token)


def test_loop_guard_noop_without_guard():
    """没设置守卫时必须是 no-op —— 否则单元测试/CLI 会被误拦。"""
    chain = M.MiddlewareChain()
    chain.add_tool(M.LoopGuardMiddleware())
    for _ in range(10):
        assert chain.run_tool(lambda: "ok", "rag") == "ok"


def test_timing_middleware_logs_and_reraises():
    chain = M.MiddlewareChain()
    chain.add_tool(M.TimingMiddleware())

    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        chain.run_tool(boom, "boom")
    assert chain.run_tool(lambda: "fine", "fine") == "fine"


def test_result_compress_offloads_large_string(tmp_path, monkeypatch):
    import compression
    monkeypatch.setattr(compression, "DEFAULT_OFFLOAD_DIR", str(tmp_path))

    chain = M.MiddlewareChain()
    chain.add_tool(M.ResultCompressMiddleware())
    huge = "政策条款内容。" * 3000
    out = chain.run_tool(lambda: huge, "query_my_documents")
    assert len(out) < len(huge)
    assert "路径：" in out


def test_result_compress_ignores_small_and_non_string():
    chain = M.MiddlewareChain()
    chain.add_tool(M.ResultCompressMiddleware())
    assert chain.run_tool(lambda: "短结果", "t") == "短结果"
    obj = {"a": 1}
    assert chain.run_tool(lambda: obj, "t") is obj


def test_metrics_counts_success_and_failure():
    chain = M.MiddlewareChain()
    mw = M.MetricsMiddleware()
    chain.add_tool(mw)
    chain.run_tool(lambda: 1, "a")
    chain.run_tool(lambda: 2, "a")
    with pytest.raises(ValueError):
        chain.run_tool(lambda: (_ for _ in ()).throw(ValueError("x")), "a")
    snap = mw.snapshot()["a"]
    assert snap["calls"] == 3 and snap["ok"] == 2 and snap["fail"] == 1


# ============================================================
# 审批中间件
# ============================================================

class _FakeHitl:
    def __init__(self, approved=False):
        self.approved = approved
        self.requested = []
        self.consumed = []

    def has_approved_query(self, key, name, requester=None):
        return self.approved

    def consume_approved_query(self, key, name, requester=None):
        self.consumed.append((key, name))
        self.approved = False

    def request_approval(self, **kw):
        self.requested.append(kw)
        return kw.get("approval_id", "approval_x")


@pytest.fixture()
def fake_hitl(monkeypatch):
    import hitl
    fake = _FakeHitl()
    for fn in ("has_approved_query", "consume_approved_query", "request_approval"):
        monkeypatch.setattr(hitl, fn, getattr(fake, fn))
    return fake


def _build_approval_chain(need=True):
    mw = M.ApprovalMiddleware()
    mw.register_guard("my_tool", lambda a, k: M.ApprovalRequest(
        need=need, kind="salary_detail", reason="个人薪酬明细", key="Q1", record_name="rag_search"))
    chain = M.MiddlewareChain()
    chain.add_tool(mw)
    return chain


def test_approval_blocks_and_creates_request(fake_hitl):
    chain = _build_approval_chain(need=True)
    out = chain.run_tool(lambda: "SHOULD_NOT_RUN", "my_tool")
    assert "需要人工复核" in out
    assert len(fake_hitl.requested) == 1
    assert fake_hitl.requested[0]["tool_name"] == "rag_search"     # record_name 生效
    assert fake_hitl.requested[0]["kind"] == "salary_detail"


def test_approval_passes_and_consumes_when_already_approved(fake_hitl):
    fake_hitl.approved = True
    chain = _build_approval_chain(need=True)
    assert chain.run_tool(lambda: "RAN", "my_tool") == "RAN"
    assert fake_hitl.consumed == [("Q1", "rag_search")]            # 消费一次，不是永久放行
    assert not fake_hitl.requested


def test_approval_noop_when_guard_absent(fake_hitl):
    chain = M.MiddlewareChain()
    chain.add_tool(M.ApprovalMiddleware())
    assert chain.run_tool(lambda: "RAN", "unregistered_tool") == "RAN"
    assert not fake_hitl.requested


def test_approval_guard_exception_does_not_block(fake_hitl):
    """守卫写坏了应该告警放行，而不是把工具拦死。"""
    mw = M.ApprovalMiddleware()

    def bad_guard(a, k):
        raise RuntimeError("守卫有 bug")

    mw.register_guard("t", bad_guard)
    chain = M.MiddlewareChain()
    chain.add_tool(mw)
    assert chain.run_tool(lambda: "RAN", "t") == "RAN"


# ============================================================
# 模型边界中间件
# ============================================================

def test_model_chain_order():
    trace = []

    class Rec(M.Middleware):
        def __init__(self, tag, order):
            self.name, self.order = tag, order

        def before_model(self, req):
            trace.append(self.name)

    chain = M.MiddlewareChain()
    chain.add_model(Rec("b", 20))
    chain.add_model(Rec("a", 10))
    chain.build([{"type": "human", "content": "hi"}])
    assert trace == ["a", "b"]        # 按 order 升序


def test_model_chain_scope_filtering():
    """记忆/黑板只对 worker 生效 —— 路由注入记忆会把路由带偏。"""
    chain = M.build_model_chain()
    assert [m.name for m in chain._models if "supervisor" in m.scopes] == ["compression"]
    assert [m.name for m in chain._models if "worker" in m.scopes] == ["memory", "board"]


def test_model_middleware_failure_is_logged_not_fatal():
    class Boom(M.Middleware):
        name, order = "boom", 10

        def before_model(self, req):
            raise RuntimeError("中间件炸了")

    chain = M.MiddlewareChain()
    chain.add_model(Boom())
    req = chain.build([{"type": "human", "content": "hi"}])
    assert any("boom:error" in n for n in req.notes)   # 留痕，不静默
    assert req.messages                                # 主流程继续


def test_model_chain_does_not_mutate_input_messages():
    msgs = [{"type": "human", "content": "hi"}]
    snapshot = [dict(m) for m in msgs]
    chain = M.build_model_chain()
    chain.build(msgs, scope="worker", state=None)
    assert msgs == snapshot


def test_memory_payload_can_be_compressed():
    """模型链至少得能把消息列表收短 —— 否则等于没接。"""
    msgs = [{"type": "human", "content": "第%d轮" % i} for i in range(40)]
    msgs.append({"type": "human", "content": "最近一句"})
    chain = M.build_model_chain()
    req = chain.build(msgs, scope="supervisor", state=None)
    assert len(req.messages) <= 12, "路由视图没有被窗口截断"
    assert "compression" in req.notes[0]


# ============================================================
# harness 兼容出口
# ============================================================

def test_harness_reexports_still_work():
    import harness
    assert harness.AgentLoopGuard is M.AgentLoopGuard
    assert harness.run_with_timeout is M.run_with_timeout
    assert callable(harness.guarded_tool)
    assert harness.with_timeout(5)(lambda: "ok")() == "ok"
    assert harness.with_retry(2, 0)(lambda: "ok")() == "ok"


def test_guarded_tool_runs_through_chain():
    import harness

    def plain(a, b=2):
        return a + b

    t = harness.guarded_tool(plain)
    func = getattr(t, "func", t)
    assert func({"a": 1, "b": 5}) == 6      # 兼容 dict 入参
    assert func(3, b=4) == 7
