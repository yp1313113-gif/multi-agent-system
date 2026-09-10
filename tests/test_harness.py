# tests/test_harness.py
"""Harness 执行保障层测试：重点验证「超时是真的超时」。"""
import concurrent.futures
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import with_timeout, with_retry, AgentLoopGuard  # noqa: E402


def test_timeout_actually_bounds_wall_clock():
    """超时必须真的在 N 秒内返回。

    踩过的坑：原来写成 `with ThreadPoolExecutor(...) as ex:`，
    with 块退出会执行 shutdown(wait=True)，也就是等任务跑完 ——
    结果 2 秒超时在 6 秒的任务上要 6.01 秒才抛出，超时完全不生效。
    """
    def slow():
        time.sleep(3)
        return "done"

    t0 = time.perf_counter()
    with pytest.raises(concurrent.futures.TimeoutError):
        with_timeout(1)(slow)()
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.8, f"超时未生效：耗时 {elapsed:.2f}s（应在 1s 左右返回）"


def test_timeout_returns_normally_when_fast_enough():
    def quick():
        return "ok"

    assert with_timeout(5)(quick)() == "ok"


def test_retry_eventually_succeeds():
    calls = {"n": 0}

    @with_retry(max_retries=3, delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_raises_last_error():
    @with_retry(max_retries=2, delay=0.01)
    def always_fail():
        raise ValueError("always")

    with pytest.raises(ValueError):
        always_fail()


def test_loop_guard_blocks_runaway_turns():
    g = AgentLoopGuard(max_turns=3, max_tool_calls=1)
    assert [g.check() for _ in range(3)] == [True, True, True]
    assert g.check() is False          # 第 4 轮应被拦截（防死循环）


def test_loop_guard_blocks_tool_abuse():
    g = AgentLoopGuard(max_turns=10, max_tool_calls=1)
    assert g.record_tool("rag") is True
    assert g.record_tool("rag") is False   # 同一工具超限 → 拦截
