# harness.py
"""Agent Harness —— **兼容出口**。

━━━ 说明 ━━━
本模块的实现已经迁移到 `middleware.py`（中间件链）。这里保留原来的公开接口，
一是为了不让既有调用点（`supervisor` / `skills.registry` / 测试）改动，
二是为了让「工具执行护栏」这个概念仍然有一个好记的入口。

    harness.with_timeout   → middleware.run_with_timeout
    harness.with_retry     → middleware.run_with_retry
    harness.AgentLoopGuard → middleware.AgentLoopGuard
    harness.guarded_tool   → middleware.TOOL_CHAIN.run_tool

**没有重复实现**：超时线程池那段最容易写错（见 `run_with_timeout` 的注释），
只允许有一份。改实现请改 `middleware.py`。

━━━ Agent Loop 架构里的位置 ━━━
思考 → 行动 → 观察 → 循环。Harness 负责「执行环境」这一侧：
工具执行的超时、有限重试、耗时日志、循环守卫。
业务 Agent（Supervisor / Worker）只负责「决策与工具选择」。
"""
from __future__ import annotations

from functools import wraps

from loguru import logger

from middleware import (          # noqa: F401  —— 兼容导出
    AgentLoopGuard,
    Middleware,
    MiddlewareChain,
    TOOL_CHAIN,
    new_loop_guard,
    reset_loop_guard,
    current_loop_guard,
    run_with_retry,
    run_with_timeout,
)

__all__ = [
    "guarded_tool", "with_timeout", "with_retry", "AgentLoopGuard",
    "new_loop_guard", "reset_loop_guard", "current_loop_guard",
]


def _normalize(args, kwargs):
    """兼容 LangChain 的调用方式：tool_input 可能是 dict（位置或关键字）。"""
    if args and isinstance(args[0], dict):
        return (), {**args[0], **kwargs}
    return args, kwargs


def with_timeout(seconds: float = 10):
    """给同步函数加超时（线程池实现，超时抛 TimeoutError）。

    实现见 `middleware.run_with_timeout` —— 注意那里关于
    「为什么不能用 `with ThreadPoolExecutor(...)`」的注释。
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            a, k = _normalize(args, kwargs)
            return run_with_timeout(lambda: fn(*a, **k), seconds)

        return wrapper

    return decorator


def with_retry(max_retries: int = 2, delay: float = 0.5):
    """有限重试（指数退避）。`max_retries` 是**总尝试次数**。"""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            a, k = _normalize(args, kwargs)
            return run_with_retry(lambda: fn(*a, **k), max_retries, delay)

        return wrapper

    return decorator


def guarded_tool(tool, timeout: float = None, max_retries: int = None):
    """把 LangChain 工具包装为受控工具：**审批 → 循环守卫 → 重试 → 超时 → 计时 → 结果卸载 → 指标**。

    超时/重试默认读 config.TOOL_TIMEOUT / TOOL_MAX_RETRIES（未配置则 10s / 2 次）。
    保持 LangChain Tool 接口（name/description），可无缝接入 create_agent。

    与旧实现的差别：护栏不再是「一个装饰器里套两层」，而是一条**有序链**。
    好处是每一层可以独立开关、独立排序、独立观测（见 `GET /middleware`）。
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        from langchain.tools import StructuredTool

    # 兼容两种工具形态：LangChain BaseTool（有 func/name）或普通函数（测试 stub）
    func = getattr(tool, "func", None) or tool
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
    description = getattr(tool, "description", None) or "harness guarded tool"

    def wrapped(*args, **kwargs):
        a, k = _normalize(args, kwargs)
        return TOOL_CHAIN.run_tool(func, name, *a, **{"__timeout__": timeout, "__max_retries__": max_retries, **k})

    # 优先 tool.copy 保留原 args_schema（langchain 解析 dict → func 参数正确）
    try:
        if hasattr(tool, "copy"):
            return tool.model_copy(update={"func": wrapped})
    except Exception as e:      # pragma: no cover
        logger.debug(f"[harness] tool.model_copy 不可用，改用 StructuredTool 重建: {e}")
    # 普通函数（测试 stub 等）兜底：重建 StructuredTool
    return StructuredTool.from_function(func=wrapped, name=name, description=description)
