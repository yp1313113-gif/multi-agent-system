"""Agent Harness：Agent Loop 的执行环境与约束层。

Agent Loop 架构（思考→行动→观察→循环）中，Harness 负责"执行环境"：
  - 工具执行统一包装：超时、有限重试、日志、耗时统计
  - 约束检查：死循环防护（配合 recursion_limit 的最大轮数 + 单工具调用上限）
  - 可观测：每次调用记录 ts / 耗时 / 成败，供 Langfuse / 日志回放

业务 Agent（Supervisor / Worker）只负责"决策与工具选择"，
Harness 保证"工具执行可靠、有界、可观测"——这是 Agent 上生产的横切面。
"""
import time
import concurrent.futures
import contextvars
from functools import wraps

from loguru import logger


def with_timeout(seconds: float = 10):
    """给同步函数加超时（线程池实现，超时抛 TimeoutError）。"""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], dict):
                kwargs = {**args[0], **kwargs}
                args = ()
            # 关键：线程池不会自动继承 contextvars 上下文，
            # 必须显式 copy_context() 传进去，否则工具线程里读到的 request_id 是默认值，
            # 日志无法与请求关联（排查问题时等于丢了链路）。
            ctx = contextvars.copy_context()

            # ⚠️ 这里绝不能写成 with ThreadPoolExecutor(...) as ex:
            #    with 块退出时会调用 shutdown(wait=True)，即「等到任务真正跑完」，
            #    于是 fut.result(timeout=N) 抛出的超时异常也要等任务结束才抛得出来 ——
            #    超时形同虚设（实测：6 秒任务设 2 秒超时，异常在 6.01 秒才出现）。
            # 正确做法：手动 shutdown(wait=False)，让调用方在 N 秒时立刻拿到 TimeoutError。
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                fut = ex.submit(ctx.run, fn, *args, **kwargs)
                return fut.result(timeout=seconds)
            finally:
                ex.shutdown(wait=False)

        return wrapper

    return decorator


def with_retry(max_retries: int = 2, delay: float = 0.5):
    """有限重试（指数退避），处理外部 API 偶发抖动。"""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], dict):
                kwargs = {**args[0], **kwargs}
                args = ()
            last = None
            for i in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    if i < max_retries - 1:
                        time.sleep(delay * (2 ** i))
            raise last

        return wrapper

    return decorator


class AgentLoopGuard:
    """Agent Loop 硬约束：防死循环 / 防工具滥用（和 LangGraph recursion_limit 双保险）。"""

    def __init__(self, max_turns: int = 10, max_tool_calls: int = 3):
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.turn = 0
        self.tool_calls: dict = {}

    def check(self) -> bool:
        """每轮循环前调用；超过最大轮数返回 False（强制终止）。"""
        self.turn += 1
        if self.turn > self.max_turns:
            logger.warning("🔴 Agent Loop 超过最大轮数，强制终止（防死循环）")
            return False
        return True

    def record_tool(self, name: str) -> bool:
        """记录一次工具调用；同一工具超限返回 False（拦截）。"""
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        if self.tool_calls[name] > self.max_tool_calls:
            logger.warning(f"🔴 工具 {name} 调用超限（>{self.max_tool_calls} 次），拦截")
            return False
        return True


def guarded_tool(tool, timeout: float = None, max_retries: int = None):
    """把 LangChain 工具包装为 Harness 受控工具：超时 + 重试 + 日志 + 耗时。

    超时/重试默认读 config.TOOL_TIMEOUT / TOOL_MAX_RETRIES（未配置则 10s / 2 次）。
    ⚠️ 超时必须「真的超时」—— 详见 with_timeout 里关于 ThreadPoolExecutor 的注释。

    保持 LangChain Tool 接口（name/description），可无缝接入 create_agent。
    """
    # 超时/重试从配置读取，便于按环境调整：
    # RAG 工具内部包含一次 LLM 生成，端到端常需十几秒，10 秒上限会误杀正常调用。
    try:
        from config import config
        if timeout is None:
            timeout = float(config.TOOL_TIMEOUT)
        if max_retries is None:
            max_retries = int(config.TOOL_MAX_RETRIES)
    except Exception:
        timeout = 10 if timeout is None else timeout
        max_retries = 2 if max_retries is None else max_retries

    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        from langchain.tools import StructuredTool

    # 兼容两种工具形态：LangChain BaseTool（有 func/name）或普通函数（测试 stub）
    func = getattr(tool, "func", None) or tool
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
    description = getattr(tool, "description", None) or "harness guarded tool"

    def wrapped(*args, **kwargs):
        # 兼容 langchain 调用方式：tool_input 可能是 dict（位置或关键字）
        if args and isinstance(args[0], dict):
            kwargs = {**args[0], **kwargs}
            args = ()
        t0 = time.perf_counter()
        try:
            result = with_retry(max_retries)(with_timeout(timeout)(func))(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[harness] ✅ {name} 调用成功，耗时 {ms:.0f}ms")
            return result
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            logger.error(f"[harness] ❌ {name} 调用失败（{ms:.0f}ms）: {e}")
            raise

    # 优先 tool.copy 保留原 args_schema（langchain 解析 dict → func 参数正确）
    try:
        if hasattr(tool, "copy"):
            return tool.model_copy(update={"func": wrapped})
    except Exception:
        pass
    # 普通函数（测试 stub 等）兜底：重建 StructuredTool
    return StructuredTool.from_function(func=wrapped, name=name, description=description)
