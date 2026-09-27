# middleware.py
"""中间件链：把 Agent 的横切关注点从「散落在各处」收敛成一条**有序链**。

━━━ 为什么需要（面试口径）━━━
Agent 的横切关注点天然会散开：记忆注入写在编排层、超时重试写在装饰器、
用量统计挂在事件回调、审批写在工具内部…… 结果是：

  · 想调整顺序（比如「先压缩再注入记忆」）要改好几处，容易漏；
  · 加一种新能力时，不知道应该插在哪一层；
  · 想在中间做一次「预检」时发现没有位置可以插。

本模块把这件事显式化：**中间件是一条有序链，每个中间件只干一件事**，
分别在「模型调用边界」和「工具调用边界」上运行。

━━━ 两种边界 ━━━
· **模型边界**（`ModelRequest`）：在把消息发给模型之前。
  当前顺序 —— 压缩(10) → 记忆注入(20) → 黑板注入(30)
· **工具边界**（`ToolCall`）：在工具执行的前后。
  当前顺序 —— 审批(0) → 循环守卫(5) → 重试(10) → 超时(20) → 计时(30)
              → 结果卸载(40) → 指标(50)

━━━ 洋葱模型 ━━━
`call(call, next_fn)` 的调用顺序是「外层 → 内层」，返回时反向。
所以 `RetryMiddleware(10)` 在 `TimeoutMiddleware(20)` 外层，
意味着「**每次重试都有独立的超时**」—— 这正是我们要的语义。

━━━ 与其他模块的分工 ━━━
· 「谁拦、拦的依据」仍属于业务：审批规则由 `hitl.needs_review` 决定，
  中间件只负责**在正确的时机强制执行**（注册表模式，见 `ApprovalMiddleware`）。
· token 记账不在这一层 —— 它挂在 `astream_events` 上（见 `cost_tracker`），
  因为那里能拿到 `langgraph_node` 元信息，能分清钱是路由花的还是 Worker 花的。
  本层的 `MetricsMiddleware` 统计的是**工具调用级**指标（次数/失败/耗时）。
"""
from __future__ import annotations

import concurrent.futures
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


# ============================================================
# 底层原语（被中间件和 harness 的兼容出口共用，保证只有一份实现）
# ============================================================

def run_with_timeout(fn: Callable[[], Any], seconds: float) -> Any:
    """在独立线程里执行 fn，超时抛 TimeoutError。

    ⚠️ **绝不能写成 `with ThreadPoolExecutor(...) as ex:`**
    `with` 块退出时会调 `shutdown(wait=True)`，即「等任务真正跑完」，
    于是 `fut.result(timeout=N)` 抛出的 TimeoutError 也要等任务结束才传得出来
    —— 超时形同虚设（实测：6 秒任务设 2 秒超时，异常在 6.01 秒才出现）。

    另一个坑：线程池**不自动继承 contextvars**，必须显式 `copy_context()`，
    否则工具线程里读到的 request_id / 发起人都是默认值，日志无法与请求关联。

    注意语义：超时的含义是「**调用方不等了**」，不是「任务停下来了」——
    Python 无法强杀线程。所以有副作用的工具还要配合幂等。
    """
    ctx = contextvars.copy_context()
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(ctx.run, fn)
        return fut.result(timeout=seconds)
    finally:
        ex.shutdown(wait=False)


def run_with_retry(fn: Callable[[], Any], max_retries: int, delay: float) -> Any:
    """指数退避重试（`max_retries` 是**总尝试次数**，不是重试次数）。"""
    last: Exception | None = None
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:      # noqa: BLE001 —— 重试要覆盖任意异常
            last = e
            if i < max_retries - 1:
                time.sleep(delay * (2 ** i))
    assert last is not None
    raise last


class AgentLoopGuard:
    """Agent Loop 硬约束：防死循环 / 防工具滥用（与 LangGraph recursion_limit 双保险）。

    `max_turns` 由调用方在每一轮循环前 `check()`；
    `record_tool()` 按工具名计数，同一工具超限即拦截。

    ⚠️ 历史上这个类只在测试里被实例化过，生产路径没有接上 ——
    现在由 `LoopGuardMiddleware` 真正挂进工具调用链。
    """

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


# 当前请求的循环守卫（按请求隔离，不能用模块级单例 —— 并发下会互相污染）
_loop_guard: contextvars.ContextVar = contextvars.ContextVar("loop_guard", default=None)


def new_loop_guard(max_turns: int = 10, max_tool_calls: int = 3):
    """为当前请求开一个循环守卫，返回 token（供 reset 还原）。"""
    return _loop_guard.set(AgentLoopGuard(max_turns, max_tool_calls))


def reset_loop_guard(token) -> None:
    try:
        _loop_guard.reset(token)
    except (ValueError, LookupError):
        _loop_guard.set(None)


def current_loop_guard():
    return _loop_guard.get()


# ============================================================
# 上下文对象
# ============================================================

@dataclass
class ModelRequest:
    """一次模型调用的请求上下文（中间件在此读写）。

    注意 `messages` 会被中间件**替换**，而不是就地修改 ——
    原始消息列表属于调用方，中间件不得污染它。
    """

    node: str = ""
    scope: str = "worker"          # worker | supervisor
    messages: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def note(self, text: str) -> None:
        self.notes.append(text)


@dataclass
class ToolCall:
    """一次工具调用的上下文。"""

    name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    timeout: float | None = None        # 单工具覆盖（None → 用 config）
    max_retries: int | None = None      # 单工具覆盖（None → 用 config）
    result: Any = None
    notes: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ============================================================
# 中间件基类与链
# ============================================================

class Middleware:
    """中间件基类。

    子类通常只需覆写 `before_model`（模型边界）或
    `before_tool` / `after_tool`（工具边界）。
    需要完全接管调用的（重试、超时）覆写 `call`。
    """

    name: str = "middleware"
    order: int = 100
    scopes: tuple = ("*",)          # 模型边界生效的 scope；"*" 表示全部
    tools: tuple = ("*",)           # 工具边界生效的工具名；"*" 表示全部

    # ---- 模型边界 ----
    def before_model(self, req: ModelRequest) -> None:  # noqa: B027
        """在消息发给模型之前。就地替换 req.messages 或写入 req.stats。"""

    # ---- 工具边界 ----
    def call(self, call: ToolCall, next_fn: Callable[[], Any]) -> Any:
        """默认实现：before → next → after（洋葱模型的一层）。"""
        blocked = self.before_tool(call)
        if blocked is not None:
            return blocked
        result = next_fn()
        replaced = self.after_tool(call, result)
        return result if replaced is None else replaced

    def before_tool(self, call: ToolCall):
        """返回非 None 表示**拦截**，该值将直接作为工具结果返回。"""
        return None

    def after_tool(self, call: ToolCall, result):
        """返回非 None 表示**替换**结果。"""
        return None

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} order={self.order}>"


class MiddlewareChain:
    """有序中间件链。

    模型边界：按 order **升序**执行 `before_model`。
    工具边界：按 order 构造洋葱 —— order 小的在外层（先执行、后返回）。
    """

    def __init__(self):
        self._models: list[Middleware] = []
        self._tools: list[Middleware] = []

    def add_model(self, mw: Middleware) -> "MiddlewareChain":
        self._models.append(mw)
        self._models.sort(key=lambda m: m.order)
        return self

    def add_tool(self, mw: Middleware) -> "MiddlewareChain":
        self._tools.append(mw)
        self._tools.sort(key=lambda m: m.order)
        return self

    # ---- 模型边界 ----
    def build(self, messages, node: str = "", scope: str = "worker", **meta) -> ModelRequest:
        """跑一遍模型中间件，返回处理后的请求上下文。"""
        req = ModelRequest(node=node, scope=scope, messages=list(messages or []), meta=meta)
        for mw in self._models:
            if "*" not in mw.scopes and scope not in mw.scopes:
                continue
            try:
                mw.before_model(req)
            except Exception as e:
                # 中间件失败不该带崩主流程，但**必须告警**（不静默降级）
                logger.warning(f"[middleware] {mw.name}.before_model 失败（已跳过）: {e}")
                req.note(f"{mw.name}:error({type(e).__name__})")
        return req

    # ---- 工具边界 ----
    def _applicable_tools(self, name: str) -> list[Middleware]:
        return [m for m in self._tools if "*" in m.tools or name in m.tools]

    def run_tool(self, fn: Callable, name: str, *args, **kwargs) -> Any:
        """按链执行工具调用。

        `__timeout__` / `__max_retries__` 是**单工具覆盖**，不进业务参数。
        复制一份 kwargs 再取，避免依赖「ToolCall 持有同一个 dict 引用」这种隐式行为。
        """
        kwargs = dict(kwargs)
        timeout = kwargs.pop("__timeout__", None)
        max_retries = kwargs.pop("__max_retries__", None)
        call = ToolCall(name=name, args=args, kwargs=kwargs, timeout=timeout, max_retries=max_retries)
        chain = self._applicable_tools(name)

        def _run(i: int) -> Any:
            if i >= len(chain):
                return fn(*args, **kwargs)
            return chain[i].call(call, lambda: _run(i + 1))

        try:
            result = _run(0)
            call.result = result
            return result
        except Exception:
            raise

    def describe(self) -> dict:
        """链的可读描述（写日志 / 接口暴露）。"""
        return {
            "model": [{"name": m.name, "order": m.order, "scopes": list(m.scopes)} for m in self._models],
            "tool": [{"name": m.name, "order": m.order, "tools": list(m.tools)} for m in self._tools],
        }


# ============================================================
# 模型边界中间件
# ============================================================

class CompressionMiddleware(Middleware):
    """上下文压缩：窗口截断 + 确定性压缩（不调模型、不花钱）。

    只在 `supervisor` scope 生效 —— Worker 只接收当前轮消息，本来就没有历史可压。

    为什么不在这里做摘要：路由每轮都跑，多一次 LLM 调用不划算。
    详见 `compression.route_view` 的说明。
    """

    name = "compression"
    order = 10
    scopes = ("supervisor",)

    def before_model(self, req: ModelRequest) -> None:
        import compression
        out, report = compression.route_view(req.messages)
        # 替换而不是就地修改 —— 原始 messages 属于调用方
        req.messages = list(out)
        req.stats["compression"] = report.to_dict()
        if report.level != "none":
            req.note(f"compression:{report.level}")
            logger.info(f"[middleware] 压缩｜{report.reason}｜{report.summary_line()}")


class MemoryMiddleware(Middleware):
    """长期记忆注入：把「关于该用户的事实」作为前缀拼进最后一条用户消息。

    只在 `worker` scope 生效 —— **路由不能注入记忆**：
    记忆里的词（「偏好」「项目」）会干扰路由令牌匹配，把路由带偏。
    """

    name = "memory"
    order = 20
    scopes = ("worker",)

    def before_model(self, req: ModelRequest) -> None:
        import context
        block = context.get_memory_prompt()
        if not block or not req.messages:
            return
        req.messages = _prefix_last_message(req.messages, block)
        req.note(f"memory:{len(block)}字")
        logger.info(f"[middleware] 已注入长期记忆（{len(block)} 字符）")


class BoardMiddleware(Middleware):
    """业务黑板注入：把「本会话已产生的业务数据」拼进最后一条用户消息。

    formatter 由编排层注入（`_format_board`）—— 中间件不持有业务格式，
    只负责「在正确的时机调用它」。
    """

    name = "board"
    order = 30
    scopes = ("worker",)

    def __init__(self, formatter: Callable[[Any], str] | None = None):
        self.formatter = formatter

    def before_model(self, req: ModelRequest) -> None:
        state = req.meta.get("state")
        if state is None or self.formatter is None or not req.messages:
            return
        text = self.formatter(state)
        if not text:
            return
        req.messages = _suffix_last_message(req.messages, text)
        req.note(f"board:{len(text)}字")
        logger.info(f"[middleware] 已注入业务数据摘要（{len(text)} 字符）")


def _prefix_last_message(messages: list, prefix: str) -> list:
    return _rewrite_last(messages, lambda c: f"{prefix}\n\n{c}")


def _suffix_last_message(messages: list, suffix: str) -> list:
    return _rewrite_last(messages, lambda c: f"{c}\n\n{suffix}")


def _rewrite_last(messages: list, fn: Callable[[str], str]) -> list:
    """对列表里最后一条 human 消息做内容改写，返回**新列表**。"""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        t = getattr(m, "type", m.get("type") if isinstance(m, dict) else None)
        if t != "human":
            continue
        content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        if not isinstance(content, str):
            return out
        try:
            from langchain_core.messages import HumanMessage
            out[i] = HumanMessage(content=fn(content))
        except Exception:
            out[i] = {"type": "human", "content": fn(content)}
        return out
    return out


# ============================================================
# 工具边界中间件
# ============================================================

@dataclass
class ApprovalRequest:
    """一次审批判定的结果。由**工具自己**的守卫返回。

    `key` / `record_name` 交给守卫指定，是为了兼容既有的审批记录格式 ——
    中间件不应该替业务决定「用什么当主键」。
    """

    need: bool = False
    kind: str = ""
    reason: str = ""
    key: str = ""              # 匹配「已批准」记录的问题键；空则用「工具名 + 参数摘要」
    record_name: str = ""      # 审批记录里的 tool_name；空则用工具名


class ApprovalMiddleware(Middleware):
    """合规复核拦截 —— **注册表模式**。

    业务规则不写死在中间件里。工具通过 `register_guard()` 声明
    「什么条件下需要人工复核」，中间件只负责在正确的时机强制执行完整的
    「已批准 → 消费 → 放行 / 未批准 → 挂起并短路」流程。

    这样做的好处：**强制执行不依赖开发者记得在每个工具里写一遍**。
    新增一个写类工具时，只要注册守卫就自动获得整套流程。

    ⚠️ 边界要清楚：只适合**纯横切**的审批（判定只依赖入参）。
    如果审批判定依赖工具内部算出来的中间结果（比如「规则库没命中才需要复核」），
    那它是**业务逻辑**，应该留在工具里 —— `tools/expense_tool.py` 就是这种情况，
    刻意没有迁移过来。强行搬进中间件只会把业务规则泄露到横切层。
    """

    name = "approval"
    order = 0

    def __init__(self):
        self._guards: dict = {}

    def register_guard(self, tool_name: str,
                       guard: Callable[[tuple, dict], ApprovalRequest]) -> None:
        """注册守卫：`guard(args, kwargs) -> ApprovalRequest`"""
        self._guards[tool_name] = guard

    def before_tool(self, call: ToolCall):
        guard = self._guards.get(call.name)
        if guard is None:
            return None
        try:
            req = guard(call.args, call.kwargs)
        except Exception as e:
            # 守卫写坏了不能把工具拦死，但必须告警（不静默）
            logger.warning(f"[middleware] approval 守卫执行失败（按放行处理）: {e}")
            return None
        if req is None or not req.need:
            return None

        import hitl
        record_name = req.record_name or call.name
        key = req.key or f"{call.name}:{str(call.args)[:120]}{str(call.kwargs)[:120]}"
        call.stats["approval_kind"] = req.kind

        if hitl.has_approved_query(key, record_name):
            hitl.consume_approved_query(key, record_name)
            logger.info(f"[middleware] ✅ {record_name} 已有复核记录（kind={req.kind}），放行")
            return None

        import uuid
        approval_id = f"approval_{uuid.uuid4().hex[:8]}"
        hitl.request_approval(
            tool_name=record_name, tool_input=key, user_message=key,
            context={"source": "middleware", "kind": req.kind},
            approval_id=approval_id, kind=req.kind, reason=req.reason,
        )
        logger.info(f"[middleware] ⏳ {record_name} 挂起待复核 {approval_id}（{req.reason}）")
        return (f"⏳ 该操作涉及{req.reason}，需要人工复核。\n"
                f"复核 ID：{approval_id}\n"
                f"请运行 'python hitl.py' 给出裁定后重新提问。")


class LoopGuardMiddleware(Middleware):
    """循环守卫：限制同一工具在一次请求内的调用次数，防死循环。

    守卫按**请求**隔离（contextvars），没有设置守卫时是 no-op ——
    这样单元测试和 CLI 场景不受影响。
    """

    name = "loop_guard"
    order = 5

    def before_tool(self, call: ToolCall):
        guard = current_loop_guard()
        if guard is None:
            return None
        if not guard.record_tool(call.name):
            return (f"🔴 工具 {call.name} 在本次请求中调用次数超限"
                    f"（>{guard.max_tool_calls} 次），已拦截以防死循环。")
        return None


class RetryMiddleware(Middleware):
    """有限重试（指数退避）。在外层 —— **每次重试都有独立的超时**。"""

    name = "retry"
    order = 10

    def call(self, call: ToolCall, next_fn: Callable[[], Any]) -> Any:
        from config import config
        retries = call.max_retries if call.max_retries is not None else int(config.TOOL_MAX_RETRIES)
        delay = float(getattr(config, "TOOL_RETRY_DELAY", 1))
        if retries <= 1:
            return next_fn()
        return run_with_retry(next_fn, retries, delay)


class TimeoutMiddleware(Middleware):
    """超时（独立线程执行）。在内层 —— 每次重试各自计时。"""

    name = "timeout"
    order = 20

    def call(self, call: ToolCall, next_fn: Callable[[], Any]) -> Any:
        from config import config
        seconds = call.timeout if call.timeout is not None else float(config.TOOL_TIMEOUT)
        return run_with_timeout(next_fn, seconds)


class TimingMiddleware(Middleware):
    """耗时日志（原 harness 的行为）。"""

    name = "timing"
    order = 30

    def call(self, call: ToolCall, next_fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            result = next_fn()
            ms = (time.perf_counter() - t0) * 1000
            call.stats["duration_ms"] = round(ms, 1)
            logger.info(f"[middleware] ✅ {call.name} 调用成功，耗时 {ms:.0f}ms")
            return result
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            call.stats["duration_ms"] = round(ms, 1)
            logger.error(f"[middleware] ❌ {call.name} 调用失败（{ms:.0f}ms）: {e}")
            raise


class ResultCompressMiddleware(Middleware):
    """工具结果卸载：超长结果落盘，只把「路径 + 哈希 + 预览」交给模型。

    这是上下文压缩的**入口侧**——在结果进入消息流之前就把它收掉，
    比等到发给模型时再压更省（少一次 JSON 序列化和一次 token 估算）。
    结果不是字符串（如 LangChain ToolMessage）时跳过。
    """

    name = "result_compress"
    order = 40

    def after_tool(self, call: ToolCall, result):
        if not isinstance(result, str) or len(result) < 2000:
            return None
        try:
            import compression
            compacted, path = compression.compact_tool_result(result, call.name)
            if path:
                call.stats["offloaded"] = path
                logger.info(f"[middleware] {call.name} 结果已卸载: {path}")
            return compacted if compacted != result else None
        except Exception as e:
            logger.warning(f"[middleware] 结果卸载失败（保留原文）: {e}")
            return None


class MetricsMiddleware(Middleware):
    """工具级指标：调用次数 / 成功 / 失败 / 累计耗时。

    注意与 `cost_tracker` 的分工：token 记账在事件层（能拿到节点信息），
    本层统计的是**工具调用**的可靠性指标。
    """

    name = "metrics"
    order = 50

    def __init__(self):
        self.counters: dict = {}

    def call(self, call: ToolCall, next_fn: Callable[[], Any]) -> Any:
        slot = self.counters.setdefault(call.name, {"calls": 0, "ok": 0, "fail": 0, "total_ms": 0.0, "offloaded": 0})
        slot["calls"] += 1
        t0 = time.perf_counter()
        try:
            result = next_fn()
            slot["ok"] += 1
            return result
        except Exception:
            slot["fail"] += 1
            raise
        finally:
            slot["total_ms"] += (time.perf_counter() - t0) * 1000
            if call.stats.get("offloaded"):
                slot["offloaded"] += 1

    def snapshot(self) -> dict:
        out = {}
        for name, s in self.counters.items():
            calls = s["calls"] or 1
            out[name] = {
                "calls": s["calls"], "ok": s["ok"], "fail": s["fail"],
                "success_rate": round(s["ok"] / calls, 3),
                "avg_ms": round(s["total_ms"] / calls, 1),
                "offloaded": s["offloaded"],
            }
        return out


# ============================================================
# 默认链
# ============================================================

APPROVAL = ApprovalMiddleware()
METRICS = MetricsMiddleware()


def build_model_chain(board_formatter: Callable | None = None) -> MiddlewareChain:
    """构造模型边界链。`board_formatter` 由编排层注入（业务格式不放在中间件层）。"""
    chain = MiddlewareChain()
    chain.add_model(CompressionMiddleware())
    chain.add_model(MemoryMiddleware())
    chain.add_model(BoardMiddleware(board_formatter))
    return chain


def build_tool_chain() -> MiddlewareChain:
    """构造工具边界链。顺序：审批 → 循环守卫 → 重试 → 超时 → 计时 → 卸载 → 指标。"""
    chain = MiddlewareChain()
    chain.add_tool(APPROVAL)
    chain.add_tool(LoopGuardMiddleware())
    chain.add_tool(RetryMiddleware())
    chain.add_tool(TimeoutMiddleware())
    chain.add_tool(TimingMiddleware())
    chain.add_tool(ResultCompressMiddleware())
    chain.add_tool(METRICS)
    return chain


TOOL_CHAIN = build_tool_chain()


def describe() -> dict:
    """整条链的可读描述（供 /middleware 接口与日志）。"""
    return {"tool_chain": TOOL_CHAIN.describe()["tool"], "tool_metrics": METRICS.snapshot()}
