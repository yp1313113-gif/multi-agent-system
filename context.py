# context.py
"""请求级上下文（request_id / 长期记忆提示块）。

━━━ 为什么用 contextvars 而不是模块级全局变量 ━━━
原实现是 `current_request_id: str = "N/A"` 这样的模块级全局变量。
在**高并发**下这是错的：本项目允许 10 个请求并发（concurrency.AsyncLimiter），
A 请求在 `await` 处让出控制权时，B 请求会把全局变量覆盖成自己的 id，
于是 A 后续所有日志都打上了 B 的 request_id —— 排查线上问题时会被彻底带偏。

contextvars.ContextVar 是 asyncio 官方的请求级隔离方案：
每个 async 任务持有自己的上下文副本，互不干扰。
（注意：线程池不会自动继承上下文，harness.with_timeout 里已用
 contextvars.copy_context() 显式传递，保证工具在线程里也能读到正确的 id。）
"""
from contextvars import ContextVar

# 当前请求 ID
_current_request_id: ContextVar = ContextVar("current_request_id", default="N/A")

# 当前请求的长期记忆提示块（由 agent.stream_chat 在请求开始时注入）
_memory_prompt: ContextVar = ContextVar("memory_prompt", default="")


def set_request_id(request_id: str):
    """设置当前请求ID（仅对当前 async 任务/线程可见）。"""
    _current_request_id.set(request_id)


def get_request_id() -> str:
    """获取当前请求ID。"""
    return _current_request_id.get()


def set_memory_prompt(text: str):
    """设置当前请求的长期记忆提示块。"""
    _memory_prompt.set(text or "")


def get_memory_prompt() -> str:
    """获取当前请求的长期记忆提示块（无则为空串）。"""
    return _memory_prompt.get()


# 兼容旧写法：模块级属性访问仍可读到当前值
def __getattr__(name):
    if name == "current_request_id":
        return _current_request_id.get()
    raise AttributeError(name)
