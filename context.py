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

# 当前请求的「发起人」（会话标识）
# ★ 用途：HITL 审批记录要绑定发起人 —— 只复用同一个人的裁定。
#   原实现没绑身份，导致「张三批准过的敏感问题，李四问同一句会被自动放行」。
#   真实多用户部署时这里应该是登录用户 ID；本项目是单机应用，用会话标识代替。
_current_requester: ContextVar = ContextVar("current_requester", default="anonymous")

# 当前 Worker 轮次的「业务黑板草稿」
# ★ 为什么是一个可变 dict，而不是让工具调 set() 换一个新对象：
#   工具被 harness.guarded_tool 包过，可能跑在线程池里；
#   copy_context() 只复制「变量 → 值」的绑定表（浅拷贝），
#   所以线程内对 dict 的【原地修改】父上下文看得见，
#   但线程内调 set() 换成新对象【不会回传】。
#   因此约定：节点开一块空 dict，工具只往里塞东西，节点再整块收口。
_board_draft: ContextVar = ContextVar("board_draft", default=None)


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


def set_requester(requester: str) -> None:
    """设置当前请求的发起人（会话标识）。"""
    _current_requester.set(requester or "anonymous")


def get_requester() -> str:
    """获取当前请求的发起人。"""
    return _current_requester.get()


# ============ 业务黑板（Worker 轮次级草稿） ============

def new_board():
    """开一块新黑板草稿，返回 token（供 reset_board 还原）。"""
    return _board_draft.set({})


def reset_board(token) -> None:
    """还原到上一层的黑板草稿（多层嵌套时不会串）。"""
    try:
        _board_draft.reset(token)
    except (ValueError, LookupError):
        _board_draft.set(None)


def board_append(key: str, item) -> None:
    """往黑板的某个列表字段追加一条（没有黑板时静默跳过）。"""
    b = _board_draft.get()
    if isinstance(b, dict):
        b.setdefault(key, []).append(item)


def board_set(key: str, value) -> None:
    """覆盖黑板的某个字段（适合「最新状态」类数据，如分摊结果 / 风险指标）。"""
    b = _board_draft.get()
    if isinstance(b, dict):
        b[key] = value


def board_draft() -> dict:
    """取出当前黑板草稿（无则返回空 dict）。"""
    b = _board_draft.get()
    return b if isinstance(b, dict) else {}


# 兼容旧写法：模块级属性访问仍可读到当前值
def __getattr__(name):
    if name == "current_request_id":
        return _current_request_id.get()
    raise AttributeError(name)
