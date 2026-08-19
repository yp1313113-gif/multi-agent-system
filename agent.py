# agent.py
"""
供 FastAPI 调用的统一对话入口（异步流式）。

复用 supervisor.SupervisorWrapper 构建的 Supervisor-Worker 多 Agent 系统，
按 session 区分对话线程（thread_id），由 AsyncSqliteSaver 持久化多轮记忆。

流式实现：使用 astream_events(version="v2") 监听 on_chat_model_stream 事件，
只转发「专职 Worker 生成答案」时的 token，过滤掉：
  - Supervisor 的路由决策（结构化输出，langgraph_node == "supervisor"）；
  - 工具调用 delta（content 为空、仅含 tool_calls）。
从而实现与原单 Agent 版一致的「逐字」流式输出效果。
"""
import asyncio
import uuid

from supervisor import create_supervisor
from config import config
import context

# 全局复用的 SupervisorWrapper（首次请求时惰性初始化）
_wrapper = None
_init_lock = asyncio.Lock()


async def _get_wrapper():
    global _wrapper
    if _wrapper is None:
        async with _init_lock:
            if _wrapper is None:
                _wrapper = await create_supervisor()
    return _wrapper


async def stream_chat(message: str, session: str = "user001"):
    """流式返回助手回复文本（token 级逐字输出）。"""
    wrapper = await _get_wrapper()
    supervisor = wrapper.supervisor

    request_id = f"req_{uuid.uuid4().hex[:8]}"
    context.set_request_id(request_id)

    config_dict = {
        "configurable": {
            "thread_id": session,
            "request_id": request_id,
        },
        "recursion_limit": config.RECURSION_LIMIT,
    }

    async for event in supervisor.astream_events(
        {"messages": [{"role": "user", "content": message}]},
        config=config_dict,
        version="v2",
    ):
        if event["event"] != "on_chat_model_stream":
            continue
        # 排除 Supervisor 路由决策的 token（结构化输出，不进入最终答案）
        if event["metadata"].get("langgraph_node") == "supervisor":
            continue
        # 只取专职 Worker 生成答案的文本内容（跳过工具调用 delta）
        chunk = event["data"]["chunk"]
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield content
