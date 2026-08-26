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
    """流式返回助手回复文本。

    实现：用 ainvoke 获取「最终回答」（与 run.py 一致的可靠取答逻辑），
    再按小块流式吐出——保证只输出最终答案一次，规避 thinking 模型
    工具调用前后重复输出预答文本的问题（演示/生产均适用）。
    """
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

    final_state = await supervisor.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config_dict,
    )
    msgs = final_state.get("messages", [])
    last = msgs[-1] if msgs else None
    answer = ""
    if last is not None and getattr(last, "type", "") == "ai":
        ak = getattr(last, "additional_kwargs", {}) or {}
        if ak.get("_worker_reply"):
            txt = getattr(last, "content", "")
            if isinstance(txt, str) and txt.strip():
                answer = txt
            else:
                answer = ak.get("reasoning_content") or ak.get("content") or ""
    if not answer:
        answer = "（抱歉，没有生成有效回答）"

    # 按小块模拟流式输出（每 4 字符 + 20ms），效果等同逐字流式
    for i in range(0, len(answer), 4):
        yield answer[i:i + 4]
        await asyncio.sleep(0.02)

