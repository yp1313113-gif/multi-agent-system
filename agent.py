# agent.py
"""
供 FastAPI 调用的统一对话入口（异步流式）。

【流式实现说明 —— 真 token 级流式】
本模块用 LangGraph 的 astream_events(version="v2") 驱动真正的 token 级流式输出，
只转发「专职 Worker 生成最终答案」的 token，过滤掉：

  1. Supervisor 的路由决策（langgraph_node == "supervisor"，结构化令牌，不该给用户看）；
  2. 工具调用 delta（content 为空、仅含 tool_call_chunks）。

【为什么要做「缓冲放行」而不是见到 token 就发】
thinking 模型在决定调用工具的那一轮，可能先吐一段"预答文本"（例如"好的，我来查一下…"），
随后才发出 tool_call。如果直接逐 token 推给前端，用户会先看到这段预答，
等工具跑完 answer 再从头吐一遍 —— 观感上就是"重复输出"。

所以每个模型轮次的前 STREAM_GRACE_CHARS 个字符先压在缓冲里：
  · 缓冲期间一旦出现 tool_call_chunks → 判定"这轮是去调工具的"，整段丢弃；
  · 缓冲期间没有出现 tool_call    → 判定"这轮是最终回答"，立刻放行，
    并转入真正的逐 token 流式（后续 token 零延迟直发）。

【兜底】
若整条链路一个 token 都没发出去（极端情况：thinking 模型 content 全程为空），
则回读 checkpointer 里的最终状态，用与 run.py 一致的可靠取答逻辑补发，
保证「永远有回答」。
"""
import asyncio
import uuid

from loguru import logger

from supervisor import create_supervisor, WORKER_NAMES, _make_llm
from config import config
import context
import memory_store
import cost_tracker

# 单个模型轮次在放行前先缓冲的字符数（用于甄别"调工具轮"与"最终回答轮"）
STREAM_GRACE_CHARS = 16

# 流式重置信号。
# 场景：某一轮模型输出（如"好的，我来查一下…"）已经推给用户了，随后才发现它其实是
# 「决定调工具」的前言轮。此时必须通知前端丢弃已显示内容，否则用户会先看到前言、
# 等工具跑完再看到完整答案，观感上就是"重复输出"。
# 实测踩坑：前言可能很长（38 字符），远超缓冲阈值，光靠缓冲阈值拦不住。
STREAM_RESET = "__DSH_STREAM_RESET__"

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


async def warmup_agent():
    """预热编排层：提前构建 Supervisor-Worker 图（服务启动阶段调用）。

    建图要连 checkpointer、编译 StateGraph、加载 4 个 Worker 的模型与工具，
    放在首个请求里会让首字延迟平白多出好几秒。
    """
    return await _get_wrapper()


def _chunk_text(chunk) -> str:
    """从 on_chat_model_stream 的 chunk 里抽出可显示文本（兼容 str / list[dict] 两种 content）。"""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return ""


def _chunk_has_tool_call(chunk) -> bool:
    """该 chunk 是否携带工具调用增量（携带即说明这一轮模型要去调工具，不是最终回答）。"""
    if getattr(chunk, "tool_call_chunks", None):
        return True
    return bool(getattr(chunk, "tool_calls", None))


def _has_tool_calls(message) -> bool:
    """该模型轮次的最终输出是否包含工具调用。

    用于在 on_chat_model_end 处判定「这一轮是去调工具的」——即它的文字内容只是前言
    （如"好的，我来查一下…"），不该作为最终答案展示给用户。
    """
    if message is None:
        return False
    if getattr(message, "tool_calls", None):
        return True
    return bool(getattr(message, "tool_call_chunks", None))


def _node_of(event) -> str:
    """判断这条模型调用是不是「某个 Worker 产出的最终回答」；是则返回 Worker 名，否则返回空串。

    实测：一次真实请求会出现四类模型调用（按 checkpoint_ns 区分）：

      supervisor | supervisor:xxx                   → 路由决策，不该给用户看
      model      | policy_agent:xxx|model:yyy       → Worker 决定调工具（内容只是前言）
      tools      | policy_agent:xxx|tools:yyy       → ⚠️ 工具内部又调了一次 LLM 生成答案
      model      | policy_agent:xxx|model:zzz       → Worker 的最终回答 ✓

    第三类最容易漏：RAG 工具内部自己会调一次模型生成答案，它在 create_agent 的
    "tools" 节点里执行。如果不过滤，用户会先看到一遍"扁平版答案"（工具产出），
    再看到 Worker 的结构化答案 —— 观感上就是整段重复。
    """
    md = event.get("metadata") or {}
    node = md.get("langgraph_node")
    ns = md.get("langgraph_checkpoint_ns") or ""
    segments = [seg.split(":")[0] for seg in ns.split("|") if seg]

    # 工具节点内部的模型调用 → 不是最终回答，一律过滤
    if node == "tools" or "tools" in segments:
        return ""

    if isinstance(node, str) and node in WORKER_NAMES:
        return node
    # create_agent 子图内部会退化成 "model"，用 checkpoint_ns 最外层还原 Worker 名
    if segments and segments[0] in WORKER_NAMES:
        return segments[0]
    return ""


def _extract_final_answer(values) -> str:
    """从最终状态里取「Worker 生成的答案」（与 run.py 一致的可靠取答逻辑）。

    只认带 _worker_reply 标记的那条 AI 消息 —— 多轮对话时从整个历史里
    倒序找 AI 会取到上一轮的旧回答（串台根源）。
    """
    msgs = (values or {}).get("messages", []) if isinstance(values, dict) else []
    if not msgs:
        return ""
    last = msgs[-1]
    answer = ""
    if last is not None and getattr(last, "type", "") == "ai":
        ak = getattr(last, "additional_kwargs", {}) or {}
        if ak.get("_worker_reply"):
            txt = getattr(last, "content", "")
            if isinstance(txt, str) and txt.strip():
                answer = txt
            else:
                # thinking 模型空 content 兜底：取 reasoning_content
                answer = ak.get("reasoning_content") or ak.get("content") or ""
    if not answer:
        # 极端兜底：倒序找最近一条非空 AI 消息
        for m in reversed(msgs):
            if getattr(m, "type", "") == "ai" and str(getattr(m, "content", "")).strip():
                answer = str(m.content)
                break
    return answer


async def stream_chat(message: str, session: str = "user001"):
    """流式返回助手回复文本（真 token 级）。"""
    wrapper = await _get_wrapper()
    supervisor = wrapper.supervisor

    request_id = f"req_{uuid.uuid4().hex[:8]}"
    context.set_request_id(request_id)

    # 长期记忆：读取该用户的事实，供 Worker 节点注入（Supervisor 不注入，避免干扰路由）
    memory_prompt = memory_store.build_memory_prompt(session)
    context.set_memory_prompt(memory_prompt)
    logger.info(
        f"[stream] 开始处理 request_id={request_id} session={session} "
        f"长期记忆={'有' if memory_prompt else '无'}"
    )

    config_dict = {
        "configurable": {
            "thread_id": session,
            "request_id": request_id,
        },
        "recursion_limit": config.RECURSION_LIMIT,
    }

    emitted_chars = 0
    answer_parts: list = []          # 累积完整答案，用于结束后做长期记忆抽取

    # 每个模型轮次独立判定：缓冲中 / 已放行 / 已判定为"调工具轮"（丢弃）
    live = False
    discarded = False
    buffer: list = []
    streamed_runs: set = set()        # 已经推送过内容的模型轮次 run_id

    try:
        async for event in supervisor.astream_events(
            {"messages": [{"role": "user", "content": message}]},
            config=config_dict,
            version="v2",
        ):
            etype = event.get("event")
            run_id = event.get("run_id")

            # 成本埋点：所有 LLM 调用都要记账（含 Supervisor 的路由调用），
            # 因此这一步必须放在「节点过滤」之前 —— 否则路由开销不可见。
            # 从 astream_events 的 metadata 取 langgraph_node，能准确定位钱花在哪个节点。
            if etype == "on_chat_model_end":
                cost_tracker.record_from_event(event, session)

            if etype not in ("on_chat_model_stream", "on_chat_model_end"):
                continue
            if not _node_of(event):
                continue

            if etype == "on_chat_model_stream":
                chunk = (event.get("data") or {}).get("chunk")

                # 出现工具调用增量 → 这一轮是"去调工具"的，已缓冲内容整段丢弃
                if _chunk_has_tool_call(chunk):
                    if not live:
                        discarded = True
                        buffer.clear()
                    continue

                text = _chunk_text(chunk)
                if not text or discarded:
                    continue

                if live:
                    emitted_chars += len(text)
                    answer_parts.append(text)
                    if run_id is not None:
                        streamed_runs.add(run_id)
                    yield text
                else:
                    buffer.append(text)
                    if sum(len(x) for x in buffer) >= STREAM_GRACE_CHARS:
                        # 缓冲够了先放行、转入真流式；若这一轮随后出现工具调用，
                        # 会在 on_chat_model_end 处发重置信号把它收回去
                        head = "".join(buffer)
                        buffer.clear()
                        live = True
                        emitted_chars += len(head)
                        answer_parts.append(head)
                        if run_id is not None:
                            streamed_runs.add(run_id)
                        yield head

            else:  # on_chat_model_end：一轮模型输出结束
                output = (event.get("data") or {}).get("output")

                if _has_tool_calls(output):
                    # 这一轮其实是「决定调工具」的前言轮
                    if run_id is not None and run_id in streamed_runs:
                        logger.info("[stream] 工具调用轮的内容已推送 → 发出前端重置信号")
                        yield STREAM_RESET
                        answer_parts.clear()
                        emitted_chars = 0
                    else:
                        buffer.clear()
                elif not live and buffer:
                    # 该轮没有工具调用、也没凑满缓冲阈值 → 放行（无工具的直接回答）
                    head = "".join(buffer)
                    buffer.clear()
                    live = True
                    emitted_chars += len(head)
                    answer_parts.append(head)
                    if run_id is not None:
                        streamed_runs.add(run_id)
                    yield head

                live, discarded, buffer = False, False, []

    except Exception as e:
        # 流式链路异常不吞掉：记录后走兜底，保证用户一定能拿到回答
        logger.error(f"[stream] astream_events 异常，转兜底取答: {e}")

    # ---- 兜底：一个 token 都没发出去 ----
    if emitted_chars == 0:
        logger.warning("[stream] 流式未产出内容，回读 checkpointer 状态兜底")
        answer = ""
        try:
            snapshot = await supervisor.aget_state(config_dict)
            answer = _extract_final_answer(getattr(snapshot, "values", None))
        except Exception as e:
            logger.error(f"[stream] 兜底读取状态失败: {e}")

        if not answer:
            # 图正常结束但没有 Worker 产出：通常是寒暄/范围外输入被 Supervisor 判为 __end__。
            # 这是设计行为，不该回"抱歉没有生成有效回答"——那像是系统故障。
            answer = (
                "你好，我是研发费用智能管理助手。你可以问我：\n"
                "· 研发费用加计扣除政策（比例、口径、申报流程）\n"
                "· 研发费用归集（费用归类到 8 类口径）\n"
                "· 研发费用风险扫描（金四对标预警）\n"
                "· 研发工时填报"
            )

        # 分块补发，前端体验与流式一致
        answer_parts.append(answer)
        for i in range(0, len(answer), 4):
            yield answer[i:i + 4]
            await asyncio.sleep(0.02)

    # ---- 会话收尾：抽取长期记忆（失败不影响主流程）----
    try:
        await memory_store.extract_and_remember(
            _make_llm(), session, message, "".join(answer_parts)
        )
    except Exception as e:
        logger.warning(f"[stream] 长期记忆抽取跳过: {e}")

    logger.info(f"[stream] 完成 request_id={request_id} 输出 {emitted_chars} 字符")
