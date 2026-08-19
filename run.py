# run.py
import asyncio
import sys
import uuid
from dotenv import load_dotenv
from loguru import logger
from supervisor import create_supervisor
from config import config
import context
import openai

# Langfuse 可选导入：未安装或未配置时自动降级，不影响主流程运行
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    _HAVE_LANGFUSE = True
except Exception:  # pragma: no cover
    Langfuse = None
    CallbackHandler = None
    _HAVE_LANGFUSE = False

# ==================== 日志配置 ====================
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> - <level>{message}</level>",
    level=config.LOG_LEVEL
)
logger.add(
    "logs/agent.log",
    rotation="10 MB",
    format="{time} | {level} | {message}",
    level="INFO"
)

load_dotenv()

# ==================== 初始化 Langfuse（可选） ====================
langfuse = None
langfuse_handler = None
if _HAVE_LANGFUSE and config.LANGFUSE_PUBLIC_KEY:
    try:
        langfuse = Langfuse()
        langfuse_handler = CallbackHandler()
        logger.info(f"✅ Langfuse handler 已创建: {langfuse_handler}")
        logger.info(f"✅ Langfuse 配置: {config.LANGFUSE_PUBLIC_KEY[:20]}...")
    except Exception as e:  # pragma: no cover
        logger.warning(f"⚠️ Langfuse 初始化失败，已降级为无追踪模式: {e}")
        langfuse = None
        langfuse_handler = None
else:
    logger.info("ℹ️ 未配置 Langfuse，跳过链路追踪（不影响主流程）")


# ==================== 模型输出文本提取 ====================
def _extract_model_text(outputs) -> str:
    """从 on_chat_model_end 的 output 中提取最终回答文本。
    兼容多种结构：ChatResult / AIMessage / AIMessageChunk / dict 等。
    取不到时返回空字符串。"""
    if outputs is None:
        return ""
    # 列表（如 generations 列表）
    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            txt = _extract_model_text(item)
            if txt:
                return txt
        return ""
    # ChatResult：generations[0].message
    generations = getattr(outputs, "generations", None)
    if generations:
        first = generations[0]
        msg = getattr(first, "message", None) or first
        return _extract_model_text(msg)
    # dict：message / generations / content
    if isinstance(outputs, dict):
        if outputs.get("message"):
            return _extract_model_text(outputs["message"])
        if outputs.get("generations"):
            return _extract_model_text(outputs["generations"])
        return str(outputs.get("content") or "")
    # 有 .content 的对象（AIMessage / AIMessageChunk）
    content = getattr(outputs, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    # 有 .text 的对象（某些 generation chunk）
    text = getattr(outputs, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    return ""


# ==================== 主函数 ====================
async def main():
    logger.info("🚀 多 Agent 系统启动中...")
    
    supervisor_wrapper = None
    try:
        supervisor_wrapper = await create_supervisor()
        supervisor = supervisor_wrapper.supervisor
    except Exception as e:
        logger.critical(f"创建 Supervisor 失败: {e}")
        print("\n❌ 系统初始化失败，请检查配置和网络。")
        return

    logger.info("✅ 系统启动成功")
    print("\n🤖 多 Agent 协作系统已启动")
    print("输入 'q' 退出，输入 '/reset' 重置对话记忆\n")
    print("💡 修改 .env 文件后，发送任意消息即可自动加载新配置")
    
    thread_id = "user001"
    
    try:
        while True:
            # ===== 配置热更新检测 =====
            if config.check_and_reload():
                logger.info("🔄 配置已更新，下次请求将使用新配置")
                print("\n💡 部分配置（如模型温度）需要重启 Agent 才能完全生效")
                print("   按 Ctrl+C 退出后重新运行即可")
            
            try:
                user_input = input("\n👤 你: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == 'q':
                    logger.info("用户主动退出")
                    break
                if user_input.lower() == '/reset':
                    # 生成新的 thread_id，清空当前对话记忆，避免历史污染
                    thread_id = f"user_{uuid.uuid4().hex[:8]}"
                    logger.info(f"🔄 已重置对话，新 thread_id: {thread_id}")
                    print(f"\n🔄 对话已重置，新 thread_id: {thread_id}")
                    continue

                request_id = f"req_{uuid.uuid4().hex[:8]}"
                context.set_request_id(request_id)

                # ===== ✅ 关键修复：将 callbacks 放入 config_dict =====
                config_dict = {
                    "configurable": {
                        "thread_id": thread_id,
                        "request_id": request_id
                    },
                    "recursion_limit": config.RECURSION_LIMIT,
                    "callbacks": [langfuse_handler] if langfuse_handler else []   # ✅ Langfuse 追踪（可选）
                }

                logger.info(f"[{request_id}] 用户请求: {user_input[:100]}...")

                print("🤖 助手: ", end="", flush=True)
                full_response = ""
                stream_error = None

                try:
                    # 改用 ainvoke 获取完整结果：直接从最终 state 的 messages 提取
                    # Worker 的最终回答。这比 astream_events 更可靠，兼容 thinking 模型。
                    final_state = await supervisor.ainvoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=config_dict,
                    )
                    msgs = final_state.get("messages", [])
                    # 诊断：打印每轮最后 3 条消息，方便排查串台 / 0 字符
                    logger.info(
                        f"[{request_id}] final_state.messages 共 {len(msgs)} 条，最后 3 条: "
                        + " | ".join(
                            f"[{i}]{getattr(m,'type','?')}"
                            f"(worker={ (getattr(m,'additional_kwargs',{}) or {}).get('_worker_reply') })"
                            f"={str(getattr(m,'content',''))[:30]!r}"
                            for i, m in enumerate(msgs[-3:], start=max(0,len(msgs)-3))
                        )
                    )
                    # 只取「本轮 Worker 的回答」：最后一条消息必须是带 _worker_reply 标记的 AI。
                    # 绝不能从整个历史里 reversed 找 AI——多轮对话时那会取到上一轮的旧回答（串台根源）。
                    last = msgs[-1] if msgs else None
                    if last is not None and getattr(last, "type", "") == "ai":
                        ak = getattr(last, "additional_kwargs", {}) or {}
                        if ak.get("_worker_reply"):
                            txt = getattr(last, "content", "")
                            if isinstance(txt, str) and txt.strip():
                                full_response = txt
                            else:
                                txt2 = ak.get("reasoning_content") or ak.get("content") or ""
                                if txt2:
                                    full_response = txt2
                        else:
                            # 最后一条是 AI 但无标记（如 supervisor 正常结束、无 worker 输出）
                            full_response = ""
                    else:
                        # 最后一条不是 AI：说明本轮未产生 worker 回答（可能被路由到 __end__）
                        full_response = ""
                except openai.APIConnectionError as e:
                    stream_error = "网络连接失败，请检查网络或VPN后重试"
                    logger.error(f"[{request_id}] OpenAI 网络错误: {e}")
                except openai.APIStatusError as e:
                    stream_error = f"API 服务异常: {e.status_code}，请稍后重试"
                    logger.error(f"[{request_id}] API 状态错误: {e}")
                except openai.RateLimitError as e:
                    stream_error = "API 调用频率过高，请稍后重试"
                    logger.error(f"[{request_id}] 限流错误: {e}")
                except openai.AuthenticationError as e:
                    stream_error = "API 认证失败，请检查 API Key 配置"
                    logger.error(f"[{request_id}] 认证错误: {e}")
                except ConnectionError as e:
                    stream_error = "网络连接失败，请检查网络后重试"
                    logger.error(f"[{request_id}] 网络错误: {e}")
                except TimeoutError as e:
                    stream_error = "请求超时，请稍后重试"
                    logger.error(f"[{request_id}] 超时错误: {e}")
                except Exception as e:
                    stream_error = f"系统处理出错: {type(e).__name__}: {e}"
                    logger.error(f"[{request_id}] 请求失败: {type(e).__name__} - {e}")

                # 输出完整回答
                if full_response:
                    print(full_response, end="", flush=True)
                if stream_error:
                    print(f"\n❌ {stream_error}")
                    logger.warning(f"[{request_id}] 返回错误提示: {stream_error}")
                else:
                    logger.info(f"[{request_id}] 响应长度: {len(full_response)} 字符")
                    print()

            except KeyboardInterrupt:
                logger.warning("用户中断 (Ctrl+C)")
                print("\n👋 已退出")
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}")
                print(f"\n❌ 系统错误: {e}")
    finally:
        if supervisor_wrapper:
            await supervisor_wrapper.close()
            logger.info("🔌 数据库连接已关闭")
        # 确保 Langfuse 数据发送完毕（可选）
        if langfuse is not None:
            langfuse.flush()


# ==================== 程序入口 ====================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 已退出")
    except Exception as e:
        logger.critical(f"致命错误: {e}")
        print(f"\n💥 程序崩溃: {e}")
