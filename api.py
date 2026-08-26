# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import os
from agent import stream_chat
from concurrency import limiter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/chat")
async def chat(message: str, session: str = "user001"):
    """
    流式输出接口（SSE）。并发控制：进入即尝试获取并发名额，
    满则排队（最长 CONCURRENCY_QUEUE_TIMEOUT 秒），超时返回 503——
    保护下游 LLM API / 数据库不被瞬时并发打爆。
    """
    if not await limiter.acquire():
        return JSONResponse(
            status_code=503,
            content={"detail": "系统繁忙：当前请求过多，请稍后重试"},
        )

    async def event_generator():
        try:
            async for token in stream_chat(message, session):
                # 使用标准的 SSE 格式
                yield f"data: {token}\n\n"
        finally:
            limiter.release()   # 流结束才释放名额

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        }
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """聊天前端页面（访问 / 即对话界面）。"""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "chat.html"), encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return {"status": "ok", "message": "AI Agent is running"}


@app.get("/health")
async def health():
    return {"status": "ok", "message": "AI Agent is running"}