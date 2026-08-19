# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from agent import stream_chat

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
    流式输出接口，使用 StreamingResponse 替代 EventSourceResponse
    确保兼容性更好
    """
    async def event_generator():
        async for token in stream_chat(message, session):
            # 使用标准的 SSE 格式
            yield f"data: {token}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        }
    )


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Agent is running"}