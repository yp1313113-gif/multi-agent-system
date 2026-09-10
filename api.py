# api.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import os
from agent import stream_chat
from concurrency import limiter
from warmup import warmup_all


@asynccontextmanager
async def lifespan(app):
    """服务启动/关闭钩子。

    启动时做**预热**：把向量模型、重排模型、BM25 索引、编排图的构建
    从「首个用户请求」挪到「服务启动」。实测冷启动 TTFT 26.6s → 热态 4.9s，
    差的这 20 秒不该由第一个用户买单。

    warmup_all 是尽力而为的：任何一步失败只告警，不会阻断服务启动。
    """
    report = await warmup_all()
    app.state.warmup_report = report
    yield


app = FastAPI(lifespan=lifespan)

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
                # SSE 的 data 行不能直接携带换行：前端按 "\n\n" 分帧，
                # 一个裸 \n token 会变成空帧被直接吞掉 —— 结果是模型的分点列表
                # 全部挤成一行（实测踩到的 bug）。
                # 所以必须转义：反斜杠 → \\，换行 → \n，前端再做反转义。
                safe = token.replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n")
                yield f"data: {safe}\n\n"
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
    """健康检查：附带缓存后端与预热结果，便于运维一眼看出降级状态。"""
    from cache import cache
    return {
        "status": "ok",
        "message": "AI Agent is running",
        "cache_backend": cache.backend,      # redis / local-lru（降级可见）
        "warmup": getattr(app.state, "warmup_report", None),
    }


@app.get("/skills")
async def skills():
    """能力清单：系统当前具备哪些技能。

    直接从技能库元数据生成，不手工维护 —— 代码和文档不会对不上。
    """
    try:
        from skills.registry import list_skills
        items = list_skills()
        return {"count": len(items), "skills": items}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"技能库读取失败: {e}"})


@app.get("/cost")
async def cost(session: str = None):
    """成本观测：token 用量与费用（按会话或全局）。

    面试演示点：Agent 一次请求会触发多次 LLM 调用（路由/决策/总结/记忆抽取），
    没有埋点就看不出钱花在哪个节点上。
    """
    try:
        from cost_tracker import session_summary, global_summary
        if session:
            return session_summary(session)
        return global_summary()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"成本读取失败: {e}"})


if __name__ == "__main__":
    # 本地启动入口。此前 api.py 缺少这一段，README 写的 `python api.py` 实际
    # 只会导入模块然后退出（服务器根本没起来）——部署文档里用的是 uvicorn api:app。
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    print(f"🚀 启动研发费用 Agent 服务： http://127.0.0.1:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)