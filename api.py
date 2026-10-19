# api.py
from contextlib import asynccontextmanager

from fastapi import UploadFile, File, FastAPI
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

    设计要点：Agent 一次请求会触发多次 LLM 调用（路由/决策/总结/记忆抽取），
    没有埋点就看不出钱花在哪个节点上。
    """
    try:
        from cost_tracker import session_summary, global_summary
        if session:
            return session_summary(session)
        return global_summary()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"成本读取失败: {e}"})


@app.get("/compression")
async def compression_stats():
    """上下文压缩观测：当前配置 + 最近一次压缩报告。

    设计要点：压缩是「先确定性、再摘要」的两段式 ——
    第一段（窗口截断 / 大结果落盘 / head-middle-tail）不调模型、不花钱；
    只有压完仍然超阈值，才调摘要模型。多数请求在第一段就结束了。
    """
    try:
        import compression
        return compression.stats()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"压缩模块读取失败: {e}"})


@app.get("/middleware")
async def middleware_stats():
    """中间件链观测：工具边界链的顺序 + 各工具的调用指标。

    设计要点：Agent 的横切关注点（审批 / 循环守卫 / 重试 / 超时 / 计时 /
    结果卸载 / 指标）是一条**有序链**，不是散落在各处的装饰器。
    顺序有语义：重试在外层 → 每次重试都有独立超时。
    """
    try:
        import middleware
        return middleware.describe()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"中间件链读取失败: {e}"})



# ============================ 业务看板接口 ============================
# 数据源：demo_seed.snapshot() 生成的「看板状态.json」
# 为什么不直接读业务黑板：contextvars 的黑板是请求级的，跨请求读不到。

from fastapi import Body

_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "看板状态.json")


def _read_state() -> dict:
    import json
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "看板状态还没生成，请先点「重新跑一遍」"}
    except Exception as e:
        return {"error": f"读取看板状态失败: {e}"}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """业务看板页面（业务流程 + 加计扣除测算 + 人工审核）。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "dashboard.html")
    with open(p, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/state")
async def api_state():
    """返回看板快照。

    如果「用户输入数据.json」比快照新（比如刚在聊天页填了一条工时），
    先重新生成一次快照 —— 这样从聊天切到看板就能直接看到新数据，
    不用手动点「重新跑一遍」。
    """
    try:
        uf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "用户输入数据.json")
        sf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "看板状态.json")
        if os.path.exists(uf) and (not os.path.exists(sf) or os.path.getmtime(uf) > os.path.getmtime(sf)):
            _run_seed(no_reset=False)
    except Exception as e:
        # 生成失败不能让看板打不开，退回返回现有快照
        import traceback
        traceback.print_exc()
    return _read_state()


def _run_seed(no_reset: bool = False) -> dict:
    """重新生成看板快照。

    种子数据在「演示种子数据.json」里，demo_seed 每次都读文件，
    所以这里直接调用即可 —— 改了种子数据不需要重启服务。
    """
    import demo_seed
    st = demo_seed.snapshot(regenerate=not no_reset)
    demo_seed.write(st)
    return {"ok": True, "generated_at": st.get("generated_at"),
            "pending": len(st.get("pending") or []), "rules": len(st.get("rules") or [])}


@app.post("/api/run")
async def api_run():
    """重跑一遍业务链，重新生成快照（子进程执行）。"""
    return _run_seed(no_reset=False)


@app.post("/api/review")
async def api_review(payload: dict = Body(...)):
    """提交人工裁定，并沉淀成规则。

    payload: {"id": "approval_xxx", "decision": "其他相关费用", "rationale": "..."}
    """
    import demo_seed
    import hitl as _hitl
    aid = (payload or {}).get("id", "")
    decision = (payload or {}).get("decision", "")
    rationale = (payload or {}).get("rationale", "")
    if not aid or not decision:
        return {"ok": False, "error": "缺少 id 或 decision"}
    ok = _hitl.resolve(aid, decision, rationale)
    _run_seed(no_reset=True)
    return {"ok": bool(ok), "state": _read_state()}



# ---------------------------- 自定义录入 ----------------------------
# 用户在浏览器的看板上自己填工时 / 填费用，追加到「用户输入数据.json」，
# 然后重新生成快照。这样演示时可以说「您随便给一笔，我现在就算给您看」。

_USER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "用户输入数据.json")


def _read_user() -> dict:
    import json
    try:
        with open(_USER_FILE, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        d = {}
    d.setdefault("timesheets", [])
    d.setdefault("expenses", [])
    return d


def _write_user(d: dict) -> None:
    import json
    with open(_USER_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


@app.post("/api/add/timesheet")
async def api_add_timesheet(payload: dict = Body(...)):
    """追加一条工时记录。"""
    p = payload or {}
    emp = (p.get("employee") or "").strip()
    proj = (p.get("project") or "").strip()
    date = (p.get("date") or "").strip() or "2026-10-19"
    try:
        hours = float(p.get("hours") or 0)
    except (TypeError, ValueError):
        hours = 0
    if not emp or not proj:
        return {"ok": False, "error": "人员和项目必填"}
    if hours <= 0 or hours > 24:
        return {"ok": False, "error": "工时必须在 0.5 ~ 24 小时之间"}
    d = _read_user()
    d["timesheets"].append({"employee": emp, "project": proj, "date": date,
                            "hours": hours, "task": (p.get("task") or "").strip()})
    _write_user(d)
    _run_seed(no_reset=False)
    return {"ok": True, "state": _read_state()}


@app.post("/api/add/expense")
async def api_add_expense(payload: dict = Body(...)):
    """追加一笔费用。金额超阈值或把握度低时，会自动进入人工复核队列。"""
    p = payload or {}
    desc = (p.get("description") or "").strip()
    cat = (p.get("category") or "其他相关费用").strip()
    try:
        amt = float(p.get("amount") or 0)
    except (TypeError, ValueError):
        amt = 0
    try:
        conf = float(p.get("confidence") if p.get("confidence") is not None else 1.0)
    except (TypeError, ValueError):
        conf = 1.0
    if not desc:
        return {"ok": False, "error": "费用说明必填"}
    if amt <= 0:
        return {"ok": False, "error": "金额必须大于 0"}
    d = _read_user()
    d["expenses"].append({"category": cat, "amount": amt,
                          "description": desc, "confidence": conf})
    _write_user(d)
    _run_seed(no_reset=False)
    return {"ok": True, "state": _read_state()}


@app.post("/api/reset")
async def api_reset():
    """清空用户自己填的内容，回到内置演示数据。"""
    _write_user({"timesheets": [], "expenses": []})
    _run_seed(no_reset=False)
    return {"ok": True, "state": _read_state()}



# ---------------------------- 费用导入 ----------------------------
# 财务从 ERP 导出明细账，整理成模板格式后上传。
# 支持 .csv 和 .xlsx（xlsx 用 import_tool 里的零依赖解析器，不需要 openpyxl）。

@app.get("/api/template/{kind}")
async def api_template(kind: str):
    """下载导入模板。kind = csv | xlsx"""
    from fastapi.responses import FileResponse
    import import_tool
    here = os.path.dirname(os.path.abspath(__file__))
    kind = (kind or "csv").lower()
    if kind == "xlsx":
        p = os.path.join(here, "费用导入模板.xlsx")
        if not os.path.exists(p):
            import_tool.write_template_xlsx(p)
        return FileResponse(p, filename="费用导入模板.xlsx",
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    p = os.path.join(here, "费用导入模板.csv")
    if not os.path.exists(p):
        import_tool.write_template_csv(p)
    return FileResponse(p, filename="费用导入模板.csv", media_type="text/csv")


# 导入分两步：先 preview（只解析、不写库），用户确认后 commit。
# 为什么要两步：财务传错文件再传一次就把数据翻倍了，
# 而研发费用数据一旦翻倍，辅助账、限额、加计扣除全错。必须先让人看一眼。

_IMPORT_CACHE = {}   # batch_id -> {rows, excluded, errors, file, ts}
_IMPORT_TTL = 600    # 预览结果保留 10 分钟


def _prune_imports():
    import time as _t
    now = _t.time()
    for k in list(_IMPORT_CACHE.keys()):
        if now - _IMPORT_CACHE[k]["ts"] > _IMPORT_TTL:
            _IMPORT_CACHE.pop(k, None)


@app.post("/api/import/preview")
async def api_import_preview(file: UploadFile = File(...)):
    """第一步：只解析和校验，不写任何数据。返回预览给用户确认。"""
    import import_tool
    import tempfile
    import time as _t

    _prune_imports()

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".csv"
    if suffix not in (".csv", ".xlsx", ".xlsm", ".xls"):
        return {"ok": False, "error": "只支持 .csv 和 .xlsx 文件"}
    if suffix == ".xls":
        return {"ok": False, "error": "不支持老版 .xls。请在 Excel 里另存为 .xlsx 或 .csv 再上传。"}

    data = await file.read()
    if not data:
        return {"ok": False, "error": "文件是空的"}
    if len(data) > 5 * 1024 * 1024:
        return {"ok": False, "error": "文件超过 5MB，请拆分后再导入"}

    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        parsed = import_tool.read_file(tmp)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    rows = parsed.get("rows") or []
    errors = parsed.get("errors") or []
    if not rows:
        return {"ok": False, "error": "没有解析出任何有效数据", "details": errors}

    # 判断哪些已经在库里了（按「说明 + 金额」判重）
    import rules as _rules
    d = _read_user()
    existing = {(e.get("description"), round(float(e.get("amount") or 0), 2))
                for e in (d.get("expenses") or [])}

    preview, excluded, dups = [], [], 0
    for r in rows:
        bad, kw = _rules.is_non_rd(r["description"])
        key = (r["description"], round(float(r["amount"]), 2))
        dup = key in existing
        if dup:
            dups += 1
        if bad:
            excluded.append({"description": r["description"], "amount": r["amount"], "keyword": kw})
            continue
        preview.append({
            "description": r["description"],
            "project": r["project"],
            "date": r["date"], "voucher": r["voucher"],
            "amount": r["amount"], "no_tax": r["no_tax"], "tax": r["tax"],
            "invoice_type": r["invoice_type"],
            "category": _rules.guess_category(r["description"]),
            "duplicate": dup,
        })

    batch_id = "imp_" + _t.strftime("%Y%m%d_%H%M%S")
    _IMPORT_CACHE[batch_id] = {"rows": preview, "excluded": excluded,
                               "errors": errors, "file": file.filename, "ts": _t.time()}

    return {
        "ok": True, "batch_id": batch_id, "file": file.filename,
        "parsed": len(rows), "rows": preview, "excluded": excluded,
        "errors": errors,
        "to_add": sum(1 for x in preview if not x["duplicate"]),
        "duplicates": dups,
        "blocked": len(excluded),
    }


@app.post("/api/import/commit")
async def api_import_commit(payload: dict = Body(...)):
    """第二步：用户确认后，真正写入。只写入未标记为重复的行。"""
    import rules as _rules
    batch_id = (payload or {}).get("batch_id", "")
    entry = _IMPORT_CACHE.get(batch_id)
    if not entry:
        return {"ok": False, "error": "预览已过期，请重新上传文件"}

    skip_dup = bool((payload or {}).get("skip_duplicates", True))
    d = _read_user()
    added = []
    for r in entry["rows"]:
        if skip_dup and r.get("duplicate"):
            continue
        d["expenses"].append({
            "category": r["category"],
            "amount": r["amount"],
            "description": r["description"],
            # 所属项目必须存下来：税务按项目归集，没有项目维度就算不出项目辅助账，
            # 也追不到这笔人工到底摊在哪个项目上。
            "project": r.get("project") or "",
            # 日期和凭证号也必须存：稽查时会按凭证号抽查原始单据，
            # 辅助账上没有凭证号，这笔费用就证明不了。
            "date": r.get("date") or "",
            "voucher": r.get("voucher") or "",
            "confidence": 0.9,
            "source": batch_id,
        })
        added.append(r)
    d.setdefault("import_batches", []).append({
        "id": batch_id, "file": entry["file"], "count": len(added),
    })
    _write_user(d)
    _IMPORT_CACHE.pop(batch_id, None)

    if not added:
        return {"ok": True, "added": 0, "message": "没有新增数据（可能全部是重复的）",
                "state": _read_state()}

    _run_seed(no_reset=False)
    st = _read_state()
    pending_desc = {p.get("description") for p in (st.get("pending") or [])}
    auto_n = sum(1 for a in added if a["description"] not in pending_desc)

    return {
        "ok": True, "batch_id": batch_id, "file": entry["file"],
        "added": len(added), "auto": auto_n, "pending": len(added) - auto_n,
        "excluded": entry["excluded"], "errors": entry["errors"],
        "rows": added, "state": st,
    }



@app.get("/api/ledger/export")
async def api_ledger_export():
    """导出辅助账 Excel（用于留存备查）。

    说明：政策没有规定辅助账必须是什么格式，只要求按研发项目设置辅助账、
    留存备查（财税〔2015〕119 号、国家税务总局公告 2015 年第 97 号）；
    2019 年第 41 号公告又取消了汇总表的报送，改为留存备查。
    导出 Excel 是因为实务中最方便，不是政策强制。
    """
    import tempfile
    from fastapi.responses import FileResponse
    import ledger_export

    st = _read_state()
    if not st or st.get("error"):
        return {"ok": False, "error": "还没有辅助账数据"}
    period = ((st.get("ledger") or {}).get("period") or "辅助账").replace("/", "-")
    fd, tmp = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        ledger_export.export_ledger(st, tmp)
        return FileResponse(
            tmp, filename=f"研发支出辅助账_{period}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=None,
        )
    except Exception as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return {"ok": False, "error": f"导出失败：{e}"}


if __name__ == "__main__":
    # 本地启动入口。此前 api.py 缺少这一段，README 写的 `python api.py` 实际
    # 只会导入模块然后退出（服务器根本没起来）——部署文档里用的是 uvicorn api:app。
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    print(f"🚀 启动研发费用 Agent 服务： http://127.0.0.1:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)