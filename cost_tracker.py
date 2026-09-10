# cost_tracker.py
"""Token 成本埋点：按「会话 / 模型 / 节点」记账，支持预算阈值告警。

━━━ 为什么 Agent 应用必须做成本埋点 ━━━
普通接口的调用成本是确定的；Agent 不是——它一次请求可能触发 3~5 次 LLM 调用
（Supervisor 路由 + Worker 决策 + 工具后总结 + 记忆抽取），
再加上多轮对话和重试，**单次请求的 token 消耗可能相差一个数量级**。

没有埋点的后果很具体：死循环或提示词膨胀导致的成本失控**发现不了**，
等到收到账单已经烧完。业界真实事故里就有"Agent 死循环一夜烧掉几百块"。

━━━ 本模块做什么 ━━━
1. 记录每一次 LLM 调用的 prompt/completion token，落到 SQLite（按 session 可查）；
2. 支持按会话/全局汇总，算出费用；
3. 超过 `COST_ALERT_PER_SESSION` 阈值时告警（可对接降级策略）。

━━━ 和 Harness 的分工 ━━━
harness.py 管的是「工具执行」的有界性（超时/重试/防死循环）；
本模块管的是「模型调用」的账。两者一横一纵，共同构成 Agent 上生产的成本与可靠性护栏。
"""
import os
import sqlite3
from datetime import datetime

from loguru import logger

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost.db")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_usage (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            model             TEXT,
            node              TEXT,
            prompt_tokens     INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cost              REAL DEFAULT 0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_session ON token_usage(session_id)")
    conn.commit()
    return conn


def _price_per_million():
    """从 config 读取单价（元 / 百万 token）。未配置则为 0，只统计 token 不算钱。"""
    try:
        from config import config
        return (
            float(getattr(config, "PRICE_INPUT_PER_M", 0) or 0),
            float(getattr(config, "PRICE_OUTPUT_PER_M", 0) or 0),
        )
    except Exception:
        return 0.0, 0.0


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = _price_per_million()
    return round(
        prompt_tokens / 1_000_000 * pin + completion_tokens / 1_000_000 * pout, 6
    )


def record_usage(session_id: str, model: str, node: str,
                 prompt_tokens: int, completion_tokens: int) -> dict:
    """记一次 LLM 调用的账。返回该条记录（含费用）。"""
    cost = compute_cost(prompt_tokens or 0, completion_tokens or 0)
    conn = _init_db()
    try:
        conn.execute(
            "INSERT INTO token_usage(ts, session_id, model, node, prompt_tokens, completion_tokens, cost) "
            "VALUES(?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), session_id or "unknown",
             model or "", node or "", prompt_tokens or 0, completion_tokens or 0, cost),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        f"[cost] session={session_id} node={node} "
        f"prompt={prompt_tokens} completion={completion_tokens} cost={cost}"
    )
    check_budget(session_id)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost": cost}


def session_summary(session_id: str) -> dict:
    """按会话汇总：调用次数 / token / 费用 / 各节点占比。"""
    if not os.path.exists(DB_PATH):
        return {"session_id": session_id, "calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "by_node": {}}
    conn = _init_db()
    try:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            "COALESCE(SUM(cost),0) FROM token_usage WHERE session_id=?",
            (session_id,),
        ).fetchone()
        by_node = {
            n: {"calls": c, "prompt_tokens": p, "completion_tokens": m}
            for n, c, p, m in conn.execute(
                "SELECT node, COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0) "
                "FROM token_usage WHERE session_id=? GROUP BY node",
                (session_id,),
            ).fetchall()
        }
        calls, pt, ct, cost = row
        return {
            "session_id": session_id, "calls": calls,
            "prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": pt + ct, "cost": round(cost, 6), "by_node": by_node,
        }
    finally:
        conn.close()


def global_summary() -> dict:
    """全局汇总（运维/面试演示用）。"""
    if not os.path.exists(DB_PATH):
        return {"calls": 0, "total_tokens": 0, "cost": 0.0, "sessions": 0}
    conn = _init_db()
    try:
        calls, pt, ct, cost, sessions = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            "COALESCE(SUM(cost),0), COUNT(DISTINCT session_id) FROM token_usage"
        ).fetchone()
        return {"calls": calls, "prompt_tokens": pt, "completion_tokens": ct,
                "total_tokens": pt + ct, "cost": round(cost, 6), "sessions": sessions}
    finally:
        conn.close()


def check_budget(session_id: str) -> bool:
    """会话费用超过阈值则告警。返回是否超限（调用方可据此降级到更便宜的模型）。"""
    try:
        from config import config
        limit = float(getattr(config, "COST_ALERT_PER_SESSION", 0) or 0)
    except Exception:
        return False
    if limit <= 0:
        return False
    spent = session_summary(session_id)["cost"]
    if spent > limit:
        logger.warning(f"🚨 [cost] 会话 {session_id} 累计费用 {spent} 已超过阈值 {limit}")
        return True
    return False


# ---------------------------------------------------------------------------
# LangChain 回调：自动记账（挂到 config["callbacks"] 即可，业务代码零侵入）
# ---------------------------------------------------------------------------
try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - 无 langchain 时模块仍可独立导入
    BaseCallbackHandler = object


def _extract_usage(response) -> tuple:
    """尽力取出 (prompt_tokens, completion_tokens, model)。

    三种返回结构都要兼容（实测踩过坑，缺一种就静默记不到账）：
      1. AIMessage  —— astream_events 的 on_chat_model_end 在本机版本里就是它；
      2. ChatResult —— generations 是扁平的 [ChatGeneration]；
      3. LLMResult  —— generations 是嵌套的 [[Generation]]，用量在 llm_output.token_usage。
    """
    # 情况 1：直接传进来的就是一个消息对象
    if hasattr(response, "usage_metadata") or hasattr(response, "response_metadata"):
        pt, ct = _usage_from_message(response)
        if pt or ct:
            model = (getattr(response, "response_metadata", {}) or {}).get("model_name", "")
            return pt, ct, model

    llm_output = getattr(response, "llm_output", None) or {}
    for key in ("token_usage", "usage"):
        u = llm_output.get(key)
        if isinstance(u, dict):
            pt = u.get("prompt_tokens") or u.get("input_tokens") or 0
            ct = u.get("completion_tokens") or u.get("output_tokens") or 0
            if pt or ct:
                return int(pt), int(ct), llm_output.get("model_name", "")

    # 兜底：从 message 上取。注意两种返回结构都要兼容：
    #   · LLMResult  → generations[0][0].message（嵌套列表）
    #   · ChatResult → generations[0].message  （扁平）
    # 只处理一种的话，astream_events 的 on_chat_model_end 会静默记不到账。
    msg = _first_message(response)
    if msg is not None:
        pt, ct = _usage_from_message(msg)
        if pt or ct:
            model = (getattr(msg, "response_metadata", {}) or {}).get("model_name", "")
            return pt, ct, model
    return 0, 0, ""


def _first_message(response):
    """从 LLMResult / ChatResult 里取出第一条消息（两种结构都兼容）。"""
    gens = getattr(response, "generations", None)
    if not gens:
        return None
    g0 = gens[0]
    if isinstance(g0, (list, tuple)):        # LLMResult 的嵌套结构
        g0 = g0[0] if g0 else None
    return getattr(g0, "message", None)


def _usage_from_message(msg) -> tuple:
    """先看 usage_metadata（新），再看 response_metadata.token_usage（DeepSeek/OpenAI 兼容接口）。"""
    um = getattr(msg, "usage_metadata", None) or {}
    pt = um.get("input_tokens") or 0
    ct = um.get("output_tokens") or 0
    if pt or ct:
        return int(pt), int(ct)
    rm = getattr(msg, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    if isinstance(tu, dict):
        return int(tu.get("prompt_tokens") or 0), int(tu.get("completion_tokens") or 0)
    return 0, 0


def record_from_event(event: dict, session_id: str):
    """从 astream_events 的 on_chat_model_end 事件记账。

    比回调更准：事件自带 metadata[`langgraph_node`]，
    能直接看出「这次调用是 Supervisor 路由花的，还是某个 Worker 花的」。
    """
    try:
        output = (event.get("data") or {}).get("output")
        pt, ct, model = _extract_usage(output)
        if not pt and not ct:
            return None
        md = event.get("metadata") or {}
        # 节点归属：langgraph_node 在 create_agent 子图内部会退化成 "model" / "tools"，
        # 看不出是哪个 Worker 花的钱。langgraph_checkpoint_ns 形如
        #   "policy_agent:uuid|model:uuid"
        # 取**最外层**那一段才是真正的 Worker 名。
        node = md.get("langgraph_node") or "unknown"
        ns = md.get("langgraph_checkpoint_ns") or ""
        segments = [seg.split(":")[0] for seg in ns.split("|") if seg]
        if segments:
            node = segments[0]
        return record_usage(session_id, model, str(node), pt, ct)
    except Exception as e:
        logger.warning(f"[cost] 事件记账失败（已忽略，不影响主流程）: {e}")
        return None


def record_message_usage(session_id: str, node: str, message):
    """从 AIMessage.usage_metadata 记账（用于图外的 LLM 调用，如长期记忆抽取）。"""
    try:
        um = getattr(message, "usage_metadata", None) or {}
        pt = int(um.get("input_tokens", 0) or 0)
        ct = int(um.get("output_tokens", 0) or 0)
        if not pt and not ct:
            return None
        model = (getattr(message, "response_metadata", {}) or {}).get("model_name", "")
        return record_usage(session_id, model, node, pt, ct)
    except Exception as e:
        logger.warning(f"[cost] 消息记账失败（已忽略）: {e}")
        return None


class CostTrackerCallback(BaseCallbackHandler):
    """把 LangChain 的每次 LLM 调用自动记到成本账上。"""

    def __init__(self, session_id: str = "unknown"):
        try:
            super().__init__()
        except Exception:
            pass
        self.session_id = session_id

    def on_llm_end(self, response, *, run_id=None, parent_run_id=None,
                   tags=None, metadata=None, **kwargs):
        try:
            pt, ct, model = _extract_usage(response)
            if not pt and not ct:
                return
            node = (metadata or {}).get("langgraph_node", "unknown")
            record_usage(self.session_id, model, node, pt, ct)
        except Exception as e:
            # 记账失败绝不能影响主流程
            logger.warning(f"[cost] 记账失败（已忽略）: {e}")
