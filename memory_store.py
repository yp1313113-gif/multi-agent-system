# memory_store.py
"""长期记忆（Long-term Memory）：跨会话记住用户的事实。

━━━ 和「会话记忆」的区别 ━━━
· 会话记忆：LangGraph 的 AsyncSqliteSaver 按 thread_id 持久化对话历史，只对本次会话有效，
  换一个 session 就"失忆"。
· 长期记忆（本模块）：按 session_id 记住**关于用户的事实**（偏好 / 身份 / 常用项目），
  下次对话开始时回读并注入，实现"这个用户上次说过……"。

━━━ 设计要点（面试考点）━━━
写记忆最大的风险不是"记不住"，而是**记忆污染**：用户随口一句被永久记住，
之后每次回答都被这条错误记忆带偏，而且很难发现。

所以本实现做了**三重准入判断**（这是核心，不是简单的 key-value 落库）：

  1. 【白名单 key】只允许 preference / identity / project 三类，其余一律丢弃。
  2. 【内容校验】拒绝问句、超长文本、以及"今天/这次/临时"这类时效性表述。
  3. 【长度上限】单条事实 <= 60 字，避免把整段对话塞进来。

这与 tools/rag_tool.py 里 _cacheable()（只缓存带引用的答案）是同一种工程思想：
**只写"确定性、可复用、非临时"的东西**，宁可少写，不可写错。
"""
import os
import sqlite3
import re
from datetime import datetime

from loguru import logger

# 基于本文件所在目录的绝对路径（与 hitl.py 同款处理，避免不同启动目录读写不同的库）
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")

# ── 准入规则 ──────────────────────────────────────────────
ALLOWED_KEYS = {"preference", "identity", "project"}
MAX_VALUE_LEN = 60

# 时效性表述：这类内容是"当下状态"，不是"用户事实"，不该长期记住
_TRANSIENT_WORDS = ["今天", "这次", "刚才", "临时", "现在", "本周", "等下", "马上"]
# 问句特征：用户在提问，不是在陈述自己的事实
_QUESTION_MARKS = ["？", "?", "吗", "呢", "多少", "怎么", "为什么", "是不是"]


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_facts (
            session_id TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (session_id, key)
        )
        """
    )
    conn.commit()
    return conn


def is_admissible(key: str, value: str) -> bool:
    """准入判断：这条事实值不值得长期记住？（返回 False 就必须丢弃）"""
    if key not in ALLOWED_KEYS:
        return False
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v or len(v) > MAX_VALUE_LEN:
        return False
    # 时效性表述 → 是临时状态，不是用户事实
    if any(w in v for w in _TRANSIENT_WORDS):
        return False
    # 问句 → 用户在提问，不是陈述事实
    if any(q in v for q in _QUESTION_MARKS):
        return False
    return True


def remember(session_id: str, facts: dict) -> dict:
    """写入事实（自动过滤不合格项）。返回实际写入的 {key: value}。"""
    if not session_id or not facts:
        return {}

    accepted = {}
    for k, v in facts.items():
        if is_admissible(k, v):
            accepted[k] = str(v).strip()
        else:
            logger.debug(f"[memory] 拒绝写入（未通过准入）: {k}={v!r}")

    if not accepted:
        return {}

    conn = _init_db()
    try:
        now = datetime.now().isoformat(timespec="seconds")
        conn.executemany(
            "INSERT INTO user_facts(session_id, key, value, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(session_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            [(session_id, k, v, now) for k, v in accepted.items()],
        )
        conn.commit()
        logger.info(f"[memory] 已记住 {len(accepted)} 条事实 session={session_id}: {accepted}")
        return accepted
    finally:
        conn.close()


def recall(session_id: str) -> dict:
    """读取该 session 的全部长期记忆。"""
    if not session_id or not os.path.exists(DB_PATH):
        return {}
    conn = _init_db()
    try:
        rows = conn.execute(
            "SELECT key, value FROM user_facts WHERE session_id = ? ORDER BY updated_at DESC",
            (session_id,),
        ).fetchall()
        return {k: v for k, v in rows}
    except Exception as e:
        logger.warning(f"[memory] 读取失败: {e}")
        return {}
    finally:
        conn.close()


def forget(session_id: str, key: str = None) -> int:
    """删除记忆（key 为空则清空该 session 全部记忆）。返回删除条数。"""
    conn = _init_db()
    try:
        if key:
            cur = conn.execute("DELETE FROM user_facts WHERE session_id=? AND key=?", (session_id, key))
        else:
            cur = conn.execute("DELETE FROM user_facts WHERE session_id=?", (session_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def build_memory_prompt(session_id: str) -> str:
    """把长期记忆拼成可注入的提示块（无记忆则返回空串）。"""
    facts = recall(session_id)
    if not facts:
        return ""
    labels = {"preference": "偏好", "identity": "身份", "project": "常用项目"}
    lines = [f"- {labels.get(k, k)}：{v}" for k, v in facts.items()]
    return "【关于该用户的长期记忆】\n" + "\n".join(lines) + "\n（可自然引用，但不要生硬复述）"


EXTRACT_PROMPT = """从下面这轮对话里，抽取「值得长期记住的用户事实」。

只允许三类 key：
- preference：用户的偏好（如"喜欢简洁回答""偏好中文回复"）
- identity：用户身份（如"是财务负责人""在制造业做研发管理"）
- project：用户常用的项目/系统名

严格要求（违反则不要输出）：
1. 不抽取时效性内容（带"今天/这次/现在"等）；
2. 不抽取用户提出的问题本身；
3. 每条 value 不超过 40 字；
4. 没有值得记住的就输出 {{}}。

只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块。

用户说：{user}
助手答：{assistant}
"""


def parse_facts_json(text: str) -> dict:
    """从 LLM 输出里稳健地解析出 facts JSON（容忍代码块包裹与多余文字）。"""
    if not text:
        return {}
    t = text.strip()
    # 去掉 markdown 代码块围栏
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    # 截取第一个 { 到最后一个 }
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return {}
    import json
    try:
        data = json.loads(t[i:j + 1])
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    def _norm(v) -> str:
        # 模型有时会返回列表（{"identity": ["财务负责人"]}），
        # 直接 str() 会存成 "['财务负责人']" —— 这种脏值会污染后续所有提示词。
        if isinstance(v, (list, tuple)):
            v = "、".join(str(x) for x in v) if v else ""
        return str(v).strip()

    out = {}
    for k, v in data.items():
        val = _norm(v)
        if val and val not in ("[]", "{}", "None", "null"):
            out[str(k)] = val
    return out


async def extract_and_remember(llm, session_id: str, user_message: str, assistant_answer: str) -> dict:
    """用一次便宜的 LLM 调用抽取事实并写入长期记忆（失败不影响主流程）。"""
    try:
        prompt = EXTRACT_PROMPT.format(user=user_message[:300], assistant=(assistant_answer or "")[:300])
        resp = await llm.ainvoke(prompt)

        # 记忆抽取本身也是一次 LLM 调用，同样要记账，否则成本会被系统性低估
        try:
            import cost_tracker
            cost_tracker.record_message_usage(session_id, "memory_extract", resp)
        except Exception:
            pass

        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        facts = parse_facts_json(text)
        if not facts:
            return {}
        return remember(session_id, facts)
    except Exception as e:
        logger.warning(f"[memory] 事实抽取失败（不影响主流程）: {e}")
        return {}
