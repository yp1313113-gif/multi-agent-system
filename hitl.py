# hitl.py
"""人工复核（Human-in-the-loop）：从「隐私审批」升级为「合规复核」。

━━━ 原实现的三个问题（代码复盘） ━━━
1. 【场景错位】触发词是 {"工资","年假","公积金","考勤","加班"} ——
   那是「员工问 HR」的话题，和研发费用归集 / 加计扣除 / 风险指标没关系，
   是从另一个场景直接移植过来的。
2. 【越权】审批记录没有绑定「谁发起的」：
   张三问「研发人员的工资明细」→ 批准 → 李四问同一句 →
   第一层「tool_input 精确匹配」命中 → 自动放行。
   人工审核的目的恰恰是「这笔数据不能随便给」，原实现却变成「批一次，对所有人永久放行」。
3. 【复用粒度太粗】第 3 层按关键词组交集匹配，命中同组任一关键词就放行。

━━━ 现在的设计 ━━━
· 触发条件换成研发费用场景的业务风险（见 needs_review）：
  归类置信度低 / 单笔金额超阈值 / 风险指标黄红灯 / 政策边界模糊 / 个人薪酬明细
· 审批记录绑定 requester（默认取当前会话）—— 只复用「同一个人」的裁定
· 复核产出是「裁定 + 依据」，而不是「批准/拒绝」，可沉淀成规则（rules.py）
· 旧库里没有发起人的记录统一标为 legacy:unbound —— 不删数据，但让它们永远不会被自动匹配命中

━━━ 为什么 requester 默认取 context 而不是参数传 ━━━
工具是在 Agent 内部被调用的，拿不到 HTTP 层参数；
agent.stream_chat 在请求开始时把 session 登记进 context，
工具侧就能读到「当前是谁在问」。
"""
import json
import os
import sqlite3
import uuid

from loguru import logger

from context import get_requester

# 使用基于本文件所在目录的绝对路径，避免不同启动目录导致读写不同的 approval.db
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approval.db")

# 单笔金额复核阈值（元）—— 大额支出需主管确认
AMOUNT_THRESHOLD = 100_000.0

# 归类置信度下限 —— 低于它说明一笔支出可能符合多个口径，归错会导致加计扣除被否
CONFIDENCE_FLOOR = 0.75

# 旧记录在被自动匹配时的占位发起人：任何真实 requester 都不会等于它
LEGACY_REQUESTER = "legacy:unbound"

# 研发费用领域的关键词组（第 3 层兜底匹配用，处理 LLM 改写导致 tool_input 变化）
# ★ 替换掉了原来的 {"年假","公积金","考勤","加班"} —— 那些是 HR 场景的词
OVERLAP_GROUPS = [
    {"工资", "薪酬", "薪资", "职工薪酬", "人员人工", "人工费用"},
    {"折旧", "固定资产", "设备折旧"},
    {"摊销", "无形资产", "软件摊销", "专利"},
    {"直接投入", "材料", "燃料", "动力", "试验", "检测"},
    {"委托研发", "委托开发", "外部研发", "外部机构"},
    {"其他相关费用", "差旅", "会议", "咨询", "专家", "知识产权"},
]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn, table: str) -> set:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approvals (
            id          TEXT PRIMARY KEY,
            tool_name   TEXT,
            tool_input  TEXT,
            user_message TEXT,
            context     TEXT,
            status      TEXT DEFAULT 'pending',
            result      TEXT,
            used        INTEGER DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            requester   TEXT,
            kind        TEXT DEFAULT 'legacy',
            reason      TEXT DEFAULT '',
            decision    TEXT,
            rationale   TEXT DEFAULT '',
            rule_id     TEXT
        )
        """
    )
    # 迁移旧库：补齐新增列（CREATE TABLE IF NOT EXISTS 不会加列）
    cols = _table_columns(conn, "approvals")
    for name, ddl in (
        ("requester", "TEXT"),
        ("kind", "TEXT DEFAULT 'legacy'"),
        ("reason", "TEXT DEFAULT ''"),
        ("decision", "TEXT"),
        ("rationale", "TEXT DEFAULT ''"),
        ("rule_id", "TEXT"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE approvals ADD COLUMN {name} {ddl}")
            logger.info(f"[hitl] 迁移：approvals 补列 {name}")

    # ★ 安全修复的关键一步：旧记录没有发起人，标成 legacy:unbound。
    #   不删数据，但让它们永远不会被任何真实 requester 命中。
    n = conn.execute(
        "UPDATE approvals SET requester = ? "
        "WHERE requester IS NULL AND status = 'approved' AND used = 0",
        (LEGACY_REQUESTER,),
    ).rowcount
    if n:
        logger.warning(f"[hitl] 已将 {n} 条无发起人的旧审批记录标记为 {LEGACY_REQUESTER}（不再自动放行）")

    conn.commit()
    conn.close()


# ============ 触发条件：什么才需要人工复核 ============

def needs_review(kind: str, payload: dict):
    """判断一条业务动作是否需要人工复核。

    返回 (是否需要, 原因)。四类触发**都是业务风险，不是隐私关键词**：

    Args:
        kind: 业务动作类型
            expense_classify —— 费用归类
            risk_fix         —— 风险指标整改
            policy_judge     —— 政策口径裁定
            salary_detail    —— 个人薪酬明细查询
        payload: 该动作的业务上下文
    """
    if kind == "expense_classify":
        amt = float(payload.get("amount", 0) or 0)
        conf = float(payload.get("confidence", 1.0))
        if conf < CONFIDENCE_FLOOR:
            return True, (f"归类置信度 {conf:.0%} 偏低（一笔支出可能同时符合多个口径，"
                          f"归错会导致加计扣除被否）")
        if amt >= AMOUNT_THRESHOLD:
            return True, f"单笔金额 ¥{amt:,.0f} 超过复核阈值 ¥{AMOUNT_THRESHOLD:,.0f}"

    if kind == "risk_fix":
        level = str(payload.get("level", "")).lower()
        if level in ("yellow", "red"):
            return True, f"风险指标为 {level}，需要确认是否整改及整改方案"

    if kind == "policy_judge":
        if not payload.get("grounded", True):
            return True, "知识库未收录明确依据，政策边界需要人工裁定"

    if kind == "salary_detail":
        return True, "查询涉及个人薪酬明细（人员人工费用），需数据权限复核"

    return False, ""


# ============ 复核记录 ============

def request_approval(tool_name, tool_input, user_message, context,
                     approval_id=None, requester=None,
                     kind="rag_search", reason=""):
    """登记一条待复核记录。

    Args:
        requester: 发起人；不传则取当前会话（context.get_requester()）
        kind: 业务动作类型，见 needs_review
        reason: 为什么要人判断（会展示在复核终端里）
    """
    init_db()
    if approval_id is None:
        approval_id = f"approval_{uuid.uuid4().hex[:8]}"
    requester = requester or get_requester() or "anonymous"

    conn = get_db()
    conn.execute(
        "INSERT INTO approvals (id, tool_name, tool_input, user_message, context, "
        "requester, kind, reason, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (approval_id, tool_name, tool_input, user_message,
         json.dumps(context, ensure_ascii=False), requester, kind, reason, "pending"),
    )
    conn.commit()
    conn.close()
    logger.info(f"📋 请求复核 [{approval_id}] kind={kind} requester={requester}｜{reason}")
    return approval_id


def _find_approved(question, tool_name, requester, cursor):
    """查找一条「同一个人」的、未消费的已批准记录。

    ★ 与原实现的关键差别：三层匹配**全部绑定了 requester**。

    Args:
        requester: 发起人 —— 只有同一个人此前批准过的记录才能被复用
    """
    # 1) 精确匹配
    row = cursor.execute(
        "SELECT id, tool_input FROM approvals WHERE tool_name = ? AND requester = ? "
        "AND tool_input = ? AND status = 'approved' AND used = 0 "
        "ORDER BY created_at LIMIT 1",
        (tool_name, requester, question),
    ).fetchone()
    if row:
        return row

    # 2) LIKE 模糊匹配
    if question:
        like_q = f"%{question}%"
        row = cursor.execute(
            "SELECT id, tool_input FROM approvals WHERE tool_name = ? AND requester = ? "
            "AND status = 'approved' AND used = 0 "
            "AND (tool_input LIKE ? OR user_message LIKE ?) "
            "ORDER BY created_at LIMIT 1",
            (tool_name, requester, like_q, like_q),
        ).fetchone()
        if row:
            return row

    # 3) 关键词交集兜底：处理 LLM 改写导致的 tool_input 变化
    #    （如"研发人员工资占比"被改写成"人员人工费用 职工薪酬 怎么算"）
    if question:
        q_keywords = {k for g in OVERLAP_GROUPS for k in g if k in question}
        if q_keywords:
            rows = cursor.execute(
                "SELECT id, tool_input, user_message FROM approvals "
                "WHERE tool_name = ? AND requester = ? AND status = 'approved' AND used = 0 "
                "ORDER BY created_at DESC",
                (tool_name, requester),
            ).fetchall()
            for r in rows:
                r_text = f"{r['tool_input']} {r['user_message']}"
                r_keywords = {k for g in OVERLAP_GROUPS for k in g if k in r_text}
                for group in OVERLAP_GROUPS:
                    if (q_keywords & group) and (r_keywords & group):
                        return r
    return None


def has_approved_query(question, tool_name, requester=None):
    init_db()
    requester = requester or get_requester() or "anonymous"
    conn = get_db()
    row = _find_approved(question, tool_name, requester, conn.cursor())
    conn.close()
    if row:
        logger.info(f"🔍 找到已批准记录（requester={requester}）: {row['id']}")
    return row is not None


def consume_approved_query(question, tool_name, requester=None):
    init_db()
    requester = requester or get_requester() or "anonymous"
    conn = get_db()
    cursor = conn.cursor()
    row = _find_approved(question, tool_name, requester, cursor)
    if not row:
        conn.close()
        return False
    cursor.execute("UPDATE approvals SET used = 1 WHERE id = ?", (row["id"],))
    conn.commit()
    conn.close()
    logger.info(f"✅ 已消费复核记录: {row['id']}（requester={requester}）")
    return True


# ============ 裁定（取代 approve / reject 的二值语义） ============

def resolve(approval_id: str, decision: str, rationale: str = "", learn: bool = True) -> bool:
    """给出「裁定 + 依据」。

    这是「人工介入一次 → 沉淀成规则 → 同类自动处理」这个闭环的入口：
    裁定落库后会尝试沉淀成规则（rules.py），下次同类问题不必再问人。

    Args:
        decision: 人的裁定，如「归入『其他相关费用』」
        rationale: 裁定依据，如「财税〔2015〕119 号：差旅费属其他相关费用」
        learn: 是否尝试沉淀成规则（rules.py 未落地时自动跳过）
    """
    init_db()
    conn = get_db()
    row = conn.execute("SELECT status FROM approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row or row["status"] != "pending":
        conn.close()
        return False
    conn.execute(
        "UPDATE approvals SET status = 'approved', decision = ?, rationale = ? WHERE id = ?",
        (decision, rationale, approval_id),
    )
    conn.commit()
    conn.close()

    if learn and decision:
        try:
            import rules  # 规则库（可选依赖）
            rule_id = rules.learn_from_approval(approval_id)
            if rule_id:
                conn = get_db()
                conn.execute("UPDATE approvals SET rule_id = ? WHERE id = ?", (rule_id, approval_id))
                conn.commit()
                conn.close()
                logger.info(f"✅ 裁定已沉淀为规则 [{rule_id}]（下次同类自动处理）")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"规则沉淀失败（裁定已保存，不影响复核）: {e}")

    logger.info(f"✅ 复核完成 [{approval_id}] 裁定={decision!r}")
    return True


def approve(approval_id):
    """保留旧语义：直接批准（不带裁定时使用）。"""
    init_db()
    conn = get_db()
    row = conn.execute("SELECT status FROM approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row or row["status"] != "pending":
        conn.close()
        return False
    conn.execute("UPDATE approvals SET status = 'approved' WHERE id = ?", (approval_id,))
    conn.commit()
    conn.close()
    logger.info(f"✅ 审核通过 [{approval_id}]")
    return True


def reject(approval_id):
    init_db()
    conn = get_db()
    row = conn.execute("SELECT status FROM approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row or row["status"] != "pending":
        conn.close()
        return False
    conn.execute("UPDATE approvals SET status = 'rejected' WHERE id = ?", (approval_id,))
    conn.commit()
    conn.close()
    logger.info(f"❌ 审核拒绝 [{approval_id}]")
    return True


def pending_list():
    """列出待复核记录（供复核终端 / 接口展示）。"""
    init_db()
    conn = get_db()
    rows = conn.execute(
        "SELECT id, kind, requester, reason, tool_input, created_at "
        "FROM approvals WHERE status = 'pending' ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def interactive_approval():
    """命令行复核终端：python hitl.py"""
    init_db()
    print("📋 研发费用合规复核终端")
    print("命令: list | resolve <id> <裁定> | approve <id> | reject <id> | exit")
    while True:
        try:
            cmd = input("\n👉 ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "list":
            rows = pending_list()
            if not rows:
                print("  （没有待复核项）")
            for r in rows:
                print(f"  [{r['id']}] kind={r['kind']} requester={r['requester']}")
                print(f"        原因：{r['reason']}")
                print(f"        内容：{str(r['tool_input'])[:60]}")
        elif cmd.startswith("resolve "):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 3:
                print("  用法：resolve <id> <裁定>")
                continue
            print("  ✅ 已记录裁定" if resolve(parts[1], parts[2]) else "  ❌ 失败（记录不存在或已处理）")
        elif cmd.startswith("approve "):
            print("  ✅ 已批准" if approve(cmd.split()[1]) else "  ❌ 失败")
        elif cmd.startswith("reject "):
            print("  ❌ 已拒绝" if reject(cmd.split()[1]) else "  ❌ 失败")
        elif cmd == "exit":
            break


if __name__ == "__main__":
    interactive_approval()
