# hitl.py
import os
import sqlite3
import json
import uuid
from loguru import logger

# 使用基于本文件所在目录的绝对路径，避免不同启动目录导致读写不同的 approval.db
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approval.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS approvals (
        id TEXT PRIMARY KEY,
        tool_name TEXT,
        tool_input TEXT,
        user_message TEXT,
        context TEXT,
        status TEXT DEFAULT 'pending',
        result TEXT,
        used INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def request_approval(tool_name, tool_input, user_message, context, approval_id=None):
    init_db()
    if approval_id is None:
        approval_id = f"approval_{uuid.uuid4().hex[:8]}"
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO approvals (id, tool_name, tool_input, user_message, context, status) VALUES (?, ?, ?, ?, ?, ?)",
        (approval_id, tool_name, tool_input, user_message, json.dumps(context), "pending")
    )
    conn.commit()
    conn.close()
    logger.info(f"📋 请求审核 [{approval_id}]")
    return approval_id


def _find_approved(question, tool_name, cursor):
    """查找一条未消费且已批准的记录。
    匹配策略：
    1. 精确匹配 tool_input（最理想）。
    2. LIKE 模糊匹配 tool_input / user_message。
    3. 关键词交集兜底：LLM 调用工具时会改写用户问题（如"我的工资是多少"改写成"工资如何计算 薪酬制度"），
       此时按敏感/业务关键词交集判断是否为同一意图。
    """
    # 1) 精确匹配
    row = cursor.execute(
        "SELECT id, tool_input FROM approvals WHERE tool_name = ? AND tool_input = ? AND status = 'approved' AND used = 0 ORDER BY created_at LIMIT 1",
        (tool_name, question)
    ).fetchone()
    if row:
        return row

    # 2) LIKE 模糊匹配
    if question:
        like_q = f"%{question}%"
        row = cursor.execute(
            "SELECT id, tool_input FROM approvals WHERE tool_name = ? AND status = 'approved' AND used = 0 AND (tool_input LIKE ? OR user_message LIKE ?) ORDER BY created_at LIMIT 1",
            (tool_name, like_q, like_q)
        ).fetchone()
        if row:
            return row

    # 3) 关键词交集兜底：处理 LLM 改写导致的 tool_input 变化
    if question:
        # 把语义相近的词归为一组；只要 question 和已批准记录在任意同一组里都有词，即视为同意图
        overlap_groups = [
            {"工资", "薪酬", "薪资", "年薪", "月薪", "待遇"},
            {"年假", "假期", "休假", "请假", "病假", "事假", "婚假", "产假", "陪产假", "丧假"},
            {"公积金", "社保", "五险一金"},
            {"考勤", "迟到", "早退", "加班"},
            {"出差", "报销", "补贴", "福利"},
        ]
        q_keywords = {k for k in set().union(*overlap_groups) if k in question}

        if q_keywords:
            rows = cursor.execute(
                "SELECT id, tool_input, user_message FROM approvals WHERE tool_name = ? AND status = 'approved' AND used = 0 ORDER BY created_at DESC",
                (tool_name,)
            ).fetchall()
            for r in rows:
                r_text = f"{r['tool_input']} {r['user_message']}"
                r_keywords = {k for k in set().union(*overlap_groups) if k in r_text}

                # 检查是否在同一组里都有词
                for group in overlap_groups:
                    if (q_keywords & group) and (r_keywords & group):
                        return r
    return None


def has_approved_query(question, tool_name):
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    row = _find_approved(question, tool_name, cursor)
    conn.close()
    if row:
        logger.info(f"🔍 找到已批准记录: {row['id']} (tool_input={row['tool_input']!r})")
    return row is not None


def consume_approved_query(question, tool_name):
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    row = _find_approved(question, tool_name, cursor)
    if not row:
        conn.close()
        return False
    cursor.execute("UPDATE approvals SET used = 1 WHERE id = ?", (row["id"],))
    conn.commit()
    conn.close()
    logger.info(f"✅ 已消费审核记录: {row['id']}")
    return True


def approve(approval_id):
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    row = cursor.execute("SELECT status FROM approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row or row["status"] != "pending":
        conn.close()
        return False
    cursor.execute("UPDATE approvals SET status = 'approved' WHERE id = ?", (approval_id,))
    conn.commit()
    conn.close()
    logger.info(f"✅ 审核通过 [{approval_id}]")
    return True

def reject(approval_id):
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    row = cursor.execute("SELECT status FROM approvals WHERE id = ?", (approval_id,)).fetchone()
    if not row or row["status"] != "pending":
        conn.close()
        return False
    cursor.execute("UPDATE approvals SET status = 'rejected' WHERE id = ?", (approval_id,))
    conn.commit()
    conn.close()
    logger.info(f"❌ 审核拒绝 [{approval_id}]")
    return True

def interactive_approval():
    init_db()
    print("📋 人工审核终端启动")
    print("命令: list, approve <id>, reject <id>, exit")
    while True:
        cmd = input("\n👉 ").strip()
        if cmd == "list":
            conn = get_db()
            rows = conn.execute("SELECT id, tool_input, status FROM approvals WHERE status='pending'").fetchall()
            conn.close()
            for r in rows:
                print(f"  [{r['id']}] {r['tool_input'][:50]}... ({r['status']})")
        elif cmd.startswith("approve "):
            if approve(cmd.split()[1]):
                print("✅ 已批准")
            else:
                print("❌ 失败")
        elif cmd.startswith("reject "):
            if reject(cmd.split()[1]):
                print("❌ 已拒绝")
            else:
                print("❌ 失败")
        elif cmd == "exit":
            break

if __name__ == "__main__":
    interactive_approval()
