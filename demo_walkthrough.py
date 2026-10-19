# -*- coding: utf-8 -*-
"""走读演示：一条命令跑完「工时填报 → 费用归集 → 人工裁定 → 规则沉淀 → 自动命中 → 风险扫描」。

运行： 双击 演示_双击运行.bat
不需要 API Key，不调模型，直接调项目里真实的业务工具。
"""
import io, os, re, sqlite3, sys, datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import context
import hitl
import rules
from tools.fill_timesheet_tool import fill_timesheet
from tools.expense_tool import classify_expense, _allocate_by_timesheet
from tools.risk_scan_tool import scan_rd_risk

LOG = []

def out(s=""):
    print(s)
    LOG.append(str(s))

def head(t):
    out()
    out("=" * 70)
    out(t)
    out("=" * 70)

# ---------- 演示前清库，保证每次结果一致 ----------
def clean_db():
    hitl.init_db(); rules.init_db()
    for p in {hitl.DB_PATH, rules.DB_PATH}:
        if not os.path.exists(p):
            continue
        conn = sqlite3.connect(p)
        for t in ("approvals", "classification_rules", "classification_log"):
            try:
                conn.execute("DELETE FROM " + t)
            except sqlite3.Error:
                pass
        conn.commit(); conn.close()

def table_rows(db_path, table, limit=10):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM %s ORDER BY rowid DESC LIMIT %d" % (table, limit))]
    except sqlite3.Error:
        return []
    finally:
        conn.close()

clean_db()

tok = context.new_board()
try:
    context.set_requester("demo-walkthrough")

    # ===== 1 工时填报 =====
    head("【第 1 步】工时填报")
    for r in [
        dict(employee="小王", project="A项目", work_date="2026-10-18", hours=6, task="模型参数调优"),
        dict(employee="小王", project="A项目", work_date="2026-10-19", hours=2, task="接口联调"),
        dict(employee="小李", project="A项目", work_date="2026-10-18", hours=2, task="数据标注"),
        dict(employee="小李", project="B项目", work_date="2026-10-19", hours=4, task="检索评测"),
    ]:
        out("  " + fill_timesheet.invoke(r))

    # ===== 2 共享状态 =====
    head("【第 2 步】业务黑板：Agent 之间的数据流")
    ts = context.board_draft().get("timesheets") or []
    for t in ts:
        out("  " + str(t))
    out("  → 共 %d 条。写进黑板，下游 Agent 才读得到" % len(ts))

    # ===== 3 工时占比分摊 =====
    head("【第 3 步】人员人工按工时占比分摊到各项目")
    alloc = _allocate_by_timesheet(50000.0)
    hrs = {}
    for t in ts:
        hrs[t["project"]] = hrs.get(t["project"], 0) + t["hours"]
    out("  各项目工时：" + "，".join("%s %sh" % (k, int(v)) for k, v in hrs.items()))
    for proj, amt in alloc.items():
        out("  %s 分摊 %.2f 元" % (proj, amt))
    out("  → 假设研发人员合计人工成本 50000 元，按项目工时占比分配")

    # ===== 4 费用归集 =====
    head("【第 4 步】费用归集（规则库现在是空的，所以全部按规则判断）")
    out("  规则库当前条数：%d" % len(table_rows(rules.DB_PATH, "classification_rules")))
    out()
    pending_ids = []
    for e in [
        dict(category="其他相关费用", amount=3000,  description="研发人员出差上海差旅费", confidence=0.9),
        dict(category="直接投入费用", amount=12000, description="检测服务费",           confidence=0.9),
        dict(category="其他相关费用", amount=150000, description="大额技术咨询费",        confidence=0.9),
        dict(category="其他相关费用", amount=8000,  description="专家咨询费",           confidence=0.5),
    ]:
        ret = classify_expense.invoke(e)
        out("  " + ret.replace("\n", "\n  "))
        m = re.search(r"(approval_[0-9a-f]+)", ret)
        if m:
            pending_ids.append(m.group(1))

    # ===== 5 人工裁定 + 规则沉淀 =====
    head("【第 5 步】人工裁定两条，并沉淀成规则")
    out("  待复核 %d 条：%s" % (len(pending_ids), "、".join(pending_ids)))
    out()
    verdicts = {
        0: "其他相关费用", 1: "专家咨询费属其他相关费用",
    }
    for i, aid in enumerate(pending_ids):
        decision = "其他相关费用"
        reason = ("财税〔2015〕119 号：技术咨询费计入其他相关费用"
                  if i == 0 else "总局 2017 年 40 号公告：专家咨询费计入其他相关费用")
        ok = hitl.resolve(aid, decision, reason)
        out("  裁定 %s → %s  （%s）" % (aid, decision, "成功" if ok else "失败"))
    out()
    rs = table_rows(rules.DB_PATH, "classification_rules")
    out("  → 规则库现在有 %d 条：" % len(rs))
    for r in rs:
        out("     关键词「%s」→ %s（来源裁定 %s）" % (r.get("keyword"), r.get("category"), str(r.get("source_approval"))[:16]))

    # ===== 6 规则命中，自动归类 =====
    head("【第 6 步】同样两条再归集一次：这次规则命中，不再问人")
    for e in [
        dict(category="其他相关费用", amount=150000, description="大额技术咨询费", confidence=0.9),
        dict(category="其他相关费用", amount=8000,  description="专家咨询费",   confidence=0.5),
    ]:
        ret = classify_expense.invoke(e)
        out("  " + ret.replace("\n", "\n  "))
    out()
    log_rows = table_rows(rules.DB_PATH, "classification_log")
    auto_n = sum(1 for r in log_rows if r.get("auto"))
    out("  → 归集日志 %d 条，其中规则命中自动归类 %d 条" % (len(log_rows), auto_n))
    if log_rows:
        out("  → 自动归类率 %.0f%%" % (100.0 * auto_n / len(log_rows)))

    # ===== 7 风险扫描 =====
    head("【第 7 步】风险扫描（对标金税四期指标）")
    out("  输入「其他相关费用占比 15%」：")
    out("  " + scan_rd_risk.invoke({"data": "其他相关费用占比 15%"}).replace("\n", "\n  "))
    out()
    out("  输入「各项指标正常」：")
    out("  " + scan_rd_risk.invoke({"data": "各项指标正常"}).replace("\n", "\n  "))

    # ===== 8 数据落在哪 =====
    head("【第 8 步】这些数据落在哪")
    board = context.board_draft()
    out("  内存里的业务黑板（进程结束即消失）：")
    for k, v in board.items():
        out("    %-12s %s" % (k, len(v) if isinstance(v, (list, dict)) else v))
    out()
    out("  落盘数据库：" + os.path.basename(hitl.DB_PATH))
    for t, cn in [("approvals", "复核记录"), ("classification_rules", "沉淀出的规则"), ("classification_log", "归集日志")]:
        out("    %-22s %d 行   （%s）" % (t, len(table_rows(hitl.DB_PATH, t, 999)), cn))

finally:
    context.reset_board(tok)

# ---------- 保存输出 ----------
md = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "演示输出_%s.txt" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
with io.open(md, "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))
out()
out("本次输出已保存：" + os.path.basename(md))
