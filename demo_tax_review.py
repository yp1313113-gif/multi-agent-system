# -*- coding: utf-8 -*-
"""演示：加计扣除测算 + 人工审核闭环。

回答两个问题：
  1. 这笔研发费用能少缴多少税
  2. 遇到问题怎么转人工，转完怎么沉淀成规则

运行： 双击 演示_加计扣除与人工审核.bat
"""
import io, os, re, sqlite3, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import context, hitl, rules
from tools.expense_tool import classify_expense

# 企业适用所得税税率（一般 25%，高新技术企业 15%，可改）
TAX_RATE = 0.25
# 加计扣除比例（现行 100%）
DEDUCT_RATE = 1.00

def out(s=""): print(s)

def head(t):
    out(); out("=" * 72); out(t); out("=" * 72)

def clean_db():
    hitl.init_db(); rules.init_db()
    for p in {hitl.DB_PATH, rules.DB_PATH}:
        if not os.path.exists(p): continue
        c = sqlite3.connect(p)
        for t in ("approvals", "classification_rules", "classification_log"):
            try: c.execute("DELETE FROM " + t)
            except sqlite3.Error: pass
        c.commit(); c.close()

def q(sql, args=(), db=None):
    p = db or hitl.DB_PATH
    if not os.path.exists(p): return []
    c = sqlite3.connect(p); c.row_factory = sqlite3.Row
    try: return [dict(r) for r in c.execute(sql, args)]
    except sqlite3.Error: return []
    finally: c.close()

clean_db()
tok = context.new_board()
try:
    context.set_requester("demo-tax-review")

    # ===== 第 1 步 归集费用 =====
    head("【第 1 步】归集本期研发费用")
    items = [
        ("人员人工费用", 32000, "9 月研发人员工资",       0.95),
        ("直接投入费用",  8600, "检测服务费",             0.95),
        ("折旧费用",      5000, "实验设备折旧",           0.95),
        ("其他相关费用",  3000, "研发人员出差差旅费",      0.95),
        ("其他相关费用", 150000, "大额技术咨询费",         0.95),
        ("其他相关费用",  8000, "专家咨询费",             0.5),
    ]
    pend = []
    for cat, amt, desc, conf in items:
        r = classify_expense.invoke(dict(category=cat, amount=amt, description=desc, confidence=conf))
        tag = "转人工" if "待复核" in r else "已归集"
        out("  [%s] %-14s ¥%-8s %s" % (tag, cat, amt, desc))
        m = re.search(r"(approval_[0-9a-f]+)", r)
        if m: pend.append(m.group(1))

    # ===== 第 2 步 待复核队列 =====
    head("【第 2 步】人工审核队列（问题在这里被拦下）")
    rows = q("SELECT id, tool_input, reason, status FROM approvals WHERE status='pending' ORDER BY rowid")
    out("  待复核 %d 条：" % len(rows))
    for r in rows:
        out("    %s  「%s」" % (r["id"], r["tool_input"]))
        out("      原因：%s" % r["reason"])
    out()
    out("  → 这两条为什么要拦：一条金额 15 万超了 10 万阈值，一条把握度只有 50%")
    out("  → 拦下来不是失败，是把「会出错的判断」交回给人")

    # ===== 第 3 步 加计扣除测算 =====
    head("【第 3 步】加计扣除测算：到底少缴多少税")
    # 只统计「已归集」的费用。待复核的那两条不算进合计 ——
    # 这也是一个真实的边界问题：classify_expense 在转人工之前就写进黑板了，
    # 所以这里要按 pending 的说明把待复核的剔除。
    pending_desc = {r["tool_input"] for r in q("SELECT tool_input FROM approvals WHERE status='pending'")}
    by_cat = {}
    expenses = context.board_draft().get("expenses") or []
    for e in expenses:
        if e.get("description") in pending_desc:
            continue
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + float(e["amount"])
    total = sum(by_cat.values())
    other = by_cat.get("其他相关费用", 0)
    base = total - other
    max_other = round(base / 9, 2) if base > 0 else 0.0
    over = round(other - max_other, 2) if other > max_other else 0.0
    dedu_total = total - max(over, 0.0)

    out("  按六类口径汇总（已归集部分）：")
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        out("    %-16s ¥%s" % (k, format(v, ",.0f")))
    out("    %-16s ¥%s" % ("合计", format(total, ",.0f")))
    out()
    out("  限额校验（加计扣除口径）：")
    out("    其他相关费用 ¥%s，占总额 %.1f%%" % (format(other, ",.0f"), 100*other/total if total else 0))
    out("    上限 = 前五项 ¥%s ÷ 9 = ¥%s" % (format(base, ",.0f"), format(max_other, ",.0f")))
    if over > 0:
        out("    [超限] 超出 ¥%s，超出部分不能加计，需剔除" % format(over, ",.0f"))
    else:
        out("    [合规] 未超限，可全额加计")
    out()
    out("  加计扣除测算：")
    out("    可加计扣除的研发费用     ¥%s" % format(dedu_total, ",.0f"))
    out("    加计扣除比例             %.0f%%" % (DEDUCT_RATE * 100))
    out("    加计扣除额               ¥%s" % format(dedu_total * DEDUCT_RATE, ",.0f"))
    out("    企业适用所得税税率       %.0f%%" % (TAX_RATE * 100))
    out("    ==> 少缴企业所得税       ¥%s" % format(dedu_total * DEDUCT_RATE * TAX_RATE, ",.0f"))
    out()
    out("  → 也就是说：这笔研发投入，国家帮你承担了 %.0f%% 的税" % (DEDUCT_RATE * TAX_RATE * 100))
    out()
    out("  【对比】假如其他相关费用是 ¥6,000 而不是 ¥%s：" % format(other, ",.0f"))
    other2 = 6000.0
    over2 = round(other2 - max_other, 2) if other2 > max_other else 0.0
    dedu2 = total - max(over2, 0.0)
    out("    上限仍是 ¥%s（前五项没变）" % format(max_other, ",.0f"))
    out("    超出 ¥%s，剔除后加计基数变成 ¥%s" % (format(over2, ",.0f"), format(dedu2, ",.0f")))
    out("    少缴税额从 ¥%s 变成 ¥%s" % (format(dedu_total*DEDUCT_RATE*TAX_RATE, ",.0f"), format(dedu2*DEDUCT_RATE*TAX_RATE, ",.0f")))
    out("    → 这就是那 10% 上限的实际后果：超出的部分不能加计，要调减")

    # ===== 第 4 步 人工裁定 =====
    head("【第 4 步】人工给出裁定，并沉淀成规则")
    for i, aid in enumerate(pend):
        reason = ("财税〔2015〕119 号：技术咨询费计入其他相关费用" if i == 0
                  else "总局 2017 年 40 号公告：专家咨询费计入其他相关费用")
        ok = hitl.resolve(aid, "其他相关费用", reason)
        out("  裁定 %s → 其他相关费用（%s）" % (aid, "成功" if ok else "失败"))

    # ===== 第 5 步 规则库 =====
    head("【第 5 步】规则库：这一次的裁定变成了以后不用再问的依据")
    rs = q("SELECT keyword, category, rationale, hit_count, source_approval FROM classification_rules ORDER BY rowid")
    out("  规则库 %d 条：" % len(rs))
    for r in rs:
        out("    关键词「%s」→ %s" % (r["keyword"], r["category"]))
        out("      依据：%s" % (r["rationale"] or "-"))
        out("      来源裁定：%s  已命中 %s 次" % (str(r["source_approval"])[:20], r["hit_count"]))
    out()
    out("  待复核队列现在是 %d 条" % len(q("SELECT id FROM approvals WHERE status='pending'")))

    # ===== 第 6 步 再归集一次 =====
    head("【第 6 步】同样两条再进来：这次不再问人")
    for cat, amt, desc, conf in [("其他相关费用", 150000, "大额技术咨询费", 0.95),
                                  ("其他相关费用", 8000, "专家咨询费", 0.5)]:
        r = classify_expense.invoke(dict(category=cat, amount=amt, description=desc, confidence=conf))
        out("  " + r.replace("\n", "\n  "))
    out()
    logs = q("SELECT COUNT(*) n, SUM(auto) a FROM classification_log")
    n = logs[0]["n"] if logs else 0
    a = logs[0]["a"] or 0 if logs else 0
    out("  → 归集日志 %d 条，规则命中自动 %s 条，自动归类率 %.0f%%" % (n, a, 100.0*a/n if n else 0))

finally:
    context.reset_board(tok)
