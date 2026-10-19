# -*- coding: utf-8 -*-
"""生成看板状态：跑一遍业务链，把结果落到 JSON，供 /api/state 读取。

为什么落 JSON 而不是放内存：
contextvars 的业务黑板是「请求级」的，跨 HTTP 请求读不到，
所以看板需要一份可跨请求读取的快照。
"""
import io, json, os, re, sqlite3, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import context, hitl, rules
from tools.fill_timesheet_tool import fill_timesheet
from tools.expense_tool import classify_expense, _allocate_by_timesheet

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "看板状态.json")
TAX_RATE = 0.25
DEDUCT_RATE = 1.00

# 演示用的原始费用（看板首次生成用）
SEED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "演示种子数据.json")
USER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "用户输入数据.json")


def load_user_input():
    """读取看板上用户自己填的工时和费用（可能不存在）。"""
    try:
        with io.open(USER_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return list(d.get("timesheets") or []), list(d.get("expenses") or [])
    except Exception:
        return [], []


def _load_seed():
    """每次都从 JSON 读种子数据。

    为什么不写成模块级常量：模块只在进程启动时加载一次，
    改了常量不重启就读不到，看板上点「重新跑一遍」会一直是旧数据。
    """
    with io.open(SEED_FILE, encoding="utf-8") as f:
        d = json.load(f)
    # 带上所属项目：税务按项目归集，没有项目维度辅助账就出不来
    exps = [(e["category"], e["amount"], e["description"], e.get("confidence", 0.95),
             e.get("project", ""), e.get("date", ""), e.get("voucher", ""))
            for e in d.get("expenses", [])]
    tss = [(t["employee"], t["project"], t["date"], t["hours"], t.get("task", ""))
           for t in d.get("timesheets", [])]

    # 追加用户在看板上自己填的（放前面，所以它们在时间线上更像"刚填的"）
    u_ts, u_ex = load_user_input()
    for t in u_ts:
        tss.append((t.get("employee", "?"), t.get("project", "?"),
                    t.get("date", ""), float(t.get("hours", 0) or 0), t.get("task", "")))
    for e in u_ex:
        exps.append((e.get("category", "其他相关费用"), float(e.get("amount", 0) or 0),
                     e.get("description", ""), float(e.get("confidence", 1.0) or 1.0),
                     e.get("project", ""), e.get("date", ""), e.get("voucher", "")))
    return exps, tss, float(d.get("tax_rate", 0.25)), float(d.get("deduct_rate", 1.0)), \
           float(d.get("total_labor_cost", 50000.0))

def alloc_by_seed(total=50000.0):
    """从种子工时直接算项目分摊，不依赖 context（跨进程也能用）。"""
    hrs = {}
    for emp, proj, d, h, task in _load_seed()[1]:
        hrs[proj] = hrs.get(proj, 0.0) + float(h)
    tot = sum(hrs.values())
    if tot <= 0:
        return {}
    return {p: round(total * h / tot, 2) for p, h in hrs.items()}


def q(sql, args=(), db=None):
    p = db or hitl.DB_PATH
    if not os.path.exists(p):
        return []
    c = sqlite3.connect(p); c.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in c.execute(sql, args)]
    except sqlite3.Error:
        return []
    finally:
        c.close()

def clean():
    hitl.init_db(); rules.init_db()
    for p in {hitl.DB_PATH, rules.DB_PATH}:
        if not os.path.exists(p):
            continue
        c = sqlite3.connect(p)
        for t in ("approvals", "classification_rules", "classification_log"):
            try: c.execute("DELETE FROM " + t)
            except sqlite3.Error: pass
        c.commit(); c.close()

_LAST = {}

def build():
    """跑一遍业务链并返回状态字典。不写库以外的副作用。"""
    # ★ 生成快照时关掉 bridge，否则「读用户数据 → 工具写用户数据」
    #   会自我繁殖：每生成一次快照，工时表就多一份。
    import bridge
    bridge.set_silent(True)
    tok = context.new_board()
    try:
        context.set_requester("dashboard")
        ts = []
        for emp, proj, d, h, task in _load_seed()[1]:
            fill_timesheet.invoke(dict(employee=emp, project=proj, work_date=d, hours=h, task=task))
            ts.append(dict(employee=emp, project=proj, date=d, hours=h, task=task))

        alloc = _allocate_by_timesheet(50000.0)

        exp = []
        for cat, amt, desc, conf, proj, _dt, _vc in _load_seed()[0]:
            ret = classify_expense.invoke(dict(category=cat, amount=amt, description=desc, confidence=conf))
            m = re.search(r"(approval_[0-9a-f]+)", ret)
            # 从工具返回里抽出「建议归入 X」，给看板的下拉框做默认值
            sm = re.search(r"建议归入「([^」]+)」", ret)
            suggested = sm.group(1) if sm else cat
            exp.append(dict(category=cat, amount=amt, description=desc, project=proj,
                            date=_dt, voucher=_vc,
                            suggested=suggested,
                            pending=bool(m), approval_id=(m.group(1) if m else None)))

        # 顺手把辅助账也算出来（看板要展示）。
        # 这里算是「同一份费用数据，两种视图」：一处汇总，一处是税务要的辅助账格式。
        board = context.board_draft()
        import ledger as _ledger
        _board_exp = board.get("expenses") or []
        # 所属期从费用日期里推断（取出现最多的年月），这样辅助账才有期间标注。
        # 真实系统应该让人选期间；这里是为了演示能呈现出"月度辅助账"。
        # 所属期：从工时记录的日期里推（取出现最多的年月）。
        # 真实系统应该让人显式选期间，这里是演示，能从数据里推就够。
        _months = {}
        for _t in ts:
            _m = str(_t.get("date") or "")[:7]
            if len(_m) == 7 and _m[4] == "-":
                _months[_m] = _months.get(_m, 0) + 1
        _period = max(_months, key=_months.get) if _months else ""
        try:
            _lg = _ledger.build_ledger(_board_exp, alloc, _period)
            _sm = _ledger.build_summary(_lg)
            _ck = _ledger.build_evidence_checklist(_lg, ts)
            _ledger_data = dict(
                period=_period,
                by_category=_lg.get("by_category", {}),
                total=_lg.get("total", 0),
                summary=_sm,
                checklist=_ck,
                skipped=[{"description": x.get("description", ""),
                          "amount": x.get("amount", 0)} for x in (_lg.get("skipped_pending") or [])],
            )
        except Exception as _e:
            _ledger_data = {"error": str(_e)[:120]}
        _LAST.update(dict(timesheets=ts, allocations=alloc, expenses=exp,
                          ledger=_ledger_data,
                          board={k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in board.items()}))
        return _LAST
    finally:
        context.reset_board(tok)
        bridge.set_silent(False)

def snapshot(regenerate=True):
    # 只有明确要求重跑时才清库重建。
    # 裁定后刷新（regenerate=False）绝不能重建 —— 否则会把刚落的裁定冲掉。
    if regenerate:
        clean()
        build()

    # 待复核：以 approvals 表的真实状态为准
    pend = q("SELECT id, tool_input, reason, context FROM approvals WHERE status='pending' ORDER BY rowid")
    approved = {r["id"] for r in q("SELECT id FROM approvals WHERE status='approved'")}
    pending = []
    for r in pend:
        try:
            ctx = json.loads(r["context"] or "{}")
        except Exception:
            ctx = {}
        # 建议类别：从原始种子里按说明匹配
        # _load_seed()[0] 是 [(类别, 金额, 说明, 置信度), ...] 的元组列表
        seed_map = {e[2]: e[0] for e in _load_seed()[0]}
        pending.append(dict(id=r["id"], description=r["tool_input"],
                            amount=ctx.get("amount"), reason=r["reason"],
                            suggested=seed_map.get(r["tool_input"], "其他相关费用")))

    # 已归集费用 = 不在待复核里的
    pend_desc = {p["description"] for p in pending}
    by_cat, total = {}, 0.0
    for cat, amt, desc, conf, proj, _dt, _vc in _load_seed()[0]:
        if desc in pend_desc:
            continue
        by_cat[cat] = by_cat.get(cat, 0.0) + amt
        total += amt

    other = by_cat.get("其他相关费用", 0.0)
    base = total - other
    max_other = round(base / 9, 2) if base > 0 else 0.0
    over = round(other - max_other, 2) if other > max_other else 0.0
    dedu = total - max(over, 0.0)

    rl = q("SELECT keyword, category, rationale, source_approval, hit_count FROM classification_rules ORDER BY rowid")
    log = q("SELECT COUNT(*) n, SUM(auto) a FROM classification_log")
    n_log = log[0]["n"] if log else 0
    n_auto = (log[0]["a"] or 0) if log else 0

    chain = [
        dict(name="工时填报", done=True, detail="%d 条" % len(_load_seed()[1])),
        dict(name="人工分摊", done=True, detail="%d 个项目" % len(alloc_by_seed())),
        dict(name="费用归集", done=True, detail="%d 笔" % len(_load_seed()[0])),
        dict(name="人工复核", done=(len(pending) == 0), detail=("待处理 %d 条" % len(pending)) if pending else "已清空"),
        dict(name="辅助账", done=(total > 0), detail="%d 类" % len(by_cat)),
        dict(name="加计扣除", done=(total > 0), detail="少缴 ¥%s" % format(round(dedu * DEDUCT_RATE * TAX_RATE), ",")),
    ]

    return dict(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        chain=chain,
        timesheets=[dict(employee=e, project=p, date=d, hours=h, task=t)
                    for e, p, d, h, t in _load_seed()[1]],
        allocations=alloc_by_seed(),
        by_category={k: v for k, v in sorted(by_cat.items(), key=lambda x: -x[1])},
        summary=dict(total=round(total, 2), other=round(other, 2),
                     other_ratio=round(other / total, 4) if total else 0,
                     base=round(base, 2), max_other=max_other,
                     over_limit=over, compliant=(over <= 0)),
        tax=dict(deduct_base=round(dedu, 2), deduct_rate=DEDUCT_RATE,
                 deduct_amount=round(dedu * DEDUCT_RATE, 2),
                 tax_rate=TAX_RATE,
                 saved=round(dedu * DEDUCT_RATE * TAX_RATE, 2),
                 formula="加计扣除额 = 可加计费用 × 100%；少缴税额 = 加计扣除额 × 25%"),
        pending=pending,
        expenses=_LAST.get("expenses") or [],
        ledger=_LAST.get("ledger") or {},
        rules=[dict(keyword=r["keyword"], category=r["category"], rationale=r["rationale"],
                    source=str(r["source_approval"])[:20], hit_count=r["hit_count"]) for r in rl],
        log=dict(total=n_log, auto=n_auto, rate=(round(100.0 * n_auto / n_log, 1) if n_log else 0)),
    )

def write(state):
    with io.open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    # 只在直接运行时替换 stdout，被 import 时不能动调用方的输出流
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    reset = "--no-reset" not in sys.argv
    st = snapshot(regenerate=reset)
    write(st)
    print("看板状态已生成：", os.path.basename(STATE_PATH))
    print("  费用合计 ¥%s，待复核 %d 条，少缴 ¥%s" % (
        format(st["summary"]["total"], ","), len(st["pending"]), format(st["tax"]["saved"], ",")))
