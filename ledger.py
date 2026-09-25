# ledger.py
"""研发支出辅助账 + 留存备查资料清单生成。

━━━ 政策背景（这不是硬塞的功能，是享受优惠的硬性前提）━━━
· 财税〔2015〕119 号：企业享受研发费用加计扣除，必须建立「研发支出辅助账」。
· 国家税务总局公告 2015 年第 97 号、2023 年第 7 号：明确研发费用辅助账样式
  与留存备查资料清单。

  也就是说：**没有辅助账和备查资料，加计扣除享受不了；税务检查时也拿不出证据。**

━━━ 为什么值得做成系统能力 ━━━
归集算得再准，最后也要落到「一张能交给税务的表」上。
财务手工做这张表通常要几天，且是纯体力活 —— 这正是系统该接手的部分。

━━━ 产物 ━━━
① 研发支出辅助账（按 8 类口径）
② 辅助账汇总表（含限额校验：其他相关费用 ≤ 10%）
③ 留存备查资料清单（7 项，逐条标注 已具备 / 待补充）
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain.tools import tool

import context
from rules import CATEGORIES

# 加计扣除口径：其他相关费用不得超过可加计扣除研发费用总额的 10%
# （高企认定口径是 20%，两者不同 —— 这里按加计扣除）
OTHER_EXPENSE_LIMIT = 0.10

# 政策要求的留存备查资料（国家税务总局公告 2015 年第 97 号 / 2023 年第 7 号）
EVIDENCE_ITEMS = [
    ("自主、委托、合作研发项目计划书与立项决议文件",
     "企业有权部门关于立项的决议或计划书",
     lambda ctx: bool(ctx.get("projects"))),
    ("研发机构或项目组的编制情况及研发人员名单",
     "研发人员名单 + 编制情况说明",
     lambda ctx: bool(ctx.get("timesheets"))),
    ("经科技行政主管部门登记的委托、合作研发项目合同",
     "涉及委托/合作研发时才需要",
     lambda ctx: bool(ctx.get("has_entrusted"))),
    ("研发人员与仪器设备、无形资产的费用分配说明（含工时记录）",
     "人员人工费用按工时占比分摊的说明",
     lambda ctx: bool(ctx.get("allocations"))),
    ("集中研发项目的决算表、分摊明细与收益分享比例",
     "涉及集中研发（集团）时才需要",
     lambda ctx: False),
    ("「研发支出」辅助账及汇总表",
     "本系统自动生成",
     lambda ctx: True),
    ("地市级（含）以上科技行政主管部门出具的鉴定意见",
     "如已取得则一并留存",
     lambda ctx: False),
]


def build_ledger(expenses: list, allocations: dict | None = None,
                 period: str = "") -> dict:
    """生成研发支出辅助账。

    Args:
        expenses: [{category, amount, description, auto}, ...]（来自业务黑板）
        allocations: 人员人工费用按工时占比分摊到各项目的结果
        period: 所属期，如 2026-09
    """
    allocations = allocations or {}
    by_category: dict[str, dict[str, Any]] = {c: {"amount": 0.0, "count": 0, "items": []}
                                              for c in CATEGORIES}
    unknown: list = []

    for e in expenses or []:
        cat = e.get("category", "")
        amt = float(e.get("amount", 0) or 0)
        if cat in by_category:
            by_category[cat]["amount"] += amt
            by_category[cat]["count"] += 1
            by_category[cat]["items"].append({
                "description": e.get("description", ""),
                "amount": amt,
                "auto": bool(e.get("auto")),
            })
        else:
            unknown.append(e)

    total = round(sum(v["amount"] for v in by_category.values()), 2)
    return {
        "period": period or datetime.now().strftime("%Y-%m"),
        "by_category": {k: {"amount": round(v["amount"], 2),
                            "count": v["count"], "items": v["items"]}
                        for k, v in by_category.items()},
        "total": total,
        "allocations": allocations,
        "unknown_category": unknown,
    }


def build_summary(ledger: dict) -> dict:
    """辅助账汇总表 + 限额校验。"""
    total = float(ledger.get("total", 0) or 0)
    by_cat = ledger.get("by_category", {})
    other = float(by_cat.get("其他相关费用", {}).get("amount", 0) or 0)
    ratio = (other / total) if total else 0.0

    # 加计扣除时可计入的其他相关费用上限：
    # 其他相关费用 ≤ 可加计扣除研发费用总额的 10%
    # 等价地：其他相关费用 /（总额 − 其他相关费用）≤ 1/9
    base = total - other
    max_other = round(base * (1 / 9), 2) if base > 0 else 0.0
    over = round(other - max_other, 2) if other > max_other else 0.0

    return {
        "total": total,
        "other_expense": other,
        "other_ratio": ratio,
        "limit_ratio": OTHER_EXPENSE_LIMIT,
        "max_other_allowed": max_other,
        "over_limit": over,
        "compliant": over <= 0,
        "allocations": ledger.get("allocations", {}),
        "non_zero_categories": {k: v["amount"] for k, v in by_cat.items() if v["amount"] > 0},
    }


def build_evidence_checklist(ledger: dict, timesheets: list | None = None,
                             projects: list | None = None) -> list[dict]:
    """生成留存备查资料清单，逐条标注 已具备 / 待补充。"""
    expenses = []
    for v in ledger.get("by_category", {}).values():
        expenses.extend(v.get("items", []))

    ctx = {
        "projects": projects or [],
        "timesheets": timesheets or [],
        "allocations": ledger.get("allocations") or {},
        "has_entrusted": any("委托" in (e.get("description") or "") for e in expenses),
    }

    out = []
    for idx, (name, note, ready_fn) in enumerate(EVIDENCE_ITEMS, 1):
        try:
            ready = bool(ready_fn(ctx))
        except Exception:
            ready = False
        out.append({
            "no": idx,
            "name": name,
            "note": note,
            "status": "已具备" if ready else "待补充",
        })
    return out


def render_markdown(ledger: dict, summary: dict, checklist: list) -> str:
    """渲染成可直接交给财务 / 附在申报材料后的一页纸。"""
    lines = [
        f"# 研发支出辅助账（所属期：{ledger.get('period', '')}）",
        "",
        "## 一、按费用类别归集",
        "",
        "| 费用类别 | 笔数 | 金额（元） |",
        "|---|---:|---:|",
    ]
    for cat in CATEGORIES:
        v = ledger["by_category"].get(cat, {"count": 0, "amount": 0.0})
        if v["count"] or v["amount"]:
            lines.append(f"| {cat} | {v['count']} | {v['amount']:,.2f} |")
    conclusion = ("✅ 其他相关费用未超限"
                  if summary["compliant"]
                  else f"🔴 其他相关费用超限 ¥{summary['over_limit']:,.2f}，需调减")
    lines += [
        f"| **合计** |  | **{summary['total']:,.2f}** |",
        "",
        "## 二、限额校验（加计扣除口径）",
        "",
        f"- 其他相关费用：¥{summary['other_expense']:,.2f}"
        f"（占比 {summary['other_ratio']:.1%}，上限 10%）",
        f"- 可计入上限：¥{summary['max_other_allowed']:,.2f}",
        f"- 结论：{conclusion}",
        "",
    ]
    if summary.get("allocations"):
        lines += ["## 三、人员人工费用按工时占比分摊", "",
                  "| 研发项目 | 分摊金额（元） |", "|---|---:|"]
        for proj, amt in summary["allocations"].items():
            lines.append(f"| {proj} | {float(amt):,.2f} |")
        lines.append("")

    lines += ["## 四、留存备查资料清单", "", "| # | 资料 | 状态 |", "|---:|---|---|"]
    for c in checklist:
        mark = "✅" if c["status"] == "已具备" else "⬜"
        lines.append(f"| {c['no']} | {c['name']} | {mark} {c['status']} |")
    lines.append("")
    return "\n".join(lines)


# ============ Agent 工具入口 ============

@tool
def generate_rd_ledger(period: str = "") -> str:
    """生成研发支出辅助账与留存备查资料清单。

    基于本次会话已归集的费用，输出：按 8 类口径的辅助账、
    限额校验（其他相关费用 ≤ 10%）、人员人工分摊表，
    以及政策要求的 7 项留存备查资料清单。

    Args:
        period: 所属期（如 2026-09）；不填用当前月份
    """
    draft = context.board_draft()
    expenses = draft.get("expenses") or []
    if not expenses:
        return ("⚠️ 本次会话还没有归集任何费用，无法生成辅助账。\n"
                "请先告诉我要归集的费用（类别、金额、说明）。")

    allocations = draft.get("allocations") or {}
    projects = draft.get("projects") or []
    timesheets = draft.get("timesheets") or []

    lg = build_ledger(expenses, allocations, period)
    sm = build_summary(lg)
    ck = build_evidence_checklist(lg, timesheets, projects)

    # ★ 写回业务黑板：辅助账汇总 + 备查清单（后续问答/交付都能读到）
    context.board_set("ledger", sm)
    context.board_set("evidence", ck)

    return render_markdown(lg, sm, ck)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    demo_expenses = [
        {"category": "人员人工费用", "amount": 32000, "description": "9月研发人员工资", "auto": False},
        {"category": "直接投入费用", "amount": 8200, "description": "研发服务器电费/材料", "auto": True},
        {"category": "折旧费用", "amount": 8000, "description": "实验设备折旧", "auto": False},
        {"category": "无形资产摊销", "amount": 1500, "description": "研发软件license摊销", "auto": False},
        {"category": "其他相关费用", "amount": 3200, "description": "研发部门差旅费", "auto": True},
        {"category": "其他相关费用", "amount": 2800, "description": "研发部门差旅费", "auto": True},
    ]
    alloc = {"智能座舱": 21333.33, "智能语音": 10666.67}
    ts = [{"employee": "张三", "project": "智能座舱", "hours": 8}]
    projs = [{"name": "智能座舱"}, {"name": "智能语音"}]

    lg = build_ledger(demo_expenses, alloc, "2026-09")
    sm = build_summary(lg)
    ck = build_evidence_checklist(lg, ts, projs)
    print(render_markdown(lg, sm, ck))
