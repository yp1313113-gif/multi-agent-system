"""研发费用数据归集工具：把费用条目归集到 8 类研发费用口径（模拟 ERP 数据）。

[黑板] 归集结果写进 State；
      其中「人员人工费用」还要按工时占比分摊到各研发项目 ——
      工时数据来自 fill_agent 写在黑板上的 timesheets，
      这就是两个 Agent 之间的真实数据流（旧实现里是断的）。
"""
from langchain.tools import tool

import context
import hitl
import rules

# 8 类口径的单一事实来源在 rules.CATEGORIES（规则库要用它校验裁定是否合法）
_CATEGORIES = rules.CATEGORIES

_MOCK_EXPENSES = [
    {"id": "E001", "desc": "研发工程师张三 11 月工资", "category": "人员人工费用", "amount": 25000},
    {"id": "E002", "desc": "研发服务器 11 月电费", "category": "直接投入费用", "amount": 3200},
    {"id": "E003", "desc": "实验设备折旧", "category": "折旧费用", "amount": 8000},
    {"id": "E004", "desc": "研发软件 license 摊销", "category": "无形资产摊销", "amount": 1500},
    {"id": "E005", "desc": "新产品设计外包费", "category": "新产品设计费等", "amount": 12000},
    {"id": "E006", "desc": "专家技术咨询费", "category": "其他相关费用", "amount": 3000},
]


def _allocate_by_timesheet(total: float) -> dict:
    """按工时占比，把人员人工费用分摊到各研发项目。

    政策口径：人员人工费用按实际工时占比在多个研发项目间分摊。
    """
    timesheets = context.board_draft().get("timesheets") or []
    by_project: dict = {}
    for t in timesheets:
        proj = t.get("project", "未知")
        by_project[proj] = by_project.get(proj, 0.0) + float(t.get("hours", 0) or 0)
    total_hours = sum(by_project.values())
    if total_hours <= 0:
        return {}
    return {p: round(total * h / total_hours, 2) for p, h in by_project.items()}


@tool
def classify_expense(category: str, amount: float, description: str,
                     confidence: float = 1.0) -> str:
    """把一笔费用归类到 8 类研发费用口径之一，返回归集结果。

    归集前会先查规则库：命中则按规则自动归类（不再问人）；
    没命中且触发复核条件时，写进待复核队列。

    Args:
        category: 费用类别（8 类口径之一）
        amount: 金额（元）
        description: 费用说明
        confidence: 本次归类的置信度 0~1；不确定时传低值会触发人工复核
    """
    if amount <= 0:
        return "❌ 金额必须大于 0。"

    # ★ 第一步：先查规则库 —— 命中就按规则归类，不再问人
    #   这就是「人工裁定一次 → 沉淀规则 → 同类自动处理」闭环的落地
    rule = rules.find_rule(description)
    auto = False
    rule_id = None
    if rule:
        category = rule["category"]
        rule_id = rule["id"]
        rules.touch_rule(rule_id)
        auto = True
    elif category not in _CATEGORIES:
        return "❌ 归类失败：类别必须是：" + "、".join(_CATEGORIES)

    # ★ 第二步：没命中规则时，判断是否触发合规复核（业务风险，不是隐私关键词）
    need, reason = (False, "")
    if not auto:
        need, reason = hitl.needs_review(
            "expense_classify", {"amount": amount, "confidence": confidence})

    # ★ 第三步：写业务黑板 + 记一笔归集日志（供自动归类率指标用）
    context.board_append("expenses", {
        "category": category,
        "amount": float(amount),
        "description": description,
        "auto": auto,
    })
    rules.log_classification(description, category, amount, auto=auto, rule_id=rule_id)

    extra = ""
    if category == "人员人工费用":
        alloc = _allocate_by_timesheet(float(amount))
        if alloc:
            context.board_set("allocations", alloc)
            detail = "；".join(f"{k} ¥{v:,.0f}" for k, v in alloc.items())
            extra = f"\n  · 已按工时占比分摊到各项目：{detail}"

    if auto:
        return (f"✅ 自动归集（命中规则 {rule_id}）：{description} → {category}，"
                f"金额 ¥{amount:,.0f}。\n  · 依据：{rule.get('rationale') or '人工历史裁定'}{extra}")

    if need:
        aid = hitl.request_approval(
            tool_name="classify_expense",
            tool_input=description,
            user_message=description,
            context={"amount": float(amount), "suggested": category, "confidence": confidence},
            kind="expense_classify",
            reason=reason,
        )
        context.board_append("pending", {"id": aid, "description": description,
                                         "suggested": category, "reason": reason})
        return (f"⏳ 已暂存待复核：{description} → 建议归入「{category}」，金额 ¥{amount:,.0f}。\n"
                f"  · 复核 ID：{aid}\n  · 原因：{reason}\n"
                f"  · 请运行 'python hitl.py' 给出裁定（裁定后自动沉淀为规则）{extra}")

    return f"✅ 已归集：{description} → {category}，金额 ¥{amount:,.0f}。{extra}"


@tool
def list_rd_expenses() -> str:
    """列出当前模拟 ERP 中的研发费用条目及归集口径。"""
    lines = ["当前 ERP 研发费用条目："]
    for e in _MOCK_EXPENSES:
        lines.append(f"· {e['id']} {e['desc']} → {e['category']} ¥{e['amount']:,}")
    return "\n".join(lines)
