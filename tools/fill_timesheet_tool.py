"""智能填报工具：自然语言描述 → 研发工时单（模拟）。

[黑板] 本工具是「工时 → 人员人工分摊 → 费用归集 → 风险扫描」这条业务链的源头，
      所以它必须把工时写进 State 的业务黑板 —— 否则后面的 Agent 看不到它。
"""
from langchain.tools import tool

import context


@tool
def fill_timesheet(employee: str, project: str, work_date: str, hours: float, task: str = "") -> str:
    """生成一张研发工时单。

    Args:
        employee: 研发人员姓名
        project: 研发项目名称
        work_date: 工作日期（YYYY-MM-DD）
        hours: 工时（小时，0.5~24）
        task: 任务描述（可空）
    """
    if not employee or not project or not work_date:
        return "❌ 填报失败：人员、项目、日期为必填项。"
    if hours <= 0 or hours > 24:
        return "❌ 填报失败：工时必须在 0.5~24 小时之间。"
    task_desc = task or "研发活动"

    # ★ 写入业务黑板（旧实现只返回一句字符串，什么都不存）
    context.board_append("timesheets", {
        "employee": employee,
        "project": project,
        "date": work_date,
        "hours": float(hours),
        "task": task_desc,
    })

    return f"✅ 工时单已生成：{employee} 于 {work_date} 在「{project}」填报 {hours}h，任务：{task_desc}。"
