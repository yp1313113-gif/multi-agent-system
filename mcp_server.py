"""研发费用 Agent 的 MCP Server：把核心工具暴露为标准 MCP 协议工具。

外部系统 / 其他 Agent / 客户端可通过 MCP 协议（stdio 传输）调用，
实现"Agent 能力标准化接入"——企业级集成：ERP / 财务软件 / 内部系统
都能通过统一协议调用本 Agent 的归集、扫描、填报能力。

运行：python mcp_server.py （stdio 传输）
"""
from mcp.server.fastmcp import FastMCP

from tools.expense_tool import classify_expense, list_rd_expenses
from tools.risk_scan_tool import scan_rd_risk, list_risk_indicators
from tools.fill_timesheet_tool import fill_timesheet

mcp = FastMCP(
    "rd-expense-agent",
    instructions="研发费用智能管理 Agent 的 MCP 工具集：费用归集 / 风险扫描 / 工时填报 / 费用条目",
)


@mcp.tool()
def classify_expense_tool(category: str, amount: float, description: str) -> str:
    """把一笔费用归类到 8 类研发费用口径之一，返回归集结果。"""
    return classify_expense.invoke({"category": category, "amount": amount, "description": description})


@mcp.tool()
def scan_rd_risk_tool(data: str) -> str:
    """对研发费用数据执行金四对标风险扫描，返回绿/黄/红预警。"""
    return scan_rd_risk.invoke({"data": data})


@mcp.tool()
def fill_timesheet_tool(employee: str, project: str, work_date: str, hours: float, task: str = "") -> str:
    """生成一张研发工时单。"""
    return fill_timesheet.invoke({"employee": employee, "project": project, "work_date": work_date, "hours": hours, "task": task})


@mcp.tool()
def list_rd_expenses_tool() -> str:
    """列出当前 ERP 中的研发费用条目及归集口径。"""
    return list_rd_expenses.invoke({})


if __name__ == "__main__":
    # stdio 传输（MCP 标准），客户端通过 mcp 协议连接
    mcp.run()
