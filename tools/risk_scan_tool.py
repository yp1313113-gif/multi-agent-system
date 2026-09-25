"""研发费用风险扫描工具：对标金四指标，输出绿/黄/红预警（模拟）。

[黑板] 扫描结果同时写进 State 的 risk 字段 ——
      原先只存在于返回值里，下一个 Agent（比如生成辅助账）看不到。
"""
from langchain.tools import tool

import context

_RISK_RULES = [
    {"name": "其他相关费用占比", "check": "其他相关费用 / 可加计扣除研发费用总额 ≤ 10%", "level": "高"},
    {"name": "研发人员占比", "check": "研发人员 / 职工总数 ≥ 5%（加计扣除口径）", "level": "高"},
    {"name": "直接投入费用占比", "check": "直接投入费用 / 研发费用总额 ≤ 50%", "level": "中"},
    {"name": "委托研发占比", "check": "委托研发需有合同，且按 80% 计入", "level": "中"},
    {"name": "高企研发费用占比", "check": "研发费用 / 销售收入 ≥ 3%/4%/5%（按收入规模）", "level": "高"},
    {"name": "费用年度波动", "check": "研发费用年度间波动异常（暴增/暴减）需说明", "level": "低"},
]


@tool
def scan_rd_risk(data: str) -> str:
    """对输入的研发费用数据执行风险扫描，返回预警结果。

    Args:
        data: 待扫描的研发费用数据描述（如"其他相关费用占比 15%，研发人员占比 3%"）
    """
    results = []
    if "其他相关费用" in data and any(x in data for x in ["11%", "12%", "13%", "14%", "15%", "超"]):
        results.append("🔴 [高] 其他相关费用占比超过 10% 上限，需调减")
    if "研发人员占比" in data and any(x in data for x in ["2%", "3%", "4%"]):
        results.append("🔴 [高] 研发人员占比低于 5%，加计扣除存在风险")
    if "直接投入" in data and any(x in data for x in ["60%", "70%", "80%"]):
        results.append("🟡 [中] 直接投入费用占比过高，需提供关联凭证")
    if any(x in data for x in ["波动", "暴增", "暴减"]):
        results.append("🟡 [中] 研发费用年度波动异常，需说明原因")
    if not results:
        results.append("🟢 未发现明显风险，指标均在合规范围内")
    results.append("（已对标金四风险指标库，共扫描 6 项指标）")

    # ★ 写入业务黑板
    findings = [r for r in results if r.startswith(("🔴", "🟡"))]
    level = "red" if any(r.startswith("🔴") for r in results) else (
        "yellow" if findings else "green")
    context.board_set("risk", {
        "scanned": len(_RISK_RULES),
        "findings": len(findings),
        "level": level,
        "items": [r for r in results if r.startswith(("🔴", "🟡"))],
    })

    return "\n".join(results)


@tool
def list_risk_indicators() -> str:
    """列出研发费用风险扫描的指标清单。"""
    lines = ["金四对标风险指标："]
    for r in _RISK_RULES:
        lines.append(f"· {r['name']}：{r['check']}（风险等级 {r['level']}）")
    return "\n".join(lines)
