---
name: risk_scan
version: 1.1.0
worker: risk_agent
handler: tools.risk_scan_tool:scan_rd_risk
description: 对研发费用数据执行金四对标风险扫描，输出绿 / 黄 / 红三级预警
when_to_use: 用户要求扫描 / 检查研发费用风险、做税务合规自检时
enabled: true
---

# 技能：risk_scan

## 输出
三级预警 + 命中指标说明。黄色以上建议人工复核（可对接 HITL 审批队列）。
