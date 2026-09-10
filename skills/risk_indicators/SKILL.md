---
name: risk_indicators
version: 1.0.0
worker: risk_agent
handler: tools.risk_scan_tool:list_risk_indicators
description: 列出全部金四对标风险指标（人员 / 直接投入 / 折旧 / 其他 / 综合五类）
when_to_use: 用户问「有哪些风险指标」「按什么标准扫」时
enabled: true
---

# 技能：risk_indicators

## 说明
指标口径与知识库「研发费用风险指标库」数据源保持一致。
