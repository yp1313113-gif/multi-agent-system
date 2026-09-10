---
name: expense_classify
version: 1.1.0
worker: expense_agent
handler: tools.expense_tool:classify_expense
description: 把一笔费用归类到 8 类研发费用口径之一（人员人工/直接投入/折旧/无形资产摊销/新产品设计费/装配调试/其他相关费用/委托研发）
when_to_use: 用户要求把某笔费用归类 / 归集到研发费用口径时
enabled: true
---

# 技能：expense_classify

## 输入要求
类别、金额、说明三项必须齐全；缺项先向用户确认，**不要猜金额**。
