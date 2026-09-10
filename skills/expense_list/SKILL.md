---
name: expense_list
version: 1.0.0
worker: expense_agent
handler: tools.expense_tool:list_rd_expenses
description: 列出当前 ERP 中的研发费用条目及其归集口径
when_to_use: 用户要求「列出研发费用条目」「看看现在有哪些费用」时
enabled: true
---

# 技能：expense_list

## 说明
只返回真实条目，不做金额推算。
