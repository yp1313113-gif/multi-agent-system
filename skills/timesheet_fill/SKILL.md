---
name: timesheet_fill
version: 1.0.0
worker: fill_agent
handler: tools.fill_timesheet_tool:fill_timesheet
description: 把自然语言描述转成一张研发工时单（人员 / 项目 / 日期 / 工时 / 任务）
when_to_use: 用户描述「某人某天在某项目干了多少小时」时
enabled: true
---

# 技能：timesheet_fill

## 输入要求
人员、项目、日期、工时四项必填；信息不全时**先向用户确认缺失字段**，不要填默认值。
