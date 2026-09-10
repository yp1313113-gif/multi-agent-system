---
name: list_sources
version: 1.0.0
worker: policy_agent
handler: tools.list_sources_tool:list_data_sources
description: 列出当前可用的知识库 / 数据源及其说明
when_to_use: 用户问「你有哪些知识库」「你能查什么」时
enabled: true
---

# 技能：list_sources

## 说明
数据源是**逻辑隔离**的：共享同一个 Chroma 集合，靠 metadata[`source`] 区分，
检索时用 `where={"source": x}` 过滤。新增知识库只需在 config.DATA_SOURCES 增加一项。
