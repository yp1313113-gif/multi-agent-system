---
name: policy_search
version: 1.2.0
worker: policy_agent
handler: tools.rag_tool:query_my_documents
description: 检索研发费用政策知识库（加计扣除、六大费用口径、高企认定、申报流程、辅助账、留存备查资料），返回带引用来源的答案
when_to_use: 用户询问研发费用加计扣除、费用口径、高企认定条件、申报流程、辅助账、留存备查资料等政策内容时
enabled: true
---

# 技能：policy_search

## 行为约定
- 必须基于检索到的上下文作答，**不得编造**；知识库没有的明确回答"未收录"。
- 答案必须带「📖 引用来源」区块，这是**缓存准入**的前提（见 tools/rag_tool.py 的 `_cacheable`）。

## 内部链路（供面试讲解）
CRAG 查询改写 → BM25 + 向量混合召回 → RRF 融合 → bge-reranker-v2-m3 重排 → grounding 相关性门控 → 生成 + 引用溯源。
