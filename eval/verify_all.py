# eval/verify_all.py
"""端到端验证：长期记忆 + 成本埋点 + 真流式，一次跑通。

验证点：
  1. 第 1 轮说出「身份/偏好」→ 应被抽取并写入长期记忆（准入通过）
  2. 第 2 轮同 session 提问 → Worker 应收到注入的记忆提示块
  3. 全过程 token 应被成本回调自动记账
"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

from agent import stream_chat
import context
import memory_store
import cost_tracker


async def turn(msg, session):
    context.set_request_id(f"verify_{int(time.time())}")
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    async for tok in stream_chat(msg, session=session):
        if ttft is None:
            ttft = time.perf_counter() - t0
        chunks.append(tok)
    total = time.perf_counter() - t0
    return "".join(chunks), (ttft or 0), total


async def main():
    session = "verify_demo_001"
    memory_store.forget(session)          # 从干净状态开始
    print("=" * 60)
    print("【第 1 轮】说出身份 + 偏好（应写入长期记忆）")
    print("=" * 60)
    a1, ttft1, t1 = await turn("我是制造业的财务负责人，以后回答尽量简洁一点。你好", session)
    print(f"回答: {a1[:150]}")
    print(f"TTFT={ttft1:.2f}s 总耗时={t1:.2f}s")

    print("\n" + "=" * 60)
    print("【长期记忆】当前 session 记住了什么")
    print("=" * 60)
    facts = memory_store.recall(session)
    print(facts if facts else "（无 —— 准入判断可能拒绝了，见下方日志）")

    print("\n" + "=" * 60)
    print("【第 2 轮】同 session 追问（记忆应被注入 Worker）")
    print("=" * 60)
    a2, ttft2, t2 = await turn("研发费用加计扣除比例是多少？", session)
    print(f"回答: {a2[:200]}")
    print(f"TTFT={ttft2:.2f}s 总耗时={t2:.2f}s")

    print("\n" + "=" * 60)
    print("【成本埋点】")
    print("=" * 60)
    s = cost_tracker.session_summary(session)
    print(f"LLM 调用次数: {s['calls']}")
    print(f"prompt tokens: {s['prompt_tokens']}  completion tokens: {s['completion_tokens']}")
    print(f"合计 tokens: {s['total_tokens']}  费用: {s['cost']}")
    print("按节点拆分（钱花在哪）:")
    for node, v in s["by_node"].items():
        print(f"   · {node}: {v['calls']} 次, prompt={v['prompt_tokens']}, completion={v['completion_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
