# eval/stream_bench.py
"""流式性能基准：测量首字延迟(TTFT) / 总耗时 / 输出字符数。

用法：
    python eval/stream_bench.py                        # 默认 3 个问题
    python eval/stream_bench.py "问题1" "问题2" ...

对照意义：
    · 第 1 问是「冷启动」（含 bge 向量/重排模型首次加载），偏慢属正常；
    · 第 2 问起是「热态」，才是生产环境（常驻服务）的真实数字。
    · ainvoke + 分块补发（旧实现）：TTFT ≈ 总耗时；
    · astream_events 真流式（新实现）：生成阶段应远快于总耗时。
"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

from agent import stream_chat

DEFAULT_QUESTIONS = [
    "研发费用加计扣除比例是多少？",
    "高企认定需要满足哪些条件？",
    "研发费用归集的辅助账要怎么建？",
]


async def ask(question: str, tag: str):
    session = f"bench_{tag}_{int(time.time())}"
    t0 = time.perf_counter()
    ttft = None
    total_chars = 0
    chunk_count = 0
    chunks = []

    async for tok in stream_chat(question, session=session):
        if ttft is None:
            ttft = time.perf_counter() - t0
        total_chars += len(tok)
        chunk_count += 1
        chunks.append(tok)

    total = time.perf_counter() - t0
    gen = (total - ttft) if ttft else 0.0
    print("=" * 56)
    print(f"[{tag}] {question}")
    print(f"  首字延迟 TTFT : {ttft:.2f}s" if ttft else "  首字延迟 TTFT : N/A")
    print(f"  总耗时        : {total:.2f}s")
    print(f"  生成阶段      : {gen:.2f}s（{chunk_count} 个流式块 / {total_chars} 字符）")
    print(f"  前置占比      : {(ttft / total * 100) if ttft else 0:.0f}% 的时间花在「出第一个字之前」")
    print("-" * 56)
    print("".join(chunks)[:400].replace("\n", " "))
    print("=" * 56)
    return {"tag": tag, "ttft": ttft, "total": total, "chars": total_chars, "chunks": chunk_count}


async def main():
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    results = []
    for i, q in enumerate(questions):
        tag = "冷启动" if i == 0 else f"热态{i}"
        results.append(await ask(q, tag))

    print("\n########## 汇总 ##########")
    for r in results:
        print(f"{r['tag']:6s} TTFT={r['ttft']:.2f}s  总耗时={r['total']:.2f}s  "
              f"块数={r['chunks']}  字符={r['chars']}")


if __name__ == "__main__":
    asyncio.run(main())
