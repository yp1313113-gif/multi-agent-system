# eval/retrieval_eval.py
"""
混合检索效果评估：对比「纯向量检索」与「BM25+向量混合检索」在测试集上的命中率。

用法：
    python eval/retrieval_eval.py

说明：
    - 测试集为手工构造的 (query, 期望命中关键词) 列表，模拟真实员工高频咨询。
    - 命中定义：top_k 召回的文档片段中，至少包含一条期望关键词。
    - 用于复现 README 中「RAG 召回率 72% → 88%」的对比结论（数值随测试集与数据变化）。
"""
import os
import sys

# 把项目根目录加入路径，便于直接运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.rag_tool import CHROMA_PATH, MODEL_PATH, embeddings, HybridRetriever, _tokenize


# ---- 测试集：模拟员工高频政策咨询 ----
TEST_SET = [
    ("年假怎么算？最多可以休几天？", ["年假"]),
    ("出差住宿标准是多少钱一晚？", ["出差", "住宿"]),
    ("报销流程是什么，需要哪些材料？", ["报销"]),
    ("请病假需要提供什么证明？", ["病假"]),
    ("公积金怎么提取？", ["公积金"]),
    ("加班有加班费吗？", ["加班"]),
    ("入职需要准备哪些材料？", ["入职"]),
    ("五险一金包含哪些？", ["五险一金"]),
    ("婚假可以请多少天？", ["婚假"]),
    ("迟到一次扣多少工资？", ["迟到"]),
    ("培训期间有补贴吗？", ["培训", "补贴"]),
    ("产假是多少天？", ["产假"]),
]


def bm25_only_search(query, hybrid: HybridRetriever, k=5):
    """纯 BM25 检索（对照组）"""
    scores = hybrid.bm25.get_scores(_tokenize(query))
    rank = sorted(range(len(hybrid.ids)), key=lambda i: -scores[i])[:k]
    return [hybrid.corpus[i] for i in rank]


def vector_only_search(query, hybrid: HybridRetriever, k=5):
    """纯向量检索（对照组）"""
    q_emb = embeddings.embed_query(query)
    res = hybrid.collection.query(query_embeddings=[q_emb], n_results=k)
    ids = res["ids"][0]
    idx = {cid: i for i, cid in enumerate(hybrid.ids)}
    return [hybrid.corpus[idx[cid]] for cid in ids]


def hit(docs, keywords):
    return any(kw in doc for doc in docs for kw in keywords)


def main():
    print("📊 混合检索效果评估")
    print(f"   测试集规模: {len(TEST_SET)} 条\n")

    hybrid = HybridRetriever(CHROMA_PATH, MODEL_PATH)

    strategies = [
        ("纯 BM25", bm25_only_search),
        ("纯向量", vector_only_search),
        ("混合(BM25+向量+Rerank)", lambda q, h: [d.page_content for d in h.hybrid_search(q)]),
    ]
    for name, fn in strategies:
        hits = sum(1 for q, kw in TEST_SET if hit(fn(q, hybrid), kw))
        rate = hits / len(TEST_SET)
        print(f"  · {name:<28} 命中 {hits}/{len(TEST_SET)}  =  {rate * 100:.1f}%")

    print("\n✅ 评估完成（如需复现 README 数值，请调整 TEST_SET 与 top_k 配置）")


if __name__ == "__main__":
    main()
