# warmup.py
"""服务预热：把「只发生一次的重活」从首个用户请求挪到服务启动阶段。

━━━ 为什么必须做 ━━━
实测冷启动 TTFT（首字延迟）26.6s、热态 4.9s —— 中间差的 20 秒全是**一次性成本**：

  1. bge-small-zh 向量模型加载 + 首次前向（算子选择、线程池初始化、内存分配）
  2. bge-reranker-v2-m3 交叉编码器加载（393 个权重张量）
  3. Chroma 集合加载 + 全量语料读入 + BM25 索引构建
  4. LangGraph Supervisor-Worker 图构建 + checkpointer(SQLite) 连接

这些成本本身免不掉，但**不该由第一个用户来承担**（"第一个用户为所有人买单"）。
在服务启动时预热，就把这笔开销从「用户可感知的延迟」变成了「部署时的一次冷启动」。

━━━ 设计原则 ━━━
预热是**尽力而为**的：任何一步失败都只告警、不阻断启动。
预热失败最坏结果是退回原来的冷启动行为，而不是服务起不来。
"""
import time

from loguru import logger


def warmup_retriever() -> dict:
    """预热检索链路：加载模型 → 跑一次真实检索（触发前向计算）。"""
    t0 = time.perf_counter()
    detail = {}
    try:
        from tools.rag_tool import get_retriever
        t1 = time.perf_counter()
        retriever = get_retriever()          # 加载 embedding + Chroma + BM25 + reranker
        detail["load_retriever"] = round(time.perf_counter() - t1, 2)

        # 关键：光加载模型不够。第一次推理还要做算子选择/内存分配，
        # 必须跑一次真实检索把这段也吃掉，否则首个用户请求依然慢。
        t2 = time.perf_counter()
        try:
            retriever.hybrid_search("研发费用加计扣除比例", source=None, top_k=3)
            detail["first_search"] = round(time.perf_counter() - t2, 2)
        except Exception as e:
            logger.warning(f"[warmup] 检索预热查询失败（不影响启动）: {e}")
            detail["first_search"] = -1
    except Exception as e:
        logger.warning(f"[warmup] 检索链路预热失败（不影响启动，首个请求会变慢）: {e}")
        detail["error"] = str(e)[:120]
    detail["total"] = round(time.perf_counter() - t0, 2)
    return detail


async def warmup_graph() -> dict:
    """预热编排层：构建 Supervisor-Worker 图 + 打开 checkpointer。"""
    t0 = time.perf_counter()
    detail = {}
    try:
        from agent import warmup_agent
        await warmup_agent()
        detail["build_graph"] = round(time.perf_counter() - t0, 2)
    except Exception as e:
        logger.warning(f"[warmup] 编排层预热失败（不影响启动）: {e}")
        detail["error"] = str(e)[:120]
    detail["total"] = round(time.perf_counter() - t0, 2)
    return detail


async def warmup_all(include_graph: bool = True) -> dict:
    """执行全部预热，返回各阶段耗时（供日志/接口展示）。"""
    try:
        from config import config
        if not getattr(config, "WARMUP_ENABLED", True):
            logger.info("[warmup] 已通过 WARMUP_ENABLED=false 关闭预热，跳过")
            return {"skipped": True}
    except Exception:
        pass

    logger.info("[warmup] 开始预热（检索链路 + 编排层）...")
    t0 = time.perf_counter()
    report = {"retriever": warmup_retriever()}
    if include_graph:
        report["graph"] = await warmup_graph()
    report["total"] = round(time.perf_counter() - t0, 2)
    logger.info(
        f"[warmup] 预热完成，总耗时 {report['total']}s "
        f"（检索 {report['retriever'].get('total')}s"
        + (f"，编排 {report['graph'].get('total')}s" if include_graph else "")
        + "）—— 首个用户请求不再承担这段冷启动成本"
    )
    return report
