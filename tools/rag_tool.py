# tools/rag_tool.py
import os
import re
import uuid
from functools import lru_cache

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from loguru import logger
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

# 中文分词：优先 jieba，未安装时回退到字符级（中文字 + 英文词）
try:
    import jieba
    jieba.setLogLevel(20)  # 关闭 jieba 的日志噪音
    def _tokenize(text: str):
        return [t for t in jieba.lcut(text) if t.strip()]
except ImportError:
    import re
    def _tokenize(text: str):
        return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())

from config import config
from hitl import has_approved_query, consume_approved_query, request_approval
import context
from cache import cache

_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_current_dir)

def find_chroma():
    candidates = [
        os.path.join(PROJECT_ROOT, "chroma_db"),
        os.path.join(_current_dir, "chroma_db"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    raise FileNotFoundError("找不到 chroma_db，请先运行 python ingest.py")

def find_model():
    candidates = [
        os.path.join(PROJECT_ROOT, "models", "bge-reranker-v2-m3"),
        os.path.join(_current_dir, "models", "bge-reranker-v2-m3"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "config.json")):
                return path
    return None  # 缺失时返回 None，HybridRetriever 自动降级为无重排模式

# embedding 延迟加载：导入模块时不强制加载模型，便于测试与优雅降级
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def get_llm():
    """延迟构造 LLM：避免导入模块时强制要求 API Key（便于测试 / 无凭证环境导入）。"""
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY or "sk-placeholder",
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=config.TEMPERATURE,
    )

rag_prompt = ChatPromptTemplate.from_template("""
你是一个研发费用政策问答助手。根据以下政策知识库回答问题。

要求：
- 只基于提供的上下文回答
- 如果上下文没有相关信息，直接说"知识库未收录该政策，建议咨询专业税务顾问，具体以主管税务机关为准"
- 回答要简洁、准确，引用来源
- 涉及数字（比例、金额、期限）时严格按上下文，不得编造

<上下文>
{context}
</上下文>

用户问题：{input}
""")

def format_docs(docs, max_chars: int = 4000):
    """拼装检索上下文，做 token 预算截断（避免超长上下文撑爆窗口）。"""
    if not docs:
        return "无相关上下文"
    parts = []
    total = 0
    for doc in docs:
        content = doc.page_content
        if total + len(content) > max_chars:
            remain = max_chars - total
            if remain > 0:
                parts.append(content[:remain])
            break
        parts.append(content)
        total += len(content)
    return "\n\n".join(parts)

_rag_chain = None

def get_rag_chain():
    """延迟构造 RAG 生成链（首次调用时构建，需 API Key）。"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = (
            {
                "context": lambda x: format_docs(x["docs"]),
                "input": lambda x: x["question"]
            }
            | rag_prompt
            | get_llm()
            | StrOutputParser()
        )
    return _rag_chain

# ============================================================
# 混合检索：BM25（稀疏） + 向量（稠密） + Reranker（重排）
# ============================================================
class HybridRetriever:
    """BM25 稀疏检索 + 向量稠密检索 + Reranker 重排的融合检索器，支持按 source 切换知识库。

    检索流程：
      1. 确定候选池：若指定 source，则只在该源的切片中检索（逻辑隔离）；
      2. BM25 对候选池做关键词检索，得到一组排名；
      3. 向量检索（bge-small-zh 嵌入，同样带 source 过滤）得到另一组排名；
      4. 用 Reciprocal Rank Fusion（RRF, k=60）融合两路排名；
      5. 用 bge-reranker-v2-m3 交叉编码器对融合结果重排，输出最终 top_k。
         （若 reranker 模型缺失，自动降级为「融合后直接截断」，保证系统仍可运行）
    """

    def __init__(self, chroma_path: str, model_path: str = None,
                 embedding_function=None, reranker=None,
                 top_k: int = 5, candidate_k: int = 20):
        self.embeddings = embedding_function or get_embeddings()
        self.vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings,
            collection_name=config.COLLECTION_NAME,
        )
        self.collection = self.vectorstore._collection

        # 加载全量文档 + 元数据（按 id 排序保证稳定顺序）
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data["ids"]
        docs = data["documents"]
        metas = data["metadatas"] or [{} for _ in ids]
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        self.ids = [ids[i] for i in order]
        self.corpus = [docs[i] for i in order]
        self.sources = [(metas[i] or {}).get("source", config.DEFAULT_DATA_SOURCE) for i in order]

        # 全量 BM25 + 每个 source 一份 BM25（小语料，构建成本低）
        self._bm25_all = BM25Okapi([_tokenize(d) for d in self.corpus])
        self._bm25_by_source = {}
        for s in sorted(set(self.sources)):
            idx = [i for i, x in enumerate(self.sources) if x == s]
            self._bm25_by_source[s] = (idx, BM25Okapi([_tokenize(self.corpus[i]) for i in idx]))

        # 重排模型（缺失则降级为无重排）
        if reranker is not None:
            self.reranker = reranker
        elif model_path:
            try:
                from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder
                cross_encoder = HuggingFaceCrossEncoder(
                    model_name=model_path,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                )
                self.reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
            except Exception as e:
                logger.warning(f"⚠️ reranker 模型加载失败，已降级为无重排模式: {e}")
                self.reranker = None
        else:
            self.reranker = None
        self.candidate_k = candidate_k

    def hybrid_search(self, query: str, source: str | None = None, top_k: int | None = None) -> list:
        top_k = top_k or (self.reranker.top_n if self.reranker else 5)
        top_k = max(1, min(top_k, len(self.corpus)))

        # 候选池（按 source 隔离）
        if source:
            pool_idx = [i for i, x in enumerate(self.sources) if x == source]
            if not pool_idx:
                logger.warning(f"[{context.get_request_id()}] ⚠️ 未知数据源: {source}")
                return []
        else:
            pool_idx = list(range(len(self.ids)))
        pool_ids = [self.ids[i] for i in pool_idx]

        # 1) BM25 排名（在候选池内）
        if source and source in self._bm25_by_source:
            idx_map, bm25 = self._bm25_by_source[source]
            scores = bm25.get_scores(_tokenize(query))
            local_rank = sorted(range(len(idx_map)), key=lambda j: -scores[j])[: self.candidate_k]
            bm25_rank = [idx_map[j] for j in local_rank]
        else:
            scores = self._bm25_all.get_scores(_tokenize(query))
            bm25_rank = sorted(pool_idx, key=lambda i: -scores[i])[: self.candidate_k]

        # 2) 向量排名（带 source 元数据过滤）
        q_emb = self.embeddings.embed_query(query)
        vec_kwargs = dict(
            query_embeddings=[q_emb],
            n_results=min(self.candidate_k, len(pool_ids)),
        )
        if source:
            vec_kwargs["where"] = {"source": source}
        vec_res = self.collection.query(**vec_kwargs)
        vec_ids = vec_res["ids"][0]

        # 3) RRF 融合
        rrf = {cid: 0.0 for cid in pool_ids}
        for rank, i in enumerate(bm25_rank):
            rrf[self.ids[i]] += 1.0 / (60 + rank + 1)
        for rank, cid in enumerate(vec_ids):
            rrf[cid] += 1.0 / (60 + rank + 1)
        fused_ids = sorted(pool_ids, key=lambda cid: -rrf[cid])[: self.candidate_k]

        # 4) 重排（或降级截断）
        from langchain_core.documents import Document
        fused_docs = [
            Document(
                page_content=self.corpus[self.ids.index(cid)],
                metadata={"id": cid, "source": self.sources[self.ids.index(cid)]},
            )
            for cid in fused_ids
        ]
        if self.reranker is None:
            return fused_docs[:top_k]
        return self.reranker.compress_documents(fused_docs, query)


_retriever_cache: dict = {}


def get_retriever():
    """获取共享的混合检索器（所有 source 共用同一集合，按 source 过滤，避免重复建索引）"""
    if "shared" not in _retriever_cache:
        _retriever_cache["shared"] = HybridRetriever(find_chroma(), find_model())
    return _retriever_cache["shared"]


def get_retriever_for_source(source_name=None):
    """兼容旧接口：返回共享检索器 + 该 source 的描述。"""
    if source_name is None:
        source_name = config.DEFAULT_DATA_SOURCE
    desc = config.DATA_SOURCES.get(source_name, {}).get("description", source_name)
    return get_retriever(), desc

# 轻量停用词（相关性门控用，过滤虚词/口语词）
_STOPWORDS = {"的", "了", "是", "吗", "呢", "啊", "呀", "什么", "怎么", "哪些", "如何", "为什么", "你们", "我们", "请", "帮", "一下", "请问", "给我", "有", "没有"}


def _is_relevant(question: str, docs, min_ratio: float = 0.4) -> bool:
    """grounding 相关性门控：问题核心词在检索片段中的覆盖率低于阈值 → 判为不相关。

    轻量离线版（不依赖 LLM，可测试）；生产版可替换为 LLM 相关性判定。
    """
    if not docs:
        return False
    tokens = [tk for tk in _tokenize(question) if tk not in _STOPWORDS and len(tk) >= 2]
    if not tokens:
        return True
    blob = " ".join(d.page_content for d in docs)
    hit = sum(1 for tk in tokens if tk in blob)
    return hit / len(tokens) >= min_ratio


def _rewrite_query(question: str):
    """CRAG 检索改写：去掉口语前缀/填充词，提取核心检索词（离线规则版，不依赖 LLM）。"""
    q = question
    for prefix in ["请问", "帮我", "我想知道", "麻烦问一下", "问一下", "一下", "有没有", "关于", "说一下", "讲讲"]:
        q = q.replace(prefix, "")
    q = q.strip().strip("，。？?！!、")
    return q if q and q != question else None


def _execute_rag_query(question, retriever, source=None):
    try:
        docs = retriever.hybrid_search(question, source=source)
        # CRAG：检索质量自检——结果过少/为空时，改写查询重搜一次（纠正性检索）
        if len(docs) < 2:
            rewritten = _rewrite_query(question)
            if rewritten:
                logger.info(f"[{context.get_request_id()}] 🔄 CRAG 检索质量低，改写查询重搜: {question!r} -> {rewritten!r}")
                docs2 = retriever.hybrid_search(rewritten, source=source)
                if len(docs2) > len(docs):
                    docs = docs2
        # grounding 相关性门控：检索片段与问题不相关 → 拒答（不生成幻觉回答）
        if not docs or not _is_relevant(question, docs):
            logger.info(f"[{context.get_request_id()}] 🚫 grounding 门控：检索结果与问题不相关，拒答")
            return "未找到相关信息：知识库未收录该内容。如需政策解读，请咨询专业税务顾问，具体以主管税务机关为准。"
        logger.info(f"[{context.get_request_id()}] 混合检索召回 {len(docs)} 个文档片段（source={source}）")
        answer = get_rag_chain().invoke({"docs": docs, "question": question})
        sources = []
        for i, doc in enumerate(docs[:2], 1):
            src_label = doc.metadata.get("source", "未知")
            excerpt = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            sources.append(f"[来源{i}·{src_label}] {excerpt}")
        if sources:
            return answer + "\n\n---\n📖 **引用来源**：\n" + "\n".join(sources)
        return answer
    except Exception as e:
        logger.error(f"RAG 执行失败: {e}")
        return f"❌ 查询失败: {e}"

def _cacheable(result: str) -> bool:
    """缓存准入：只有「带引用来源的真实回答」才允许缓存。

    防缓存污染：拒答 / 无来源 / 异常 / 门控拒绝的回答绝不入缓存，
    避免错误答案被反复命中（企业级缓存卫生）。
    """
    if not result:
        return False
    if result.startswith("❌") or result.startswith("未找到"):
        return False
    return "📖" in result or "引用来源" in result


def _cached_rag_query(question, source):
    if config.CACHE_ENABLED:
        cache_key = f"{source}:{question}"
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"[{context.get_request_id()}] ✅ 缓存命中")
            return cached
        logger.info(f"[{context.get_request_id()}] 📦 缓存未命中")
        retriever = get_retriever()
        result = _execute_rag_query(question, retriever, source)
        if _cacheable(result):
            cache.set(cache_key, result)
            logger.info(f"[{context.get_request_id()}] ✅ 回答带引用来源，已写入缓存（准入通过）")
        else:
            logger.info(f"[{context.get_request_id()}] ⏸ 拒答/无来源回答不缓存（缓存准入拦截）")
        return result
    else:
        retriever = get_retriever()
        return _execute_rag_query(question, retriever, source)

print("✅ rag_tool 模块正在被加载...")

@tool
def query_my_documents(question: str, source: str = None) -> str:
    """
    查询公司行政管理政策、制度、流程。
    当用户询问关于年假、考勤、报销、出差等政策时使用此工具。
    """
    print("\n" + "=" * 50)
    print("🔥 query_my_documents 被调用了！")
    print("=" * 50 + "\n")

    try:
        logger.info(f"[{context.get_request_id()}] 🔍 HITL_ENABLED = {config.HITL_ENABLED}")

        if source is None:
            source = config.DEFAULT_DATA_SOURCE

        # ===== 第一步：HITL 审核（优先） =====
        sensitive_keywords = ["我的", "个人", "工资", "薪资", "薪酬"]
        is_sensitive = any(kw in question for kw in sensitive_keywords)
        logger.info(f"[{context.get_request_id()}] 🔍 is_sensitive = {is_sensitive}")

        if config.HITL_ENABLED and is_sensitive:
            if has_approved_query(question, "rag_search"):
                consume_approved_query(question, "rag_search")
                logger.info(f"[{context.get_request_id()}] ✅ HITL 已批准")
                # 批准后继续执行查询（不 return）
            else:
                approval_id = f"approval_{uuid.uuid4().hex[:8]}"
                request_approval(
                    tool_name="rag_search",
                    tool_input=question,
                    user_message=question,
                    context={"source": "rag_tool", "data_source": source},
                    approval_id=approval_id
                )
                logger.info(f"[{context.get_request_id()}] 🔍 HITL 待审核: {approval_id}")
                # ✅ 关键：这里必须 return，停止执行
                return f"⏳ 此问题涉及个人信息，需要人工审核，审核ID: {approval_id}\n请运行 'python hitl.py' 批准后重新提问。"

        # ===== 第二步：判断是否属于政策类问题 =====
        policy_keywords = [
            "加计扣除", "高企", "高新技术企业", "口径", "费用", "归集", "辅助账",
            "备查", "申报", "研发活动", "研发人员", "研发费用", "委托研发", "摊销", "折旧",
            "直接投入", "人员人工", "无形资产", "样品", "政策", "制度", "规定", "流程", "知识库", "数据源"
        ]
        if not any(kw in question for kw in policy_keywords):
            return "我的知识库主要覆盖研发费用政策（加计扣除、六大费用口径、高企认定、申报流程、辅助账、留存备查资料等），您的问题不在这个范围内。如需了解其他事项，请咨询对应部门。"

        # ===== 第三步：缓存查询 =====
        if config.CACHE_ENABLED:
            try:
                logger.info(f"[{context.get_request_id()}] 📦 尝试从 Redis 读取缓存")
                return _cached_rag_query(question, source)
            except Exception as e:
                logger.warning(f"[{context.get_request_id()}] 缓存查询失败: {e}")

        # ===== 第四步：直接查询 =====
        retriever = get_retriever()
        return _execute_rag_query(question, retriever, source)

    except Exception as e:
        logger.error(f"[{context.get_request_id()}] query_my_documents 未捕获异常: {e}")
        return f"❌ 工具执行出错：{str(e)}"
    # rag_tool.py 最末尾
if __name__ == "__main__":
    print("✅ rag_tool 模块加载成功")
    print(f"✅ query_my_documents 类型: {type(query_my_documents)}")