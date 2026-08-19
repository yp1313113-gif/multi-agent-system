# tests/test_tools.py
"""
工具层单元测试。

运行：
  pip install pytest
  pytest -q                       # 默认跳过需要模型/向量库的用例
  MODEL_TESTS=1 pytest -q        # 包含混合检索（需先 python ingest.py 且有模型权重）

设计：重模型用例（rag_tool 导入会加载 embedding/reranker）通过环境变量门控，
      保证无 GPU / 无模型环境也能跑通核心逻辑测试（AST 安全、数据源列表）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.math_tool import calculate
from tools.list_sources_tool import list_data_sources
from tools.data_clean import clean_text
from tools.doc_parser import parse_chapters, split_sections, chapter_to_source
from config import config

# 切块用轻量依赖；若环境未安装则跳过（不阻塞核心逻辑测试）
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAVE_SPLITTER = True
except Exception:  # pragma: no cover
    _HAVE_SPLITTER = False


# ---------------------------------------------------------------------------
# 1. 计算工具 AST 安全防护（不依赖任何模型）
# ---------------------------------------------------------------------------
def test_calculate_safe_basic():
    assert "4" in calculate.invoke("2 ** 2")


def test_calculate_safe_parentheses():
    # (1+2)*3 = 9
    assert "9" in calculate.invoke("(1 + 2) * 3")


def test_calculate_rejects_code_injection():
    # 代码注入表达式必须被拦截，绝不可执行系统命令
    out = calculate.invoke("__import__('os').system('echo hacked')")
    assert "❌" in out
    assert "不支持" in out


def test_calculate_rejects_dunder_access():
    out = calculate.invoke("open('/etc/passwd').read()")
    assert "❌" in out


# ---------------------------------------------------------------------------
# 2. 数据源列表（验证多源逻辑隔离已落地：应列出 3 个真实源）
# ---------------------------------------------------------------------------
def test_list_data_sources_has_three_sources():
    out = list_data_sources.invoke({})
    for name in config.DATA_SOURCES.keys():
        assert name in out, f"数据源列表缺少: {name}"
    # 默认应至少包含 3 个源
    assert len(config.DATA_SOURCES) >= 3


# ---------------------------------------------------------------------------
# 3. 数据管线：清洗 → 切章/小节 → 多源映射 → 切块元数据继承
#    （轻量、无需模型/向量库，CI 可直接跑）
# ---------------------------------------------------------------------------
MANUAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "公司行政管理手册.txt",
)

# 章号 -> 期望逻辑源
_EXPECT_SRC = {"通用制度": [1, 6, 7, 8], "考勤与假期": [2, 3], "薪酬与福利": [4, 5]}


def test_clean_preserves_headings():
    raw = open(MANUAL_PATH, encoding="utf-8").read()
    cleaned = clean_text(raw)
    # 清洗不应破坏章节/小节标题结构
    assert len(parse_chapters(cleaned)) == 8
    assert len(split_sections(cleaned)) >= 8  # 至少含小节


def test_chapter_to_source_mapping():
    raw = open(MANUAL_PATH, encoding="utf-8").read()
    cleaned = clean_text(raw)
    got = {src: [] for src in _EXPECT_SRC}
    for num, _ in parse_chapters(cleaned):
        got[chapter_to_source(num)].append(num)
    for src, chs in _EXPECT_SRC.items():
        assert sorted(got[src]) == chs, f"{src} 章号映射错误: {got[src]}"


def test_chunk_metadata_inheritance():
    if not _HAVE_SPLITTER:
        import pytest
        pytest.skip("未安装 langchain-text-splitters")
    raw = open(MANUAL_PATH, encoding="utf-8").read()
    cleaned = clean_text(raw)
    docs = []
    for num, body in parse_chapters(cleaned):
        src = chapter_to_source(num)
        for sec_title, sec_body in split_sections(body):
            meta = {"source": src, "chapter": num}
            if sec_title:
                meta["section"] = sec_title
            docs.append(Document(page_content=sec_body, metadata=meta))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 0
    # 每个切片都必须继承 source + chapter，且 source 与章号一致（多源隔离）
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        assert c.metadata["source"] in _EXPECT_SRC
        assert c.metadata["chapter"] in _EXPECT_SRC[c.metadata["source"]]
    # 三个源都应出现
    assert {c.metadata["source"] for c in chunks} == set(_EXPECT_SRC)


# ---------------------------------------------------------------------------
# 4. 混合检索：按 source 过滤（需向量库 + 模型，默认跳过）
# ---------------------------------------------------------------------------
MODEL_TESTS_ENABLED = os.getenv("MODEL_TESTS") == "1"


def _require_rag_env():
    return os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db"
    ))


def test_hybrid_source_filter():
    """指定 source 时，召回的切片 metadata.source 必须全部命中该源。"""
    if not _require_rag_env():
        import pytest
        pytest.skip("需先 `python ingest.py` 构建向量库")
    if not MODEL_TESTS_ENABLED:
        import pytest
        pytest.skip("设置 MODEL_TESTS=1 以运行模型相关用例")

    from tools.rag_tool import get_retriever
    retriever = get_retriever()
    docs = retriever.hybrid_search("年假有几天", source="考勤与假期", top_k=3)
    assert len(docs) > 0
    for d in docs:
        assert d.metadata.get("source") == "考勤与假期"

    # 切换到另一个源应得到不同（不重叠）的上下文
    docs2 = retriever.hybrid_search("工资什么时候发", source="薪酬与福利", top_k=3)
    assert len(docs2) > 0
    for d in docs2:
        assert d.metadata.get("source") == "薪酬与福利"
