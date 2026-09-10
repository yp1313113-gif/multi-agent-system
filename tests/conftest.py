# tests/conftest.py
"""
编排层测试的环境隔离：

1. supervisor.py 在模块级 import tools.rag_tool（其依赖 chromadb / redis / 模型权重），
   而本目录的 test_supervisor_agents.py 只验证「编排层」确定性逻辑（路由 / 拒答 / 跨轮），
   并不触发真实 RAG 工具。因此在导入 supervisor 之前，用轻量替身预占 sys.modules，
   避免在 CI / 无模型环境安装 chromadb、redis 与加载 embedding/reranker 模型。

2. AsyncSqliteSaver 会把多轮状态持久化到 checkpoints.db；若上一次运行残留了
   同名 thread_id 的状态，本轮测试会「续接」旧对话导致断言失真。
   因此每个 pytest 会话开始时删除 checkpoints.db，保证从干净状态起跑。

真实 RAG 检索质量由 MODEL_TESTS=1 门控的用例覆盖（见 test_tools.py 的约定）。
"""
import os
import sys
import time
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_stub_rag():
    module = types.ModuleType("tools.rag_tool")

    def query_my_documents(*args, **kwargs):
        # 编排层测试中不会被真正调用（Worker 被 FakeWorker 替换）
        return "（stub）编排层测试不触发真实 RAG 工具"

    module.query_my_documents = query_my_documents
    return module


sys.modules.setdefault("tools.rag_tool", _make_stub_rag())


@pytest.fixture(scope="session", autouse=True)
def _clean_checkpoints_db():
    """会话开始时删除残留的 checkpoints.db，避免跨运行 thread 状态串扰。"""
    for candidate in (os.path.join(ROOT, "checkpoints.db"), os.path.join(os.getcwd(), "checkpoints.db")):
        for _ in range(3):
            try:
                if os.path.exists(candidate):
                    os.remove(candidate)
                break
            except PermissionError:
                time.sleep(0.2)
        else:
            # 静默失败会变成"断言数字莫名其妙变大"的假故障，排查成本极高。
            # 这里必须显式报错：多半是还有进程（如性能基准脚本）占着这个库。
            raise RuntimeError(
                f"无法删除残留的 {candidate}（被占用）。"
                "请先结束占用该文件的 Python 进程（例如 eval/stream_bench.py 的基准运行）再执行测试。"
            )
    yield
