# tests/test_warmup.py
"""服务预热测试。

预热本身要加载真实模型（慢），所以这里只测**行为契约**：
  · 开关能关掉预热（CI / 测试环境不该加载 700MB 模型）
  · 任一步失败都不阻断启动（预热是尽力而为，不是强依赖）
"""
import asyncio
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warmup  # noqa: E402


def test_warmup_can_be_disabled(monkeypatch):
    """WARMUP_ENABLED=false 时应直接跳过 —— CI 环境不该为预热付出模型加载时间。"""
    from config import config
    monkeypatch.setattr(config, "WARMUP_ENABLED", False, raising=False)
    report = asyncio.run(warmup.warmup_all())
    assert report.get("skipped") is True


def test_retriever_failure_does_not_raise(monkeypatch):
    """检索链路预热失败必须只记警告并返回，不能把服务启动带崩。"""
    import tools.rag_tool as rag
    def boom():
        raise RuntimeError("模型文件缺失")
    monkeypatch.setattr(rag, "get_retriever", boom, raising=False)

    report = warmup.warmup_retriever()
    assert "error" in report
    assert report["total"] >= 0


def test_graph_failure_does_not_raise(monkeypatch):
    """编排层预热失败同样不能阻断启动。"""
    import agent
    async def boom():
        raise RuntimeError("图构建失败")
    monkeypatch.setattr(agent, "warmup_agent", boom, raising=False)

    report = asyncio.run(warmup.warmup_graph())
    assert "error" in report


def test_warmup_all_disabled_still_returns_report(monkeypatch):
    """关闭预热时返回结构要保持稳定，便于 /health 直接透出。"""
    from config import config
    monkeypatch.setattr(config, "WARMUP_ENABLED", False, raising=False)
    r = asyncio.run(warmup.warmup_all(include_graph=False))
    assert isinstance(r, dict)
