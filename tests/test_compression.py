# tests/test_compression.py
"""上下文压缩（compression.py）的测试。

重点验证的是**失败路径与不变量**，而不是「正常能跑」：

  1. 压缩后的 token 数必须真的下降（否则等于没做）；
  2. 最近 N 条消息必须**原样保留**（压掉会让回答前言不搭后语）；
  3. 传入的消息列表**不能被就地修改**（压缩视图与原始记录必须分离）；
  4. 结构化结果瘦身后**关键字段不能丢**（count / id / source 这类骨架信息）；
  5. 超大结果必须**完整落盘**（摘要是不可逆的，原文必须还能查到）；
  6. 内容相同的结果落盘必须**落在同一个文件**（不产生重名副本）；
  7. 未达阈值时必须**零成本返回**（不能白白调摘要模型）；
  8. 摘要模型挂掉时必须**降级而不是崩**。

沿用本项目约定：不依赖 pytest-asyncio，异步用例在同步函数里 asyncio.run()。
"""
import asyncio
import json
import os

import pytest

import compression as C


# ---------- 测试替身 ----------

class FakeLLM:
    """只回固定摘要文本的假模型，用于验证「第二段」是否被调用、调用了几次。"""

    def __init__(self, reply="用户在问加计扣除比例，已答 100%；尚未解决：辅助账保存年限。"):
        self.reply = reply
        self.calls = 0
        self.prompts = []

    async def ainvoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)

        class _R:
            content = self.reply
        return _R()


class BoomLLM:
    """总是抛异常的模型，用于验证摘要失败时的降级路径。"""

    async def ainvoke(self, prompt):
        raise RuntimeError("模拟摘要模型 500")


def _msgs(*pairs):
    return [{"type": t, "content": c} for t, c in pairs]


# ---------- 1. token 估算 ----------

def test_estimate_tokens_basic():
    assert C.estimate_tokens("") == 0
    assert C.estimate_tokens(None) == 0
    assert 2 <= C.estimate_tokens("你好世界") <= 5          # 中文 ~0.7 token/字
    assert 1 <= C.estimate_tokens("hello world") <= 5       # 英文 ~0.25 token/字
    assert C.estimate_tokens("a" * 100) > C.estimate_tokens("a" * 10)


def test_estimate_messages_tokens_counts_overhead():
    one = C.estimate_messages_tokens(_msgs(("human", "hi")))
    many = C.estimate_messages_tokens(_msgs(("human", "hi"), ("ai", "hi")))
    assert many > one


# ---------- 2. head / middle / tail ----------

def test_head_middle_tail_reduces_and_keeps_ends():
    text = "".join(str(i % 10) for i in range(3000))
    out = C.head_middle_tail(text, 300)
    assert len(out) < len(text)
    assert "省略" in out
    assert out.startswith(text[:50])
    assert out.rstrip().endswith(text[-50:])   # 结尾必须保留：结论常在尾部


def test_head_middle_tail_noop_when_under_budget():
    assert C.head_middle_tail("短文本", 1000) == "短文本"


# ---------- 3. 结构化瘦身 ----------

def test_compact_structured_keeps_skeleton():
    big = {
        "count": 42,
        "source": "研发费用政策库",
        "items": [{"id": i, "name": "SKU-%04d" % i, "detail": "x" * 500} for i in range(60)],
    }
    raw = json.dumps(big, ensure_ascii=False)
    slim = C.compact_structured(raw, 800)
    assert slim is not None
    assert len(slim) < len(raw)
    assert '"count": 42' in slim
    assert "_truncated" in slim     # 长列表被标记截断
    assert "_total" in slim         # 但总数必须保留，否则模型以为只有 20 条


def test_compact_structured_returns_none_for_non_json():
    assert C.compact_structured("这不是 JSON", 100) is None
    assert C.compact_structured("{坏掉的 json", 100) is None


# ---------- 4. 工具结果压缩 ----------

def test_small_tool_result_untouched():
    out, path = C.compact_tool_result("简短结果", "some_tool")
    assert out == "简短结果"
    assert path == ""


def test_huge_tool_result_offloaded_with_hash(tmp_path):
    huge = "政策条款内容。" * 3000
    out, path = C.compact_tool_result(huge, "query_my_documents", offload_dir=str(tmp_path))
    assert path and os.path.exists(path)
    assert open(path, encoding="utf-8").read() == huge   # 完整内容确实落盘
    assert "路径：" in out
    assert "sha256：" in out
    assert len(out) < len(huge)


def test_same_content_same_file(tmp_path):
    """同内容重复落盘必须落在同一个文件，否则会攒出一堆重复副本。"""
    huge = "重复内容。" * 3000
    _, p1 = C.compact_tool_result(huge, "t", offload_dir=str(tmp_path))
    _, p2 = C.compact_tool_result(huge, "t", offload_dir=str(tmp_path))
    assert p1 == p2
    assert len(os.listdir(tmp_path)) == 1


# ---------- 5. 消息级压缩 ----------

def test_compact_messages_preserves_recent():
    msgs = _msgs(
        ("human", "很早的问题"),
        ("tool", "超长工具结果" * 2000),
        ("human", "最近一句"),
    )
    out, rep = C.compact_messages(msgs, keep=1)
    assert len(out) == len(msgs)
    assert rep.compacted >= 1
    assert rep.final_tokens < rep.trigger_tokens
    assert C._msg_text(out[-1]) == "最近一句"    # 最近一条原样保留


def test_compact_messages_does_not_mutate_input():
    """不变量：传入的列表与其中的消息都不能被就地修改。"""
    original = _msgs(("tool", "超长" * 5000), ("human", "末尾"))
    snapshot = [dict(m) for m in original]
    C.compact_messages(original, keep=1)
    assert original == snapshot


# ---------- 6. 路由视图 ----------

def test_route_view_window_truncates():
    msgs = _msgs(*[("human", "第%d轮" % i) for i in range(30)])
    out, rep = C.route_view(msgs, window=5)
    assert len(out) == 5
    assert C._msg_text(out[-1]) == "第29轮"      # 保留最近
    assert rep.level == "deterministic"
    assert "窗口截断" in rep.reason


def test_route_view_compacts_large_result_inside_window():
    """窗口内的大工具结果仍然必须被压缩。

    回归测试：曾经写成 compact_messages(msgs, keep=window)，
    等于把窗口内每条消息都划进保护名单 —— 一个几十 K 的工具结果会原样
    送进路由请求，白花 token。这个 bug 在「只看节省比例」时很容易漏掉。
    """
    msgs = _msgs(
        ("human", "早期问题"),
        ("tool", "超大工具结果" * 4000),      # 落在窗口内（最新一条之前）
        ("human", "现在的问题"),
    )
    out, rep = C.route_view(msgs, window=10)   # window 大于条数，走不到截断分支
    assert rep.compacted >= 1, "窗口内的大结果没有被压缩"
    assert rep.final_tokens < rep.trigger_tokens / 2, "节省不明显，可能又压不动了"
    assert C._msg_text(out[-1]) == "现在的问题"   # 最新一条仍原样保留


def test_route_view_noop_when_short():
    msgs = _msgs(("human", "只有一个"))
    out, rep = C.route_view(msgs, window=12)
    assert len(out) == 1
    assert rep.level == "none"


# ---------- 7. 检索上下文拼装 ----------

def test_fit_context_compacts_oversized_single_doc():
    class D:
        def __init__(self, t):
            self.page_content = t

    docs = [D("段落A" * 2000), D("段落B" * 50)]
    ctx, rep = C.fit_context(docs, 1200)
    assert len(ctx) <= 1400      # 省略标记会略超预算
    assert rep.compacted == 1    # 只有超长那篇被压
    assert "段落B" in ctx        # 短的那篇仍然完整


def test_fit_context_empty():
    ctx, _ = C.fit_context([], 500)
    assert ctx == "无相关上下文"


# ---------- 8. 两段式主入口 ----------

def test_maybe_compress_below_threshold_costs_nothing():
    """未达阈值必须零成本返回，不能白白调摘要模型。"""
    llm = FakeLLM()
    msgs = _msgs(("human", "很短的问题"))
    out, rep = asyncio.run(C.maybe_compress(llm, msgs, budget=100000))
    assert rep.level == "none"
    assert llm.calls == 0
    assert out == msgs


def test_maybe_compress_deterministic_is_enough(tmp_path, monkeypatch):
    """确定性压缩就能降到阈值以下时，不应该调摘要模型 —— 这是两段式的核心收益。"""
    monkeypatch.setattr(C, "DEFAULT_OFFLOAD_DIR", str(tmp_path))
    llm = FakeLLM()
    msgs = _msgs(
        ("human", "问题"),
        ("tool", "超大工具结果" * 3000),
        ("human", "最近"),
    )
    out, rep = asyncio.run(C.maybe_compress(llm, msgs, budget=5000, keep=1))
    assert rep.level == "deterministic"
    assert llm.calls == 0
    assert rep.final_tokens < 5000


def test_maybe_compress_summarizes_when_still_over(monkeypatch):
    """确定性压缩后仍超阈值 → 才进入第二段摘要。"""
    monkeypatch.setattr(C, "tool_result_limit", lambda: 10 ** 9)   # 强制「压不动」
    monkeypatch.setattr(C, "keep_recent", lambda: 2)
    llm = FakeLLM()
    msgs = _msgs(*[("human", "第%d轮说了很多话" % i * 200) for i in range(10)])
    out, rep = asyncio.run(C.maybe_compress(llm, msgs, budget=100))
    assert llm.calls == 1
    assert rep.level == "summarized"
    assert rep.summarized == 8          # 10 - keep(2)
    assert rep.history_file             # 原文必须落盘留底
    assert "较早对话摘要" in C._msg_text(out[0])


def test_summarize_failure_degrades_gracefully(monkeypatch):
    """摘要模型挂了不能把主流程带崩 —— 降级为确定性压缩结果。"""
    monkeypatch.setattr(C, "tool_result_limit", lambda: 10 ** 9)
    monkeypatch.setattr(C, "keep_recent", lambda: 2)
    msgs = _msgs(*[("human", "第%d轮" % i * 300) for i in range(10)])
    out, rep = asyncio.run(C.maybe_compress(BoomLLM(), msgs, budget=100))
    # 关键不变量：**不能声称自己做了摘要**
    assert rep.level != "summarized"
    assert rep.summarized == 0
    assert "摘要失败" in rep.reason
    assert len(out) > 0


def test_maybe_compress_disabled(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: False)
    msgs = _msgs(("human", "问题" * 5000))
    out, rep = asyncio.run(C.maybe_compress(FakeLLM(), msgs, budget=10))
    assert rep.enabled is False
    assert out == msgs


# ---------- 9. 报告 ----------

def test_report_saved_ratio():
    r = C.CompressionReport(trigger_tokens=1000, final_tokens=250)
    d = r.to_dict()
    assert d["saved_tokens"] == 750
    assert d["saved_ratio"] == 0.75


def test_stats_shape():
    s = C.stats()
    for k in ("enabled", "budget_tokens", "tool_result_token_limit",
              "preview_tokens", "keep_recent_messages", "route_context_window"):
        assert k in s
