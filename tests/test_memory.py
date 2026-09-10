# tests/test_memory.py
"""长期记忆测试：重点验证「准入判断」——防记忆污染的核心。

不需要 API Key 与模型权重，全部离线可跑。
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import memory_store  # noqa: E402


@pytest.fixture
def mem(tmp_path, monkeypatch):
    """把记忆库指向临时文件，避免污染项目根目录的 memory.db。"""
    monkeypatch.setattr(memory_store, "DB_PATH", str(tmp_path / "memory_test.db"))
    return memory_store


# ---------------------------------------------------------------------------
# 1. 准入判断：什么该记、什么必须丢弃
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,value,expected", [
    # ✅ 合规：偏好 / 身份 / 项目
    ("preference", "喜欢简洁回答", True),
    ("identity", "制造业财务负责人", True),
    ("project", "研发费用管理系统", True),
    # ❌ 非白名单 key 一律拒绝
    ("random", "随便什么", False),
    ("chat", "你好", False),
    # ❌ 问句：用户在提问，不是陈述自己的事实
    ("preference", "加计扣除是多少？", False),
    ("identity", "这个怎么算呢", False),
    # ❌ 时效性表述：是当下状态，不是长期事实
    ("preference", "今天想先看政策", False),
    ("project", "这次要算折旧", False),
    # ❌ 超长：避免把整段对话塞进来
    ("preference", "x" * 61, False),
    # ❌ 空值
    ("preference", "   ", False),
])
def test_is_admissible(mem, key, value, expected):
    assert mem.is_admissible(key, value) is expected


# ---------------------------------------------------------------------------
# 2. 写入 / 读取 / 删除
# ---------------------------------------------------------------------------

def test_remember_only_persists_admissible(mem):
    """write 时自动过滤：只有通过准入的才落库。"""
    accepted = mem.remember("u1", {
        "preference": "喜欢简洁回答",
        "identity": "今天很忙",          # 时效性 → 应被拒
        "garbage": "非白名单",           # key 不合法 → 应被拒
    })
    assert accepted == {"preference": "喜欢简洁回答"}
    assert mem.recall("u1") == {"preference": "喜欢简洁回答"}


def test_recall_isolated_by_session(mem):
    """不同 session 的记忆互相隔离（这是长期记忆的边界）。"""
    mem.remember("u1", {"preference": "喜欢简洁回答"})
    mem.remember("u2", {"identity": "财务负责人"})
    assert mem.recall("u1") == {"preference": "喜欢简洁回答"}
    assert mem.recall("u2") == {"identity": "财务负责人"}
    assert mem.recall("u404") == {}


def test_remember_upsert_overwrites(mem):
    """同一 key 重复写入 → 更新而不是新增。"""
    mem.remember("u1", {"preference": "喜欢简洁回答"})
    mem.remember("u1", {"preference": "喜欢详细回答"})
    assert mem.recall("u1") == {"preference": "喜欢详细回答"}


def test_forget(mem):
    mem.remember("u1", {"preference": "A", "identity": "B"})
    assert mem.forget("u1", "preference") == 1
    assert mem.recall("u1") == {"identity": "B"}
    assert mem.forget("u1") == 1        # 清空该 session 剩余记忆
    assert mem.recall("u1") == {}


def test_empty_inputs_are_safe(mem):
    """空 session / 空 facts 不应崩，也不应写库。"""
    assert mem.remember("", {"preference": "A"}) == {}
    assert mem.remember("u1", {}) == {}
    assert mem.recall("") == {}


# ---------------------------------------------------------------------------
# 3. 提示块拼装（注入 Worker 用）
# ---------------------------------------------------------------------------

def test_build_memory_prompt(mem):
    assert mem.build_memory_prompt("nobody") == ""      # 无记忆 → 空串，不注入
    mem.remember("u1", {"preference": "喜欢简洁回答", "identity": "财务负责人"})
    prompt = mem.build_memory_prompt("u1")
    assert "长期记忆" in prompt
    assert "偏好：喜欢简洁回答" in prompt
    assert "身份：财务负责人" in prompt


# ---------------------------------------------------------------------------
# 4. LLM 输出解析的健壮性（脏输出不能把主流程带崩）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"preference": "喜欢简洁回答"}', {"preference": "喜欢简洁回答"}),
    ('```json\n{"identity": "财务"}\n```', {"identity": "财务"}),
    ('好的，这是结果：{"project": "研发系统"} 以上', {"project": "研发系统"}),
    ("{}", {}),
    ("完全不是 JSON", {}),
    ("", {}),
    ("[1,2,3]", {}),          # 不是对象 → 丢弃
])
def test_parse_facts_json_is_robust(raw, expected):
    assert memory_store.parse_facts_json(raw) == expected
