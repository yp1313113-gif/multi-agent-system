# tests/test_skills.py
"""技能库测试：验证「声明式目录 → 可装配能力」这条链路。

注意：本模块依赖 tests/conftest.py 把 tools.rag_tool 预占为轻量替身，
因此不会加载 chromadb / 模型权重，可离线快速运行。
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skills import registry  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 目录扫描与元数据解析
# ---------------------------------------------------------------------------

def test_load_skills_scans_all_packs():
    skills = registry.load_skills(refresh=True)
    assert len(skills) >= 7
    for name in ("policy_search", "expense_classify", "risk_scan", "timesheet_fill"):
        assert name in skills


def test_every_skill_has_required_metadata():
    """每条技能都必须有 description / version / worker / handler —— 缺一不可装配。"""
    for s in registry.load_skills(refresh=True).values():
        assert s.description, f"{s.name} 缺 description"
        assert s.worker, f"{s.name} 缺 worker"
        assert ":" in s.handler, f"{s.name} 的 handler 格式应为 '模块:属性'"
        assert s.version


def test_skill_exposes_capability_manifest():
    """技能库能自描述「系统当前具备哪些能力」——供 MCP / 文档 / 路由提示词复用。"""
    manifest = registry.list_skills()
    assert len(manifest) >= 7
    one = next(m for m in manifest if m["name"] == "policy_search")
    assert one["worker"] == "policy_agent"
    assert one["enabled"] is True


# ---------------------------------------------------------------------------
# 2. 动态装配
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("worker", [
    "policy_agent",
    "expense_agent",
    "risk_agent",
    "fill_agent",
])
def test_tools_for_worker(worker):
    """每个 Worker 装配到的工具数，应等于元数据里声明归属它的技能数。

    这里刻意不写死数字（原来写的是 2/2/2/1）——
    技能库的设计初衷就是「新增能力 = 新增一个目录，主流程零改动」，
    测试如果把数量写死，每加一个技能都要来改测试，等于把这个设计毁掉。
    改为「跟技能目录对齐」，新增技能时这个测试自动跟着走。
    """
    declared = [s for s in registry.load_skills().values()
                if s.enabled and s.worker == worker]
    assert len(declared) >= 1, f"{worker} 一个技能都没有"
    assert len(registry.tools_for_worker(worker)) == len(declared)


def test_tools_for_unknown_worker_is_empty():
    assert registry.tools_for_worker("no_such_worker") == []


def test_assembled_tools_are_guarded():
    """装配出来的工具必须过 Harness（保持 name 可路由，且可被调用）。"""
    pairs = registry.tools_for_worker("fill_agent")
    skill, tool = pairs[0]
    assert skill.name == "timesheet_fill"
    assert getattr(tool, "name", None) or getattr(tool, "__name__", None)


# ---------------------------------------------------------------------------
# 3. front-matter 解析的健壮性
# ---------------------------------------------------------------------------

def test_parse_front_matter_basic():
    meta, body = registry._parse_front_matter(
        "---\nname: demo\nversion: 1.0.0\n---\n\n正文内容"
    )
    assert meta == {"name": "demo", "version": "1.0.0"}
    assert body.strip() == "正文内容"


def test_parse_front_matter_without_block():
    meta, body = registry._parse_front_matter("没有 front matter 的纯文本")
    assert meta == {}
    assert body == "没有 front matter 的纯文本"


def test_parse_front_matter_ignores_comments_and_junk():
    meta, _ = registry._parse_front_matter(
        "---\n# 注释\nname: demo\n这一行没有冒号\nversion: 2.0.0\n---\n"
    )
    assert meta == {"name": "demo", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# 4. 容错：坏技能包不能让服务起不来
# ---------------------------------------------------------------------------

def test_broken_skill_is_skipped_not_fatal(tmp_path, monkeypatch):
    """缺 description 的技能包应被跳过 + 告警，其余技能照常可用。"""
    good = tmp_path / "good_skill"
    good.mkdir()
    (good / "SKILL.md").write_text(
        "---\nname: good_skill\nversion: 1.0.0\nworker: fill_agent\n"
        "handler: tools.fill_timesheet_tool:fill_timesheet\ndescription: 正常技能\n---\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad_skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: bad_skill\nversion: 1.0.0\n---\n", encoding="utf-8")

    monkeypatch.setattr(registry, "SKILLS_DIR", str(tmp_path))
    skills = registry.load_skills(refresh=True)

    assert "good_skill" in skills
    assert "bad_skill" not in skills        # 缺 description → 跳过而不是抛异常

    # 清理缓存，避免污染后续用例
    monkeypatch.undo()
    registry.load_skills(refresh=True)
