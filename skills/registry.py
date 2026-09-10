# skills/registry.py
"""技能库（Skill Registry）：把「工具 + 提示词 + 评测」打包成可版本化的能力单元。

━━━ 和「一堆函数」的区别 ━━━
原来新增一个能力要改三处代码：在 tools/ 写实现、在 supervisor.initialize 里
手工加进某个 Worker 的工具列表、再在 MCP server 里补一个 @mcp.tool()。
改完还得记得同步文档——漏一处就是线上事故。

技能库把这件事变成一个**声明式目录**：

    skills/<skill_name>/SKILL.md      ← 元数据（名字/版本/说明/何时使用/归属 Worker）
    skills/<skill_name>/handler.py    ← 实现（可选，也可指向已有的 tools/ 模块）
    skills/<skill_name>/eval.json     ← 该技能自己的评测用例

启动时扫描目录 → 解析元数据 → 动态加载 handler → 按归属自动装配到对应 Worker。
新增能力 = 新增一个目录，主流程代码零改动。

━━━ 为什么这个设计值钱（面试要讲的）━━━
1. **可版本化**：每个技能带 `version`，能力上线可以灰度、可以回滚；
2. **可评测**：eval.json 跟着技能走，改了这个技能就知道该跑哪批用例；
3. **可发现**：registry 能直接导出「系统当前具备哪些能力」，
   MCP / 文档 / 路由提示词都能从同一份元数据生成，不会出现"代码和文档对不上"。
"""
import importlib
import os
import re
from dataclasses import dataclass, field

from loguru import logger

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Skill:
    """一个技能（= 一个可版本化的能力单元）。"""

    name: str
    version: str = "0.0.0"
    description: str = ""
    when_to_use: str = ""
    handler: str = ""          # "模块路径:属性名"
    worker: str = ""           # 归属的 Worker 节点名
    enabled: bool = True
    dir_path: str = ""
    body: str = field(default="", repr=False)

    @property
    def eval_path(self) -> str:
        """该技能自带的评测用例路径（不存在则为空串）。"""
        p = os.path.join(self.dir_path, "eval.json")
        return p if os.path.exists(p) else ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "worker": self.worker,
            "enabled": self.enabled,
            "has_eval": bool(self.eval_path),
        }


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)


def _parse_front_matter(text: str):
    """解析 SKILL.md 的 YAML front matter（只支持 key: value 的简单子集，零依赖）。

    返回 (metadata_dict, body)。没有 front matter 时返回 ({}, 原文)。
    """
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta, body = {}, m.group(2)
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            meta[k] = v
    return meta, body


def _parse_bool(v, default=True) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("true", "1", "yes", "on")


_cache: dict = {}


def load_skills(refresh: bool = False) -> dict:
    """扫描 skills/ 目录，返回 {skill_name: Skill}。

    容错原则：单个技能包写坏了（缺字段 / 元数据非法）只跳过它并告警，
    绝不让整个服务起不来 —— 技能库是"可插拔"的，插头坏了不该烧主板。
    """
    if _cache and not refresh:
        return _cache

    skills: dict = {}
    if not os.path.isdir(SKILLS_DIR):
        return skills

    for entry in sorted(os.listdir(SKILLS_DIR)):
        d = os.path.join(SKILLS_DIR, entry)
        skill_md = os.path.join(d, "SKILL.md")
        if not os.path.isdir(d) or not os.path.exists(skill_md):
            continue
        try:
            with open(skill_md, encoding="utf-8") as f:
                meta, body = _parse_front_matter(f.read())
            name = meta.get("name") or entry
            if not meta.get("description"):
                logger.warning(f"[skills] 跳过 {entry}：SKILL.md 缺少 description")
                continue
            skills[name] = Skill(
                name=name,
                version=meta.get("version", "0.0.0"),
                description=meta.get("description", ""),
                when_to_use=meta.get("when_to_use", ""),
                handler=meta.get("handler", ""),
                worker=meta.get("worker", ""),
                enabled=_parse_bool(meta.get("enabled"), True),
                dir_path=d,
                body=body.strip(),
            )
        except Exception as e:
            logger.warning(f"[skills] 加载 {entry} 失败（已跳过，不影响启动）: {e}")

    _cache.clear()
    _cache.update(skills)
    logger.info(f"[skills] 已加载 {len(skills)} 个技能: {sorted(skills)}")
    return skills


def get_skill(name: str):
    return load_skills().get(name)


def list_skills() -> list:
    """列出全部技能（供 /skills 接口、MCP、文档生成复用同一份元数据）。"""
    return [s.to_dict() for s in load_skills().values()]


def resolve_handler(skill: Skill):
    """把 "模块路径:属性名" 动态解析成真实可调用对象。"""
    if not skill.handler or ":" not in skill.handler:
        return None
    mod_path, attr = skill.handler.split(":", 1)
    module = importlib.import_module(mod_path.strip())
    return getattr(module, attr.strip(), None)


def tools_for_worker(worker: str, guard: bool = True) -> list:
    """取出归属该 Worker 的全部技能，解析成可用的工具列表。

    guard=True 时统一过 Harness 的 guarded_tool（超时/重试/耗时日志），
    保证「技能」和原来的「工具」享受同一套执行保障 —— 这是不走形的关键。
    """
    out = []
    for skill in load_skills().values():
        if not skill.enabled or skill.worker != worker:
            continue
        try:
            fn = resolve_handler(skill)
            if fn is None:
                logger.warning(f"[skills] {skill.name} 无法解析 handler={skill.handler!r}，已跳过")
                continue
            if guard:
                from harness import guarded_tool
                fn = guarded_tool(fn)
            out.append((skill, fn))
        except Exception as e:
            logger.warning(f"[skills] {skill.name} 装配失败（已跳过）: {e}")
    return out
