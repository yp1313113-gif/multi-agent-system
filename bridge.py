# bridge.py
"""把 Agent 侧的业务动作同步到看板的数据文件（仅演示用）。

━━━ 为什么需要这一层 ━━━
Agent 的业务黑板是 contextvars 的**请求级**状态，HTTP 请求结束就没了。
看板是另一个页面、另一个请求，跨请求读不到黑板。
所以演示模式下，工具写黑板的同时也追加一份到「用户输入数据.json」，
看板重新生成快照时就能看到 —— 这样「在聊天里说一句」和
「切到看板看到那条数据」才连得上。

━━━ 这不是生产架构 ━━━
生产上应该是同一个数据库 + 事件通知（或者干脆就是一个应用里的两个视图）。
这里用文件桥接，是因为项目结构是「Agent 服务 + 独立看板页」，
不想为了演示把两个模块耦死。

━━━ 三个设计约束 ━━━
1. 软依赖：文件不存在、目录不可写、JSON 坏了，都只告警不抛异常 ——
   绝不能让一次「写看板数据」失败，把正常的业务调用搞挂。
2. 可关闭：环境变量 BRIDGE_TO_DASHBOARD=false 就完全停用。
3. 不改业务返回值：桥接失败不影响工具返回给用户的话术。
"""
import io
import json
import os

from loguru import logger

_USER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "用户输入数据.json")


# 静默开关：生成看板快照时会调用业务工具，如果那时还往用户数据里写，
# 就会变成「快照读用户数据 → 工具写用户数据 → 下次快照又多一条」，无限繁殖。
# demo_seed 在生成快照前会把它打开，生成完再关掉。
_silent = False


def set_silent(v: bool) -> None:
    """临时关闭同步（生成快照时用）。"""
    global _silent
    _silent = bool(v)


def _enabled() -> bool:
    if _silent:
        return False
    return os.getenv("BRIDGE_TO_DASHBOARD", "true").lower() not in ("0", "false", "no")


def _load(path: str) -> dict:
    try:
        with io.open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        d = {}
    d.setdefault("timesheets", [])
    d.setdefault("expenses", [])
    return d


def _save(path: str, d: dict) -> None:
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def append_timesheet(employee: str, project: str, date: str,
                     hours: float, task: str = "") -> bool:
    """把一条工时同步到看板数据。成功返回 True。"""
    if not _enabled():
        return False
    try:
        d = _load(_USER_FILE)
        d["timesheets"].append({
            "employee": employee, "project": project,
            "date": date, "hours": float(hours), "task": task or "",
        })
        _save(_USER_FILE, d)
        logger.info(f"[bridge] 工时已同步到看板：{employee} {project} {hours}h")
        return True
    except Exception as e:
        logger.warning(f"[bridge] 工时同步失败（不影响业务）：{e}")
        return False


def append_expense(category: str, amount: float, description: str,
                   confidence: float = 1.0) -> bool:
    """把一笔费用同步到看板数据。成功返回 True。"""
    if not _enabled():
        return False
    try:
        d = _load(_USER_FILE)
        d["expenses"].append({
            "category": category, "amount": float(amount),
            "description": description, "confidence": float(confidence),
        })
        _save(_USER_FILE, d)
        logger.info(f"[bridge] 费用已同步到看板：{description} ¥{amount}")
        return True
    except Exception as e:
        logger.warning(f"[bridge] 费用同步失败（不影响业务）：{e}")
        return False


def clear() -> None:
    """清空看板侧的用户数据（看板点「清空我填的」时不用它，那条走 API）。"""
    try:
        _save(_USER_FILE, {"timesheets": [], "expenses": []})
    except Exception as e:
        logger.warning(f"[bridge] 清空失败：{e}")
