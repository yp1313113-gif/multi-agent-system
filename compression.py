# compression.py
"""上下文压缩（Context Compression）：确定性压缩 → 重新计量 → 摘要压缩。

━━━ 为什么需要（面试口径）━━━
长对话 + 大段工具结果会把模型上下文撑满。不处理的后果是两个：

  1. 请求直接失败（ContextOverflowError）；
  2. 没爆但成本失控 —— 每一轮都带着几万 token 的历史，账单翻倍。

业界常见做法是「用一个 LLM 把历史总结掉」。但那有两个问题：
  · **贵**：每次压缩都要一次完整的 LLM 调用；
  · **丢**：摘要之后原始细节不可恢复，而工具结果里的数字（金额、比例、
    备查期限、订单号）恰恰是不能丢的。

所以本模块采用**两段式**：

  ┌─ 第一段【确定性压缩】—— 不调模型、不花钱、结果可复现
  │    · 超大工具结果 → 完整落盘，请求里只留「路径 + 内容哈希 + 预览」
  │    · 结构化结果   → 按预算保留关键字段（数量/ID/来源/标题/分数）
  │    · 其他文本     → 保留 head + middle + tail
  └─ 重新计量 ── 降到阈值以下就结束，**不调摘要模型**

  ┌─ 第二段【摘要压缩】—— 只有第一段压完仍然超标才做
  │    · 较早历史原文落盘（可追溯），生成一条 summary 消息
  └─ 保留最近 N 条原文

━━━ 关键设计：完整内容不丢 ━━━
压缩**只修改「模型请求视图」**，不改数据库里的原始记录。
用户翻历史看到的还是完整的；模型看到的才是压缩过的。
这与「把消息删掉」有本质区别 —— 后者是不可逆的信息损失。

━━━ 统计口径 ━━━
token 数是**近似估算**，只用于压力判断和预览长度，不是计费口径。
真实计费以 cost_tracker 记录的主模型 usage 为准。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OFFLOAD_DIR = os.path.join(PROJECT_ROOT, "outputs", "large_tool_results")
DEFAULT_HISTORY_DIR = os.path.join(PROJECT_ROOT, "outputs", "conversation_history")


# ============================================================
# 配置读取（config 不可用时退化为内置默认值 —— 保证模块可独立测试）
# ============================================================

def _cfg(name: str, default):
    try:
        from config import config
        v = getattr(config, name, None)
        return default if v is None else v
    except Exception:
        return default


def budget_tokens() -> int:
    """压力阈值：请求估算超过它才触发压缩。"""
    return int(_cfg("CONTEXT_PRESSURE_TOKENS", 6000))


def tool_result_limit() -> int:
    """单个工具结果超过这个 token 数就落盘。"""
    return int(_cfg("SUMMARY_TOOL_RESULT_TOKEN_LIMIT", 800))


def preview_tokens() -> int:
    """落盘后请求里保留的预览 token 数。"""
    return int(_cfg("TOOL_RESULT_PREVIEW_TOKENS", 200))


def keep_recent() -> int:
    """摘要时保留最近多少条消息原文。"""
    return int(_cfg("SUMMARY_KEEP_MESSAGES", 6))


def enabled() -> bool:
    return bool(_cfg("COMPRESSION_ENABLED", True))


# ============================================================
# token 近似估算
# ============================================================
_CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def estimate_tokens(text) -> int:
    """近似估算 token 数（不引入 tiktoken 依赖）。

    口径：中日韩全角字符约 0.7 token/字；其余字符约 0.25 token/字（≈ 4 字符 1 token）。
    误差在 ±20% 量级，对「压力判断」够用 —— 它要的是量级，不是账。

    用法上要记住：**这个数不是计费口径**。
    """
    if text is None:
        return 0
    s = text if isinstance(text, str) else str(text)
    if not s:
        return 0
    cjk = len(_CJK.findall(s))
    other = len(s) - cjk
    return int(cjk * 0.7 + other * 0.25) + 1


def _msg_text(msg) -> str:
    """从 LangChain 消息对象或 dict 里取出可计入上下文的文本。"""
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content or "")


def _msg_type(msg) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role") or msg.get("type") or "?")
    return str(getattr(msg, "type", "?"))


def estimate_messages_tokens(messages) -> int:
    """估算一组消息的总 token（含每条消息的角色开销）。"""
    total = 0
    for m in messages or []:
        total += estimate_tokens(_msg_text(m)) + 4   # 4 ≈ 单条消息的角色/结构开销
    return total


# ============================================================
# 确定性压缩原语
# ============================================================

def _rel(path: str) -> str:
    """尽量转成相对路径显示；跨盘符时退化为绝对路径，**绝不因此抛异常**。

    踩过的坑（测试抓出来的）：把卸载目录配到另一个盘时，
    os.path.relpath 会抛 ValueError: path is on mount 'C:', start on mount 'E:'。
    一个纯粹为了日志好看的转换，不该有机会把主流程带崩。
    """
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except Exception:
        return path


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _write_offload(text: str, tool_name: str, directory: str) -> str:
    """把完整内容落盘，返回绝对路径。

    文件名用「工具名 + 内容哈希」—— 同一内容重复落盘会稳定落在同一个文件，
    不会产生一堆重名副本（可追溯、可去重）。
    """
    safe_tool = re.sub(r"[^0-9A-Za-z_\-]", "_", tool_name or "tool")[:40]
    h = _content_hash(text)[:12]
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{safe_tool}-{h}.txt")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return path


def head_middle_tail(text: str, budget_chars: int,
                     head_ratio: float = 0.4, middle_ratio: float = 0.2) -> str:
    """保留 head + middle + tail，中间省略。

    为什么不是「只保留开头」：工具结果的**结论往往在末尾**（"综上，风险为红灯"
    "建议安全库存调整为 10"），砍掉尾巴等于把答案砍了。
    为什么还要中间：结构化文本（表格、清单）有时关键行在中段。
    """
    if budget_chars <= 0 or len(text) <= budget_chars:
        return text
    head_n = int(budget_chars * head_ratio)
    mid_n = int(budget_chars * middle_ratio)
    tail_n = max(0, budget_chars - head_n - mid_n)

    head = text[:head_n]
    tail = text[-tail_n:] if tail_n else ""
    mid_start = max(head_n, (len(text) - mid_n) // 2)
    middle = text[mid_start:mid_start + mid_n]

    omitted = len(text) - len(head) - len(middle) - len(tail)
    return (
        f"{head}\n"
        f"\n……（省略 {omitted} 字符）……\n"
        f"{middle}\n"
        f"\n……（省略后段，以下是结尾）……\n"
        f"{tail}"
    )


# 结构化结果里「值得保留」的字段（按优先级）
_STRUCT_KEYS = ("count", "total", "id", "ids", "source", "sources", "title", "titles",
                "url", "urls", "score", "scores", "name", "names", "sku", "status",
                "category", "amount", "success")


def compact_structured(text: str, budget_chars: int) -> str | None:
    """如果 text 是可解析的 JSON，按字段白名单压缩；否则返回 None（交给通用路径）。

    为什么单独处理结构化结果：JSON 里 80% 的体积可能是冗长的嵌套详情，
    而**列表数量、ID、来源、分数**这些「骨架信息」才是模型需要的。
    整段 head/tail 截断会把 JSON 截成语法不合法的半截，模型读不懂。
    """
    s = (text or "").strip()
    if not (s.startswith("{") or s.startswith("[")):
        return None
    try:
        data = json.loads(s)
    except Exception:
        return None

    def _slim(obj, depth=0):
        """递归瘦身：深度超限的嵌套直接摘要成类型描述。"""
        if depth >= 3:
            if isinstance(obj, list):
                return f"<list[{len(obj)}]>"
            if isinstance(obj, dict):
                return f"<dict[{len(obj)}]>"
            return obj
        if isinstance(obj, list):
            if len(obj) > 20:
                head = [_slim(x, depth + 1) for x in obj[:20]]
                return {"_truncated": True, "_total": len(obj), "_items": head}
            return [_slim(x, depth + 1) for x in obj]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in _STRUCT_KEYS or depth == 0:
                    out[k] = _slim(v, depth + 1)
                else:
                    out[k] = _slim(v, depth + 1) if not isinstance(v, (dict, list)) else "<omitted>"
            return out
        if isinstance(obj, str) and len(obj) > 200:
            return obj[:200] + "…"
        return obj

    slim = _slim(data)
    rendered = json.dumps(slim, ensure_ascii=False, indent=1)
    if len(rendered) > budget_chars:
        rendered = head_middle_tail(rendered, budget_chars)
    return rendered


def compact_tool_result(text: str, tool_name: str = "tool",
                        limit: int | None = None,
                        preview: int | None = None,
                        offload_dir: str | None = None) -> tuple[str, str]:
    """确定性压缩一条工具结果。

    Returns:
        (压缩后的文本, 卸载文件路径或空串)

    分级策略（从便宜到贵）：

      ① 结构化结果  → 按字段白名单瘦身（保留骨架，丢弃冗长详情）；
      ② 超大文本    → 完整落盘，请求里只留 [路径 + 哈希 + 预览]；
      ③ 中等文本    → head + middle + tail。
    """
    if not text:
        return text, ""
    limit = tool_result_limit() if limit is None else limit
    preview = preview_tokens() if preview is None else preview
    offload_dir = offload_dir or DEFAULT_OFFLOAD_DIR

    tokens = estimate_tokens(text)
    if tokens <= limit:
        return text, ""

    # ① 结构化：先尝试字段瘦身
    preview_chars = max(400, int(preview / 0.7))    # token → 字符（按中文口径）
    slimmed = compact_structured(text, preview_chars * 3)
    if slimmed is not None:
        return slimmed, ""

    # ② 超大：落盘 + 预览
    path = _write_offload(text, tool_name, offload_dir)
    h = _content_hash(text)
    head = text[:preview_chars]
    note = (
        f"[大工具结果已卸载到文件，完整内容不在上下文中]\n"
        f"路径：{_rel(path)}\n"
        f"sha256：{h}\n"
        f"原始约 {len(text)} 字符 / 约 {tokens} token。"
        f"以下为前 {preview} token 预览：\n"
        f"---\n{head}\n---"
    )
    return note, path


def compact_message(msg, tool_name: str = None, offload_dir: str | None = None):
    """对单条消息做确定性压缩（返回新消息；不可变则原样返回）。

    不改原对象 —— 压缩视图与原始记录必须分离。
    """
    text = _msg_text(msg)
    if not text:
        return msg, ""
    limit = tool_result_limit()
    if estimate_tokens(text) <= limit:
        return msg, ""
    compacted, path = compact_tool_result(text, tool_name or _msg_type(msg), offload_dir=offload_dir)
    if compacted == text:
        return msg, ""

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        t = _msg_type(msg)
        if t == "tool":
            new = ToolMessage(content=compacted, tool_call_id=getattr(msg, "tool_call_id", ""))
        elif t == "ai":
            new = AIMessage(content=compacted)
        elif t == "system":
            new = SystemMessage(content=compacted)
        else:
            new = HumanMessage(content=compacted)
        return new, path
    except Exception:
        # 无 langchain 环境（单测）时退化为 dict
        return {"type": _msg_type(msg), "content": compacted}, path


# ============================================================
# 报告
# ============================================================

@dataclass
class CompressionReport:
    """一次压缩的可观测结果（写日志 / 走 /compression 接口）。"""

    enabled: bool = True
    level: str = "none"              # none | deterministic | summarized
    reason: str = ""
    budget_tokens: int = 0
    trigger_tokens: int = 0          # 压缩前估算
    final_tokens: int = 0            # 压缩后估算
    messages_before: int = 0
    messages_after: int = 0
    compacted: int = 0               # 被确定性压缩的消息数
    summarized: int = 0              # 被摘要的历史消息数
    summary_tokens: int = 0
    offloaded: list = field(default_factory=list)
    history_file: str = ""
    at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["saved_tokens"] = max(0, self.trigger_tokens - self.final_tokens)
        d["saved_ratio"] = (
            round(1 - self.final_tokens / self.trigger_tokens, 3)
            if self.trigger_tokens > 0 else 0.0
        )
        return d

    def summary_line(self) -> str:
        return (
            f"level={self.level} 估算 {self.trigger_tokens}→{self.final_tokens} token "
            f"（省 {self.trigger_tokens - self.final_tokens}）"
            f"确定性压缩 {self.compacted} 条｜摘要 {self.summarized} 条"
            f"｜卸载 {len(self.offloaded)} 个文件"
        )


# 最近一次压缩报告（供 /compression 接口读取；多进程下只反映本进程）
LAST_REPORT: dict = {}


def _store(report: CompressionReport) -> CompressionReport:
    global LAST_REPORT
    LAST_REPORT = report.to_dict()
    return report


# ============================================================
# 第一段：确定性压缩
# ============================================================

def compact_messages(messages, keep: int | None = None,
                     offload_dir: str | None = None) -> tuple[list, CompressionReport]:
    """对消息列表做确定性压缩：**不动最近 N 条**，只压较早的大块内容。

    为什么保留最近 N 条完整：模型需要「刚才说了什么」的原文才能接上话。
    把最近几轮也压掉，回答会立刻变得前言不搭后语。

    ⚠️ 返回的是**新列表**，原 messages 不被修改。
    """
    keep = keep_recent() if keep is None else keep
    msgs = list(messages or [])
    report = CompressionReport(
        enabled=True,
        budget_tokens=budget_tokens(),
        trigger_tokens=estimate_messages_tokens(msgs),
        messages_before=len(msgs),
        at=datetime.now().isoformat(timespec="seconds"),
    )

    cutoff = max(0, len(msgs) - keep)
    out = []
    for i, m in enumerate(msgs):
        if i >= cutoff:
            out.append(m)          # 最近 N 条：原样保留
            continue
        new_m, path = compact_message(m, offload_dir=offload_dir)
        if path:
            report.offloaded.append(_rel(path))
        if new_m is not m and _msg_text(new_m) != _msg_text(m):
            report.compacted += 1
        out.append(new_m)

    report.messages_after = len(out)
    report.final_tokens = estimate_messages_tokens(out)
    report.level = "deterministic" if report.compacted else "none"
    report.reason = "确定性压缩" if report.compacted else "无需压缩"
    return out, report


# ============================================================
# 第二段：摘要压缩（唯一会花钱的一段）
# ============================================================

SUMMARY_PROMPT = """把下面这段较早的对话压缩成一段摘要，供后续对话继续使用。

要求：
1. **数字必须原样保留**（金额、比例、期限、工时、订单号、日期），一个都不能改；
2. 保留用户的身份与偏好、已经得出的结论、尚未解决的问题；
3. 丢失寒暄、重复表述、中间推理过程；
4. 用中文，尽量紧凑，不超过 300 字；
5. 只输出摘要正文，不要任何前言或标记。

对话原文：
{history}
"""


async def summarize_history(llm, messages, keep: int | None = None,
                            history_dir: str | None = None
                            ) -> tuple[list, CompressionReport]:
    """把较早历史摘要成一条消息，保留最近 N 条原文。

    完整原文先落盘再摘要 —— 摘要是有损的，原文必须还能查到。
    """
    keep = keep_recent() if keep is None else keep
    history_dir = history_dir or DEFAULT_HISTORY_DIR
    msgs = list(messages or [])
    report = CompressionReport(
        enabled=True,
        budget_tokens=budget_tokens(),
        trigger_tokens=estimate_messages_tokens(msgs),
        messages_before=len(msgs),
        at=datetime.now().isoformat(timespec="seconds"),
    )

    if len(msgs) <= keep:
        report.final_tokens = report.trigger_tokens
        report.messages_after = len(msgs)
        report.reason = "消息数不足，无需摘要"
        return msgs, report

    older, recent = msgs[:-keep], msgs[-keep:]
    rendered = "\n".join(f"[{_msg_type(m)}] {_msg_text(m)}" for m in older)

    # 原文落盘（摘要不可逆，必须留底）
    try:
        os.makedirs(history_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        hpath = os.path.join(history_dir, f"history-{stamp}-{_content_hash(rendered)[:8]}.txt")
        with open(hpath, "w", encoding="utf-8") as f:
            f.write(rendered)
        report.history_file = _rel(hpath)
    except Exception as e:
        logger.warning(f"[compression] 历史原文落盘失败（继续摘要）: {e}")

    try:
        resp = await llm.ainvoke(SUMMARY_PROMPT.format(history=rendered))
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = (body or "").strip()
    except Exception as e:
        logger.error(f"[compression] 摘要模型调用失败，退化为确定性压缩: {e}")
        report.reason = f"摘要失败({type(e).__name__})，退化为确定性压缩"
        det, det_report = compact_messages(msgs, keep=keep)
        det_report.reason = report.reason
        det_report.history_file = report.history_file
        return det, det_report

    try:
        from langchain_core.messages import SystemMessage
        summary_msg = SystemMessage(content=f"【较早对话摘要】\n{body}")
    except Exception:
        summary_msg = {"type": "system", "content": f"【较早对话摘要】\n{body}"}

    out = [summary_msg] + list(recent)
    report.summarized = len(older)
    report.summary_tokens = estimate_tokens(body)
    report.messages_after = len(out)
    report.final_tokens = estimate_messages_tokens(out)
    report.level = "summarized"
    report.reason = f"确定性压缩后仍超阈值，摘要 {len(older)} 条较早历史"
    return out, report


# ============================================================
# 主入口
# ============================================================

async def maybe_compress(llm, messages, keep: int | None = None,
                         budget: int | None = None, force: bool = False
                         ) -> tuple[list, CompressionReport]:
    """两段式压缩主入口。

    流程（对应 Yuxi 的自动压缩流程图）：
        计量 → 未达阈值 → 直接用原消息（0 成本）
              → 达阈值 → 确定性压缩 → 重新计量 → 降下来了就结束（0 成本）
                                              → 仍达阈值 → 摘要（唯一花钱的一步）
    """
    msgs = list(messages or [])
    budget = budget_tokens() if budget is None else budget

    if not enabled():
        r = CompressionReport(enabled=False, reason="压缩已关闭", budget_tokens=budget,
                              trigger_tokens=estimate_messages_tokens(msgs),
                              final_tokens=estimate_messages_tokens(msgs),
                              messages_before=len(msgs), messages_after=len(msgs))
        return msgs, _store(r)

    trigger = estimate_messages_tokens(msgs)
    if not force and trigger < budget:
        r = CompressionReport(level="none", reason=f"未达阈值（{trigger} < {budget}）",
                              budget_tokens=budget, trigger_tokens=trigger, final_tokens=trigger,
                              messages_before=len(msgs), messages_after=len(msgs),
                              at=datetime.now().isoformat(timespec="seconds"))
        return msgs, _store(r)

    # ── 第一段：确定性压缩（不花钱）
    det, report = compact_messages(msgs, keep=keep)
    if not force and report.final_tokens < budget:
        report.reason = "确定性压缩后已低于阈值，跳过摘要模型（省一次 LLM 调用）"
        logger.info(f"[compression] {report.summary_line()}")
        return det, _store(report)

    # ── 第二段：仍然超标 → 摘要
    if llm is None:
        report.reason = "仍超阈值但未提供摘要模型，保留确定性压缩结果"
        logger.warning(f"[compression] {report.reason}｜{report.summary_line()}")
        return det, _store(report)

    out, sreport = await summarize_history(llm, det, keep=keep)
    sreport.trigger_tokens = trigger          # 以「原始」为基准算节省
    sreport.offloaded = report.offloaded + sreport.offloaded
    sreport.compacted = report.compacted
    logger.info(f"[compression] {sreport.summary_line()}")
    return out, _store(sreport)


# ============================================================
# RAG 检索上下文拼装（原 rag_tool.format_docs 的升级版）
# ============================================================

def fit_context(docs, max_chars: int | None = None) -> tuple[str, CompressionReport]:
    """把检索到的文档拼成上下文，并做 token 预算截断。

    相比简单的「累加到超预算就停」，这里多做一步：
    **单篇文档自身就超过预算份额时，对它做 head+middle+tail**，
    而不是直接整篇丢弃 —— 后者会让「只召回了一篇超长文档」的场景直接空手而归。
    """
    max_chars = int(_cfg("RAG_CONTEXT_MAX_CHARS", 4000)) if max_chars is None else max_chars
    report = CompressionReport(budget_tokens=estimate_tokens("x" * max_chars),
                               at=datetime.now().isoformat(timespec="seconds"))
    docs = list(docs or [])
    if not docs:
        return "无相关上下文", report

    share = max(400, max_chars // max(1, min(len(docs), 4)))
    parts, total = [], 0
    for d in docs:
        content = getattr(d, "page_content", None)
        if content is None and isinstance(d, str):
            content = d
        content = content or ""
        if len(content) > share:
            before = len(content)
            content = head_middle_tail(content, share)
            report.compacted += 1
            logger.debug(f"[compression] 单篇文档 {before}→{len(content)} 字符（head/middle/tail）")
        if total + len(content) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            content = content[:remain]
        parts.append(content)
        total += len(content)
    report.messages_after = len(parts)
    report.final_tokens = estimate_tokens("\n\n".join(parts))
    report.level = "deterministic" if report.compacted else "none"
    report.reason = f"检索上下文 {len(parts)} 篇 / 预算 {max_chars} 字符"
    return "\n\n".join(parts), report


# ============================================================
# 路由请求视图（Supervisor 专用）
# ============================================================

# 路由视图里「必须保留原文」的最新消息条数。
# 路由只依赖最新一条 user 消息（系统提示词里写明了），所以保护 1 条即可 ——
# 保护得越多，窗口内的大块工具结果就越压不掉，token 白花。
_ROUTE_KEEP_RAW = 1


def route_window() -> int:
    """路由视图保留的消息条数。"""
    return int(_cfg("ROUTE_CONTEXT_WINDOW", 12))


def route_view(messages, window: int | None = None) -> tuple[list, CompressionReport]:
    """给 Supervisor 路由用的消息视图：窗口截断 + 确定性压缩。

    ━━━ 为什么这里默认**不摘要** ━━━
    路由每一轮请求都要跑。如果每次都调一次摘要模型，省下的 token 还不够
    付那一次调用 —— 而且摘要器本身也要把整段历史读一遍，等于没省。

    路由的真实需求只是「最近说了什么」（系统提示词里也明确写了
    「只依据最新一条 user 消息选择 Worker」）。所以这一层做两件零成本的事：
      · 窗口截断：只保留最近 window 条；
      · 确定性压缩：对窗口内的大块工具结果做 head/middle/tail 或落盘。

    摘要压缩留给真正需要长历史的场景（见 maybe_compress）。
    """
    window = route_window() if window is None else window
    msgs = list(messages or [])
    report = CompressionReport(
        budget_tokens=budget_tokens(),
        trigger_tokens=estimate_messages_tokens(msgs),
        messages_before=len(msgs),
        at=datetime.now().isoformat(timespec="seconds"),
    )

    # ① 窗口截断
    dropped = 0
    if window > 0 and len(msgs) > window:
        dropped = len(msgs) - window
        msgs = msgs[-window:]

    # ② 窗口内确定性压缩 —— 只保护**最新一条**
    #   ⚠️ 两个踩过的坑，都写在这里免得被改回去：
    #   · 传 keep=window：等于把窗口内每一条都划进保护名单，一条都压不动
    #     （实测：一个几十 K 的工具结果原样送进路由请求，白花 token）；
    #   · 传 keep=2：倒数第二条如果是大块工具结果，同样会被保护住。
    #   路由的提示词里明确写了「只依据最新一条 user 消息选择 Worker」，
    #   所以保护最新一条就够了 —— 这也正是路由上下文该省的地方。
    msgs, det = compact_messages(msgs, keep=min(_ROUTE_KEEP_RAW, len(msgs)))
    report.compacted = det.compacted
    report.offloaded = det.offloaded
    report.messages_after = len(msgs)
    report.final_tokens = estimate_messages_tokens(msgs)
    report.level = "deterministic" if (dropped or det.compacted) else "none"
    report.reason = (
        f"窗口截断丢弃 {dropped} 条较早消息"
        + (f"；确定性压缩 {det.compacted} 条" if det.compacted else "")
    ) if report.level != "none" else "无需压缩"
    return msgs, report


def stats() -> dict:
    """压缩模块当前配置 + 最近一次报告（供 /compression 接口）。"""
    return {
        "enabled": enabled(),
        "route_context_window": route_window(),
        "budget_tokens": budget_tokens(),
        "tool_result_token_limit": tool_result_limit(),
        "preview_tokens": preview_tokens(),
        "keep_recent_messages": keep_recent(),
        "offload_dir": _rel(DEFAULT_OFFLOAD_DIR),
        "history_dir": _rel(DEFAULT_HISTORY_DIR),
        "last_report": LAST_REPORT or None,
    }
