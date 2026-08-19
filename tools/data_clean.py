# tools/data_clean.py
"""
RAG 数据清洗模块。

在「切块 → 向量化」之前对原始文本做规范化与降噪，提升召回质量：
  - Unicode 归一化（NFC）、去除 BOM / 控制字符；
  - 引号、全角空格、不间断空格、制表符归一化；
  - 折叠多余空行、去除行尾空白；
  - 剔除页码（纯数字行 / “第 N 页”）、重复行、纯装饰分隔行；
  - 保留 Markdown 标题（# / ## / ###），不破坏章节结构。

设计原则：清洗只做“降噪”，不改写语义，保证检索结果可回溯到原文。
"""
import re
import unicodedata

# 标点 / 空白归一化映射（仅处理容易引入噪声的字符，保留中文句读）
_CHAR_MAP = {
    "\u201c": '"', "\u201d": '"',   # “ ”
    "\u2018": "'", "\u2019": "'",   # ‘ ’
    "\u2014": "-", "\u2013": "-",   # — –
    "\u3000": " ",                  # 全角空格
    "\ufeff": "",                   # BOM
    "\u00a0": " ",                  # 不间断空格
    "\t": " ",                      # 制表符
}

# 需要整行剔除的模式
_PAGE_DIGIT = re.compile(r"^\s*\d+\s*$")                         # 纯数字页码
_PAGE_CN = re.compile(r"^\s*第\s*[\d一二三四五六七八九十]+\s*页\s*$")  # 第 N 页
_DECOR = re.compile(r"^[\s\-=_~·]{4,}$")                         # 纯装饰分隔线
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")           # 控制字符


def _normalize_chars(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for bad, good in _CHAR_MAP.items():
        text = text.replace(bad, good)
    text = _CONTROL.sub("", text)
    return text


def _clean_lines(text: str) -> list:
    out, prev = [], None
    for raw in text.split("\n"):
        line = raw.strip()
        # 跳过页码 / 装饰行
        if _PAGE_DIGIT.match(line) or _PAGE_CN.match(line) or _DECOR.match(line):
            continue
        line = line.rstrip()
        # 空行：连续空行折叠为单个，保留段落分隔
        if line == "":
            if out and out[-1] == "":
                continue
            out.append("")
            continue
        # 连续重复行去重（如重复出现的页眉 / 页脚标语）
        if line == prev:
            continue
        out.append(line)
        prev = line
    # 去除首尾空行
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


def clean_text(text: str) -> str:
    """对单篇原始文本做完整清洗，返回规范化后的文本。"""
    if not text:
        return ""
    text = _normalize_chars(text)
    lines = _clean_lines(text)
    return "\n".join(lines)


def clean_documents(texts: list) -> list:
    """批量清洗（接受字符串列表或 LangChain Document 列表）。"""
    result = []
    for t in texts:
        if hasattr(t, "page_content"):
            t.page_content = clean_text(t.page_content)
            result.append(t)
        else:
            result.append(clean_text(t))
    return result


if __name__ == "__main__":
    # 自测：用一段带噪声的文本验证清洗效果
    sample = (
        "\ufeff　　# 标题　　\n"
        "　　第一章 概述\n"
        "本文档说明“员工”的‘福利’安排。\u3000\u3000\n"
        "\n\n\n"
        "第 3 页\n"
        "==========\n"
        "工资发放日　为每月 10 日。\n"
        "工资发放日　为每月 10 日。\n"
    )
    print("--- 清洗前 ---")
    print(repr(sample))
    print("--- 清洗后 ---")
    print(repr(clean_text(sample)))
