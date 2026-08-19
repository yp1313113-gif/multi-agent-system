# tools/doc_parser.py
"""
RAG 文档解析：按「章 → 小节」切分并归属逻辑数据源。

该模块只依赖 config（不引入 langchain / chromadb 等重型依赖），
便于独立单元测试与复用。

章节约定（与 ingest.py / data/公司行政管理手册.txt 一致）：
  - `## 第X章`  作为一级切分（中文数字 X）；
  - `### 小节`  作为二级切分（如 `### 一、工作时间`）；
  - 章号依据 config.DATA_SOURCES[].chapters 映射到逻辑数据源
    （考勤与假期 / 薪酬与福利 / 通用制度）。
"""
import re

from config import config

_CN_NUM = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
           '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}

# 小节标题，如 `### 一、工作时间`
_SEC = re.compile(r'^###\s*(.+)$', re.M)
# 章标题，如 `## 第一章`
_CHAPTER = re.compile(r'^##\s*第([一二三四五六七八九十\d]+)章', re.M)


def cn2int(s: str) -> int:
    """把中文数字（一~十）转成整数。"""
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    if len(s) == 2 and s[0] == '十':
        return 10 + (int(s[1]) if s[1].isdigit() else _CN_NUM.get(s[1], 0))
    if len(s) == 2 and s[1] == '十':
        return _CN_NUM.get(s[0], 0) * 10
    return 0


def parse_chapters(text: str):
    """按 `## 第X章` 切分正文，返回 [(章号, 正文), ...]。"""
    matches = list(_CHAPTER.finditer(text))
    chapters = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chapters.append((cn2int(m.group(1)), text[start:end]))
    return chapters


def split_sections(chapter_body: str):
    """把某一章按 `### 小节` 拆成 [(小节标题, 正文), ...]，无小节则整体返回。"""
    ms = list(_SEC.finditer(chapter_body))
    if not ms:
        return [(None, chapter_body)]
    secs = []
    for i, mm in enumerate(ms):
        start = mm.start()
        end = ms[i + 1].start() if i + 1 < len(ms) else len(chapter_body)
        secs.append((mm.group(1).strip(), chapter_body[start:end]))
    return secs


def chapter_to_source(chap_num: int) -> str:
    """根据章号返回所属逻辑数据源名称。"""
    for name, info in config.DATA_SOURCES.items():
        if chap_num in info.get("chapters", []):
            return name
    return config.DEFAULT_DATA_SOURCE
