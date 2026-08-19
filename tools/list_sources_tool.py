# tools/list_sources_tool.py
from langchain.tools import tool
from config import config


@tool
def list_data_sources() -> str:
    """
    列出所有可用的知识库数据源。
    当用户询问"有哪些知识库"、"可以用哪些数据源"时，调用此工具。
    """
    sources = config.DATA_SOURCES
    if not sources:
        return "📚 暂无可用知识库"

    result = "📚 可用知识库：\n"
    for name, info in sources.items():
        result += f"  - {name}: {info['description']}\n"
    result += f"\n💡 当前默认数据源: {config.DEFAULT_DATA_SOURCE}\n"
    result += "\n提示：你可以指定数据源查询，例如「查一下考勤与假期里的年假规定」"
    return result