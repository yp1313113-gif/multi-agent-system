# exceptions.py
"""
自定义异常分类
"""

class AgentError(Exception):
    """所有 Agent 异常的基类"""
    pass

class ToolTimeoutError(AgentError):
    """工具调用超时"""
    pass

class ToolRateLimitError(AgentError):
    """工具调用频率限制"""
    pass

class KnowledgeBaseError(AgentError):
    """知识库查询失败"""
    pass

class APIAuthenticationError(AgentError):
    """API 认证失败"""
    pass

class ToolExecutionError(AgentError):
    """工具执行失败（未知）"""
    pass