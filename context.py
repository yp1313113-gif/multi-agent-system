# context.py
"""
全局上下文：用于请求ID追踪
"""

# 当前请求ID（全局变量）
current_request_id: str = "N/A"


def set_request_id(request_id: str):
    """设置当前请求ID"""
    global current_request_id
    current_request_id = request_id


def get_request_id() -> str:
    """获取当前请求ID"""
    return current_request_id