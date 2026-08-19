# tools/math_tool.py
import ast
import operator
from langchain.tools import tool
from exceptions import ToolExecutionError

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")

@tool
def calculate(expression: str) -> str:
    """数学计算，支持 + - * / ** 和括号。"""
    try:
        tree = ast.parse(expression, mode="eval").body
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except SyntaxError:
        return f"❌ 表达式语法错误: {expression}"
    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        raise ToolExecutionError(f"计算执行失败: {e}")