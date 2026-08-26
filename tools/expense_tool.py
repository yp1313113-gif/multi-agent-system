"""研发费用数据归集工具：把费用条目归集到 8 类研发费用口径（模拟 ERP 数据）。"""
from langchain.tools import tool

_CATEGORIES = [
    "人员人工费用", "直接投入费用", "折旧费用", "无形资产摊销",
    "新产品设计费等", "装配调试费用", "其他相关费用", "委托研发费用",
]

_MOCK_EXPENSES = [
    {"id": "E001", "desc": "研发工程师张三 11 月工资", "category": "人员人工费用", "amount": 25000},
    {"id": "E002", "desc": "研发服务器 11 月电费", "category": "直接投入费用", "amount": 3200},
    {"id": "E003", "desc": "实验设备折旧", "category": "折旧费用", "amount": 8000},
    {"id": "E004", "desc": "研发软件 license 摊销", "category": "无形资产摊销", "amount": 1500},
    {"id": "E005", "desc": "新产品设计外包费", "category": "新产品设计费等", "amount": 12000},
    {"id": "E006", "desc": "专家技术咨询费", "category": "其他相关费用", "amount": 3000},
]


@tool
def classify_expense(category: str, amount: float, description: str) -> str:
    """把一笔费用归类到 8 类研发费用口径之一，返回归集结果。

    Args:
        category: 费用类别（人员人工费用/直接投入费用/折旧费用/无形资产摊销/新产品设计费等/装配调试费用/其他相关费用/委托研发费用）
        amount: 金额（元）
        description: 费用说明
    """
    if category not in _CATEGORIES:
        return "❌ 归类失败：类别必须是：" + "、".join(_CATEGORIES)
    if amount <= 0:
        return "❌ 金额必须大于 0。"
    return f"✅ 已归集：{description} → {category}，金额 ¥{amount:,.0f}。"


@tool
def list_rd_expenses() -> str:
    """列出当前模拟 ERP 中的研发费用条目及归集口径。"""
    lines = ["当前 ERP 研发费用条目："]
    for e in _MOCK_EXPENSES:
        lines.append(f"· {e['id']} {e['desc']} → {e['category']} ¥{e['amount']:,}")
    return "\n".join(lines)
