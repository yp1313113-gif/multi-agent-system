# rules.py
"""归类规则库：把「人工复核过的裁定」沉淀成可复用规则。

━━━ 为什么需要这一层 ━━━
研发费用归集最大的成本不是算钱，是「同一类支出反复要人判断该归到哪一类」。
财务每次都要问：这笔差旅费算不算研发费用？算「其他相关费用」还是「直接投入」？

    hitl.py  负责「请人来裁定一次」
    rules.py 负责「把这一次裁定，变成以后不用再问的人」

━━━ 闭环 ━━━
    一笔支出归类不确定 → 写 pending → 人工裁定
        ↓
    learn_from_approval() 沉淀规则
        ↓
    下次同类支出 → find_rule() 命中 → 自动归类（不问人）
        ↓
    指标：自动归类率 = 自动归类笔数 / 总归集笔数

━━━ 两个设计取舍 ━━━
1. 关键词用【白名单词表】抽取，而不是分词 / 模型。
   为什么？规则一旦抽错就会持续误杀（比如把"研发"当关键词，所有含"研发"的支出都被归到同一类）。
   白名单可解释、不会凭空造词，宁可少沉淀一条规则，也不要污染规则库。

2. 规则带 source_approval —— 任何一条规则都能回溯到「哪一次复核定下的」。
   没有可追溯性的规则库，用久了没人敢删。
"""
import os
import sqlite3
import uuid
from datetime import datetime

from loguru import logger

# 研发费用归集的六类口径（单一事实来源；tools/expense_tool.py 从这里引入）
# 依据：财税〔2015〕119 号、国家税务总局公告 2015 年第 97 号
#
# 为什么不是 8 类：早期版本多了「装配调试费用」和「委托研发费用」，两个都不对 ——
#   1. 装配调试费用不是法定的六类之一；
#   2. 委托研发不走这六类，它是单独一条规则：按实际发生额的 80% 计入委托方研发费用，
#      而且要有委托合同，境外委托还有额外限制。
# 所以委托研发在 expense_tool 里作为独立分支处理，不放进 CATEGORIES。
CATEGORIES = [
    "人员人工费用", "直接投入费用", "折旧费用", "无形资产摊销",
    "新产品设计费等", "其他相关费用",
]

# 关键词白名单：只有命中这些词的费用说明，才允许沉淀成规则
KEYWORD_WHITELIST = [
    "差旅费", "会议费", "专家咨询费", "技术咨询费", "技术服务费",
    "材料费", "检测费", "试验费", "专利费", "软件摊销", "设备折旧",
    "设计费", "外包费", "委托开发费", "知识产权费", "职工薪酬",
    "资料费",
]
# 注意：早期版本把「办公费」「通讯费」「培训费」也放进白名单，这是错的。
# 依据国家税务总局公告 2017 年第 40 号，「其他相关费用」采用列举式：
#   技术图书资料费、资料翻译费、专家咨询费、高新科技研发保险费、
#   研发成果的检索/分析/评议/论证/评审/评估费、知识产权申请注册代理费、
#   差旅费、会议费。
# 办公、通讯、培训都不在列举范围内，不能计入研发费用，更谈不上加计扣除。

# 关键词 → 六类口径的默认归类。
# 依据：财税〔2015〕119 号对六类研发费用范围的界定。
# 这张表只在「还没有人工裁定过同类费用」时兜底使用；
# 一旦规则库里沉淀了某个关键词的裁定，以规则库为准（人工裁定优先于默认映射）。
#
# 顺序有意义：先匹配更具体、更长的词。
# 例如「技术咨询费」要排在「咨询费」前面，「软件摊销」要排在「摊销」前面。
KEYWORD_CATEGORY = [
    ("职工薪酬", "人员人工费用"),
    ("工资", "人员人工费用"),
    ("薪金", "人员人工费用"),
    ("社保", "人员人工费用"),
    ("公积金", "人员人工费用"),
    ("劳务费", "人员人工费用"),

    ("材料费", "直接投入费用"),
    ("原材料", "直接投入费用"),
    ("燃料", "直接投入费用"),
    ("动力费", "直接投入费用"),
    ("检测费", "直接投入费用"),
    ("试验费", "直接投入费用"),
    ("模具费", "直接投入费用"),
    ("租赁费", "直接投入费用"),
    ("维修费", "直接投入费用"),
    ("外包费", "直接投入费用"),

    ("设备折旧", "折旧费用"),
    ("折旧", "折旧费用"),

    ("软件摊销", "无形资产摊销"),
    ("无形资产摊销", "无形资产摊销"),
    ("摊销", "无形资产摊销"),

    ("设计费", "新产品设计费等"),
    ("新工艺", "新产品设计费等"),
    ("临床试验", "新产品设计费等"),

    # 短词根兜底：实际费用说明里经常是「检测服务费」「外包开发费」这种，
    # 中间夹了别的字，长词匹配不到，所以补一层更短的词根。
    # 注意顺序：短词根必须放在长词之后，否则会抢走更精确的匹配。
    ("检测", "直接投入费用"),
    ("测试", "直接投入费用"),
    ("化验", "直接投入费用"),
    ("外包", "直接投入费用"),
    ("委托开发", "直接投入费用"),
    ("耗材", "直接投入费用"),
    ("材料", "直接投入费用"),
    ("服务器", "直接投入费用"),
    ("云服务", "直接投入费用"),
    ("算力", "直接投入费用"),

    # 其余都按「其他相关费用」兜底（差旅、会议、咨询、专利、资料等）
]


# 明确不属于研发费用范围的科目（负面清单）。
# 依据：财税〔2015〕119 号 规定可加计扣除的研发费用是「列举式」的，
# 国家税务总局公告 2017 年第 40 号 又对「其他相关费用」做了列举。
# 列举之外的支出不能计入，否则属于虚增研发费用，稽查时要调减并可能加收滞纳金。
#
# 这一层的价值和「关键词归到哪一类」不同：
#   前者是「这笔算不算研发费用」，后者是「算的话归到哪一类」。
# 判断「不算」比判断「归哪类」更重要 —— 算错类别只是分类错，算错范围是虚报。
NON_RD_KEYWORDS = [
    # 行政后勤类
    "办公用品", "办公费", "文具", "打印纸", "清洁用品", "绿化", "物业费",
    "水费", "电费", "话费", "通讯费", "快递费", "邮寄费",
    "饮用水", "桶装水", "矿泉水", "茶叶", "清洁",
    # 人事福利类
    "培训费", "招聘", "福利费", "团建", "聚餐", "体检费", "工装",
    # 招待宣传类
    "业务招待", "招待费", "礼品", "广告费", "业务宣传", "赞助",
    # 违规类（这类即使和研发有关也不能扣）
    "罚款", "滞纳金", "违约金", "捐赠",
]


def is_non_rd(description: str) -> tuple:
    """判断这笔费用是不是明确不能计入研发费用。

    返回 (是否排除, 命中的关键词)。命中返回原因，供导入时给用户解释。
    """
    text = description or ""
    for kw in NON_RD_KEYWORDS:
        if kw in text:
            return True, kw
    return False, ""


def guess_category(description: str) -> str:
    """按费用说明猜一个类别。猜不出返回「其他相关费用」。

    这是兜底：没有人工裁定过的关键词，先按政策口径给个初始归类，
    再交给人工复核或规则库纠正。
    """
    text = description or ""
    for kw, cat in KEYWORD_CATEGORY:
        if kw in text and cat in CATEGORIES:
            return cat
    return "其他相关费用"


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approval.db")


def get_db():
    """与 hitl 共用同一个库（复核记录和沉淀出的规则放在一起，便于追溯）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS classification_rules (
            id              TEXT PRIMARY KEY,
            keyword         TEXT NOT NULL,
            scope           TEXT DEFAULT '',
            category        TEXT NOT NULL,
            rationale       TEXT DEFAULT '',
            source_approval TEXT,
            source_text     TEXT DEFAULT '',
            hit_count       INTEGER DEFAULT 0,
            created_at      TIMESTAMP,
            UNIQUE (keyword, scope)
        );

        CREATE TABLE IF NOT EXISTS classification_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            category    TEXT,
            amount      REAL,
            auto        INTEGER DEFAULT 0,   -- 1=规则命中自动归类，0=人工/默认
            rule_id     TEXT,
            created_at  TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============ 规则抽取与沉淀 ============

def _extract_keyword(text: str) -> str:
    """从费用说明里抽一个可复用的关键词（白名单内）。找不到返回空串。"""
    text = text or ""
    for kw in KEYWORD_WHITELIST:
        if kw in text:
            return kw
    return ""


def upsert_rule(keyword: str, category: str, rationale: str = "",
                source_approval: str = "", source_text: str = "",
                scope: str = "") -> str:
    init_db()
    rid = f"rule_{uuid.uuid4().hex[:8]}"
    conn = get_db()
    conn.execute(
        """INSERT INTO classification_rules
               (id, keyword, scope, category, rationale, source_approval, source_text, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (keyword, scope) DO UPDATE SET
               category        = excluded.category,
               rationale       = excluded.rationale,
               source_approval = excluded.source_approval,
               source_text     = excluded.source_text""",
        (rid, keyword, scope, category, rationale, source_approval, source_text, _now()),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM classification_rules WHERE keyword = ? AND scope = ?", (keyword, scope)
    ).fetchone()
    conn.close()
    return row["id"] if row else rid


def learn_from_approval(approval_id: str):
    """把一次复核裁定沉淀成规则。返回规则 ID；无法沉淀则返回 None。

    由 hitl.resolve() 调用（延迟导入，避免循环依赖）。
    """
    import hitl

    conn = hitl.get_db()
    row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
    conn.close()
    if not row or row["status"] != "approved":
        return None
    if row["kind"] != "expense_classify":
        return None          # 只有「归类」类裁定能变成归类规则
    decision = (row["decision"] or "").strip()
    if decision not in CATEGORIES:
        return None          # 裁定不是合法口径之一 —— 不沉淀，避免污染规则库
    source_text = row["tool_input"] or ""
    keyword = _extract_keyword(source_text)
    if not keyword:
        logger.info(f"[rules] 「{source_text[:30]}」里没有白名单关键词，本次不沉淀规则")
        return None
    return upsert_rule(
        keyword=keyword,
        category=decision,
        rationale=row["rationale"] or "",
        source_approval=approval_id,
        source_text=source_text,
    )


# ============ 规则匹配与命中统计 ============

def find_rule(description: str, scope: str = ""):
    """按费用说明找一条可复用规则。

    匹配优先级：带作用域且命中 > 全局命中；同优先级下优先命中次数多的。
    """
    init_db()
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM classification_rules ORDER BY (scope = '') ASC, hit_count DESC"
    ).fetchall()
    conn.close()
    for r in rows:
        r_scope = r["scope"] or ""
        if r_scope and r_scope != scope:
            continue
        if r["keyword"] and r["keyword"] in (description or ""):
            return dict(r)
    return None


def touch_rule(rule_id: str) -> None:
    """规则被命中一次 —— hit_count 是「规则库在起作用」的证据。"""
    conn = get_db()
    conn.execute(
        "UPDATE classification_rules SET hit_count = hit_count + 1 WHERE id = ?", (rule_id,)
    )
    conn.commit()
    conn.close()


def list_rules():
    init_db()
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM classification_rules ORDER BY hit_count DESC, created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============ 指标 ============

def log_classification(description: str, category: str, amount: float,
                       auto: bool, rule_id=None) -> None:
    init_db()
    conn = get_db()
    conn.execute(
        "INSERT INTO classification_log (description, category, amount, auto, rule_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (description, category, float(amount or 0), 1 if auto else 0, rule_id, _now()),
    )
    conn.commit()
    conn.close()


def metrics():
    """归集指标：自动归类率 + 人工介入笔数。

    自动归类率 = 规则命中自动归类的笔数 / 总归集笔数
    """
    init_db()
    conn = get_db()
    row = conn.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(auto), 0) AS auto_n FROM classification_log"
    ).fetchone()
    rules_n = conn.execute("SELECT COUNT(*) AS n FROM classification_rules").fetchone()["n"]
    hits = conn.execute(
        "SELECT COALESCE(SUM(hit_count), 0) AS h FROM classification_rules"
    ).fetchone()["h"]
    conn.close()
    total = int(row["total"])
    auto_n = int(row["auto_n"])
    return {
        "total": total,
        "auto": auto_n,
        "manual": total - auto_n,
        "auto_rate": (auto_n / total) if total else 0.0,
        "rules": int(rules_n),
        "rule_hits": int(hits),
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    import hitl
    import context

    init_db()
    hitl.init_db()
    conn = get_db()
    conn.execute("DELETE FROM classification_rules")
    conn.execute("DELETE FROM classification_log")
    conn.commit()
    conn.close()
    c = hitl.get_db()
    c.execute("DELETE FROM approvals")
    c.commit()
    c.close()

    print("=== 归类规则沉淀闭环演示 ===")
    context.set_requester("财务小张")

    print("\n① 第 1 笔：研发部门差旅费 ¥3,200 —— 无规则，需人工裁定")
    aid = hitl.request_approval(
        tool_name="classify_expense",
        tool_input="研发部门出差北京的差旅费",
        user_message="研发部门出差北京的差旅费",
        context={"amount": 3200},
        kind="expense_classify",
        reason="归类置信度偏低，差旅费可能属「其他相关费用」",
    )
    print("   登记复核：", aid)

    rid = hitl.resolve(aid, decision="其他相关费用",
                       rationale="差旅费不属于六类直接费用，计入其他相关费用（限额 10%）")
    print("   裁定后自动归类率：", f"{metrics()['auto_rate']:.0%}", "（还没有归类记录）")

    print("\n② 第 2 笔：另一笔研发部门差旅费 ¥2,800 —— 应该命中规则、自动归类")
    rule = find_rule("研发部门出差上海的差旅费")
    if rule:
        touch_rule(rule["id"])
        log_classification("研发部门出差上海的差旅费", rule["category"], 2800, auto=True,
                           rule_id=rule["id"])
        print(f"   ✅ 命中规则 [{rule['id']}] keyword={rule['keyword']}"
              f" → {rule['category']}（依据：{rule['rationale'][:24]}…）")
    else:
        print("   ❌ 未命中规则")

    print("\n③ 第 3 笔：研发服务器电费 ¥1,200 —— 没有对应规则，走人工")
    log_classification("研发服务器电费", "直接投入费用", 1200, auto=False)

    m = metrics()
    print("\n=== 指标 ===")
    print(f"  总归集 {m['total']} 笔｜自动 {m['auto']} 笔｜人工 {m['manual']} 笔"
          f"｜自动归类率 {m['auto_rate']:.0%}")
    print(f"  规则库 {m['rules']} 条｜累计命中 {m['rule_hits']} 次")

    print("\n=== 规则库 ===")
    for r in list_rules():
        print(f"  [{r['id']}] {r['keyword']} → {r['category']}"
              f"（来源复核 {r['source_approval']}，命中 {r['hit_count']} 次）")
