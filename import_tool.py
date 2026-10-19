# import_tool.py
"""费用导入：把财务导出的 CSV / Excel 读进来，逐行归集到研发费用六类口径。

━━━ 为什么自己解析 xlsx 而不用 openpyxl ━━━
本项目的虚拟环境里没有 openpyxl，pip 装不上（网络受限）。
xlsx 本质是一个 zip 包，里面是几张 XML 表，只读「简单二维表格」的话
用标准库 zipfile + xml.etree 就够，而且零依赖、换台机器也能跑。
代价是不支持合并单元格、公式、多个 sheet 的复杂场景 —— 对「财务导出的明细账」
这种规整表格来说完全够用。

━━━ 模板列 ━━━
日期, 凭证号, 费用说明, 不含税金额, 税额, 发票类型, 所属项目

「发票类型」只认三种：专票 / 普票 / 无票。
  · 专票：进项税可抵扣，计入【不含税金额】
  · 普票：进项税不可抵扣，计入【不含税 + 税额】
  · 无票：不涉及进项税，计入【不含税金额】
这一列决定了成本金额，录错整张表都要重算。
"""
import csv
import io
import os
import re
import zipfile
from xml.etree import ElementTree as ET

# xlsx 里单元格的命名空间
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

COLUMNS = ["日期", "凭证号", "费用说明", "不含税金额", "税额", "发票类型", "所属项目"]

# 发票类型别名，容错用
_INV_ALIAS = {
    "专票": "专票", "增值税专用发票": "专票", "专用发票": "专票", "special": "专票",
    "普票": "普票", "增值税普通发票": "普票", "普通发票": "普票", "normal": "普票",
    "无票": "无票", "不开票": "无票", "内部": "无票", "none": "无票", "": "无票",
}


def _norm_inv(s: str) -> str:
    k = (s or "").strip()
    return _INV_ALIAS.get(k, _INV_ALIAS.get(k.lower(), "无票"))


def compute_amount(no_tax: float, tax: float, inv_type: str) -> float:
    """算「计入研发费用的金额」。

    专票 -> 不含税（税被抵掉了，不构成成本）
    普票 -> 价税合计（税不能抵，也是成本）
    无票 -> 不含税
    """
    no_tax = float(no_tax or 0)
    tax = float(tax or 0)
    t = _norm_inv(inv_type)
    if t == "普票":
        return round(no_tax + tax, 2)
    return round(no_tax, 2)


# ============================ xlsx 最小读取器 ============================

def _col_to_idx(ref: str) -> int:
    """把 A1 这种坐标里的列名转成 0 基索引。"""
    m = re.match(r"([A-Z]+)", ref or "")
    if not m:
        return 0
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def _read_shared_strings(zf) -> list:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    out = []
    for si in root.findall(_NS + "si"):
        # 文本可能被拆成多个 <t>（富文本），要拼起来
        parts = [t.text or "" for t in si.iter(_NS + "t")]
        out.append("".join(parts))
    return out


def _pick_sheet(zf) -> str:
    """挑第一张表（按 workbook.xml 的顺序，而不是文件名排序）。"""
    try:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = wb.find(_NS + "sheets/" + _NS + "sheet")
        if sheet is not None:
            rid = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            if rid:
                rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
                for rel in rels:
                    if rel.get("Id") == rid:
                        target = rel.get("Target", "").lstrip("/")
                        if not target.startswith("xl/"):
                            target = "xl/" + target
                        return target
    except Exception:
        pass
    return "xl/worksheets/sheet1.xml"


def read_xlsx_rows(path: str) -> list:
    """读 xlsx 的第一张表，返回二维字符串列表。零依赖实现。"""
    with zipfile.ZipFile(path) as zf:
        shared = _read_shared_strings(zf)
        sheet_path = _pick_sheet(zf)
        root = ET.fromstring(zf.read(sheet_path))

        rows = []
        for row in root.iter(_NS + "row"):
            cells = {}
            for c in row.findall(_NS + "c"):
                ref = c.get("r", "")
                typ = c.get("t", "")
                idx = _col_to_idx(ref)
                v = c.find(_NS + "v")
                if typ == "inlineStr":
                    is_el = c.find(_NS + "is")
                    text = "".join(t.text or "" for t in is_el.iter(_NS + "t")) if is_el is not None else ""
                    cells[idx] = text
                elif v is None:
                    cells[idx] = ""
                elif typ == "s":
                    cells[idx] = shared[int(v.text)] if v.text and int(v.text) < len(shared) else ""
                else:
                    cells[idx] = v.text or ""
            if cells:
                width = max(cells) + 1
                rows.append([cells.get(i, "") for i in range(width)])
        return rows


# ============================ 统一入口 ============================

def read_rows(path: str) -> list:
    """按扩展名分发。CSV 走标准库，xlsx 走自解析。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        return read_xlsx_rows(path)
    if ext == ".xls":
        raise ValueError("不支持老版 .xls，请在 Excel 里另存为 .xlsx 或 .csv")
    # CSV：先试 utf-8-sig（Excel 存的中文 CSV 常带 BOM），失败再试 gbk
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return [list(r) for r in csv.reader(f)]
        except UnicodeDecodeError:
            continue
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return [list(r) for r in csv.reader(f)]


def parse_rows(raw_rows: list) -> dict:
    """把原始二维表解析成结构化行。返回 {rows, errors}。

    容错设计：
      · 表头位置不固定：在前 5 行里找同时含「费用说明」和「金额」的那一行
      · 列顺序不固定：按表头名匹配
      · 金额列允许叫「不含税金额」或「金额」
      · 空行跳过
    """
    errors = []
    rows = []
    header_idx = -1
    col_map = {}

    for i, r in enumerate(raw_rows[:8]):
        joined = "|".join(str(x).strip() for x in r)
        if ("费用说明" in joined or "摘要" in joined) and ("金额" in joined):
            header_idx = i
            for j, cell in enumerate(r):
                name = str(cell).strip()
                if name in COLUMNS:
                    col_map[name] = j
                elif name == "金额":
                    col_map.setdefault("不含税金额", j)
                elif name == "摘要":
                    col_map.setdefault("费用说明", j)
            break

    if header_idx < 0:
        return {"rows": [], "errors": ["没找到表头。前 8 行里需要有「费用说明」和「金额」这两列。"]}

    if "费用说明" not in col_map:
        return {"rows": [], "errors": ["缺少「费用说明」列，无法判断这笔费用是什么。"]}
    if "不含税金额" not in col_map:
        return {"rows": [], "errors": ["缺少「金额」或「不含税金额」列。"]}

    def cell(r, name, default=""):
        j = col_map.get(name)
        if j is None or j >= len(r):
            return default
        return str(r[j]).strip()

    for i, r in enumerate(raw_rows[header_idx + 1:], start=header_idx + 2):
        if not any(str(x).strip() for x in r):
            continue
        # 模板示例行：日期列带着标记，直接跳过并说明原因。
        # 放在其它校验之前，这样它不会当成"金额错"之类的问题报出来。
        if TEMPLATE_MARK[:4] in cell(r, "日期") or "请删除" in cell(r, "日期"):
            errors.append(f"第 {i} 行：这是模板里的示例行，已跳过（填真实数据时删掉它即可）")
            continue
        desc = cell(r, "费用说明")
        if not desc:
            errors.append(f"第 {i} 行：费用说明为空，已跳过")
            continue
        amt_raw = cell(r, "不含税金额").replace(",", "").replace("¥", "").replace("￥", "")
        tax_raw = cell(r, "税额").replace(",", "").replace("¥", "").replace("￥", "")
        try:
            no_tax = float(amt_raw) if amt_raw else 0.0
        except ValueError:
            errors.append(f"第 {i} 行：金额「{amt_raw}」不是数字，已跳过")
            continue
        try:
            tax = float(tax_raw) if tax_raw else 0.0
        except ValueError:
            tax = 0.0
        inv = _norm_inv(cell(r, "发票类型"))
        rows.append({
            "line": i,
            "date": cell(r, "日期"),
            "voucher": cell(r, "凭证号"),
            "description": desc,
            "no_tax": no_tax,
            "tax": tax,
            "invoice_type": inv,
            "project": cell(r, "所属项目"),
            "amount": compute_amount(no_tax, tax, inv),
        })
    return {"rows": rows, "errors": errors}


def read_file(path: str) -> dict:
    """读文件 + 解析，一步到位。"""
    try:
        raw = read_rows(path)
    except Exception as e:
        return {"rows": [], "errors": [f"读取失败：{e}"]}
    return parse_rows(raw)


# 模板里示例行的日期列写上这个标记。
# 解析器看到它就直接跳过 —— 这样即使财务忘了删示例，也不会把示例混进辅助账。
# 为什么不干脆不放示例：新手需要看格式（列名、金额写法、发票类型的取值）。
TEMPLATE_MARK = "示例行，请删除"


def write_template_csv(path: str) -> None:
    """生成 CSV 模板。示例行的日期列带标记，导入时会被自动跳过。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerow([TEMPLATE_MARK, "ZP-001", "研发人员工资", "32000", "0", "无票", "A项目"])
        w.writerow([TEMPLATE_MARK, "ZP-002", "采购原材料A型", "20000", "2600", "专票", "A项目"])
        w.writerow([TEMPLATE_MARK, "ZP-004", "办公用品", "1000", "60", "普票", "A项目"])


def write_template_xlsx(path: str) -> None:
    """生成 xlsx 模板。用最小手写 OOXML 实现，不依赖 openpyxl。

    ⚠️ 只写一个 sheet、纯内联字符串，够用就行。
    """
    rows = [
        COLUMNS,
        [TEMPLATE_MARK, "ZP-001", "研发人员工资", "32000", "0", "无票", "A项目"],
        [TEMPLATE_MARK, "ZP-002", "采购原材料A型", "20000", "2600", "专票", "A项目"],
        [TEMPLATE_MARK, "ZP-004", "办公用品", "1000", "60", "普票", "A项目"],
    ]

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    def col_name(n):
        s = ""
        while n >= 0:
            s = chr(n % 26 + 65) + s
            n = n // 26 - 1
        return s

    sheet_rows = []
    for ri, r in enumerate(rows, start=1):
        cells = []
        for ci, v in enumerate(r):
            ref = f"{col_name(ci)}{ri}"
            if v == "" or v is None:
                continue
            if isinstance(v, (int, float)) or (str(v).replace(".", "").isdigit() and v != ""):
                cells.append(f'<c r="{ref}"><v>{esc(v)}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{esc(v)}</t></is></c>')
        sheet_rows.append(f'<row r="{ri}">' + "".join(cells) + "</row>")

    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             "<sheetData>" + "".join(sheet_rows) + "</sheetData></worksheet>")
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="费用明细" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>')
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                     '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                     "</Types>")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels",
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                    'Target="xl/workbook.xml"/></Relationships>')
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
