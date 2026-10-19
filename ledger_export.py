# ledger_export.py
"""把辅助账导出成 Excel。

━━━ 为什么要导出 ━━━
政策没有规定辅助账必须是什么格式，只说「按研发项目设置辅助账，留存备查」
（财税〔2015〕119 号、国家税务总局公告 2015 年第 97 号）。
国家税务总局公告 2019 年第 41 号 又进一步取消了《研发支出辅助账汇总表》的报送，
全部改为留存备查。

所以导出的意义不是「满足某个格式要求」，而是：
  1. 留存备查要能拿得出来，稽查时企业得提供
  2. 政策要求按项目归集，导出后能按项目拆开看
  3. 中小企业一般没有专门的研发费用辅助账软件
  4. 给代账或客户时，对方不用登录系统就能用

━━━ 为什么自己写 xlsx ━━━
项目虚拟环境里没有 openpyxl，pip 装不上（网络受限）。
xlsx 本质是 zip + XML，写简单的二维表用标准库就够。
"""
import io
import os
import zipfile

from xml.sax.saxutils import escape


def _col_name(n: int) -> str:
    """0 基列号转 Excel 列名：0 -> A, 25 -> Z, 26 -> AA。"""
    s = ""
    while n >= 0:
        s = chr(n % 26 + 65) + s
        n = n // 26 - 1
    return s


def _is_number(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    try:
        float(str(v).replace(",", ""))
        return str(v).strip() != ""
    except (TypeError, ValueError):
        return False


def _cell(ref: str, v) -> str:
    """生成一个单元格。数字写成数值型，其余写成内联字符串。"""
    if v is None or v == "":
        return ""
    if _is_number(v):
        try:
            num = float(str(v).replace(",", ""))
            # 整数不要写成 6.0：Excel 会把 6.0 显示成 6，但读回来是 6.0，
            # 序号、笔数这类字段显示成 1.0 很难看，所以整数直接写整数。
            if num == int(num) and abs(num) < 1e15:
                return f'<c r="{ref}"><v>{int(num)}</v></c>'
            return f'<c r="{ref}"><v>{num}</v></c>'
        except ValueError:
            pass
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'


def _sheet_xml(rows: list, col_widths: list = None) -> str:
    xml_rows = []
    for ri, r in enumerate(rows, start=1):
        cells = "".join(_cell(f"{_col_name(ci)}{ri}", v) for ci, v in enumerate(r))
        xml_rows.append(f'<row r="{ri}">{cells}</row>')
    cols = ""
    if col_widths:
        cols = "<cols>" + "".join(
            f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>'
            for i, w in enumerate(col_widths)) + "</cols>"
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + cols + "<sheetData>" + "".join(xml_rows) + "</sheetData></worksheet>")


def _write_xlsx(path: str, sheet_name: str, rows: list, col_widths: list = None) -> None:
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="' + escape(sheet_name[:31]) + '" sheetId="1" r:id="rId1"/>'
                '</sheets></workbook>')
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
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" '
                 'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                 'Target="xl/workbook.xml"/></Relationships>')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        zf.writestr("xl/worksheets/sheet1.xml", _sheet_xml(rows, col_widths))


def build_ledger_rows(state: dict) -> list:
    """把看板状态里的辅助账数据摊平成表格行。"""
    L = state.get("ledger") or {}
    sm = L.get("summary") or {}
    period = L.get("period") or ""

    rows = []
    rows.append(["研发支出辅助账", "", "", "", "", ""])
    rows.append(["所属期", period, "", "编制依据",
                 "财税〔2015〕119 号；国家税务总局公告 2015 年第 97 号", ""])
    rows.append(["说明", "本表为研发费用辅助账，按项目归集，用于留存备查", "", "", "", ""])
    rows.append([])

    # 一、明细（按项目分组）
    rows.append(["一、费用明细（按项目归集）", "", "", "", "", ""])
    rows.append(["研发项目", "日期", "凭证号", "费用类别", "金额（元）", "费用说明"])

    by_proj = {}
    for exp in (state.get("expenses") or []):
        if exp.get("pending"):
            continue
        proj = exp.get("project") or "未指定项目"
        by_proj.setdefault(proj, []).append(exp)

    subtotal = 0.0
    for proj in sorted(by_proj.keys()):
        items = by_proj[proj]
        proj_sum = 0.0
        for e in items:
            amt = float(e.get("amount") or 0)
            proj_sum += amt
            rows.append([proj, e.get("date", ""), e.get("voucher", ""),
                         e.get("category", ""), round(amt, 2), e.get("description", "")])
        rows.append([proj + " 小计", "", "", "", round(proj_sum, 2), ""])
        subtotal += proj_sum
    rows.append(["合计", "", "", "", round(subtotal, 2), ""])
    rows.append([])

    # 二、按类别汇总
    rows.append(["二、按费用类别汇总", "", "", "", "", ""])
    rows.append(["费用类别", "笔数", "", "", "金额（元）", ""])
    for cat, v in (L.get("by_category") or {}).items():
        amt = float((v or {}).get("amount") or 0)
        if amt == 0:
            continue
        rows.append([cat, (v or {}).get("count", 0), "", "", round(amt, 2), ""])
    rows.append(["合计", "", "", "", round(float(sm.get("total") or 0), 2), ""])
    rows.append([])

    # 三、限额校验
    rows.append(["三、其他相关费用限额校验（加计扣除口径）", "", "", "", "", ""])
    rows.append(["其他相关费用", "", "", "", round(float(sm.get("other_expense") or 0), 2), ""])
    rows.append(["占研发费用总额比例",
                 f"{float(sm.get('other_ratio') or 0) * 100:.1f}%", "", "", "", ""])
    rows.append(["可计入上限（前五项 ÷ 9）", "", "", "",
                 round(float(sm.get("max_other_allowed") or 0), 2), ""])
    rows.append(["结论", "未超限" if sm.get("compliant") else
                 f"超出 {float(sm.get('over_limit') or 0):.2f}，超出部分需剔除", "", "", "", ""])
    rows.append([])

    # 四、工时分摊
    rows.append(["四、人员人工费用按工时占比分摊", "", "", "", "", ""])
    rows.append(["研发项目", "分摊金额（元）", "", "", "", ""])
    alloc = sm.get("allocations") or {}
    if alloc:
        for proj, amt in alloc.items():
            rows.append([proj, round(float(amt), 2), "", "", "", ""])
    else:
        rows.append(["本期未涉及跨项目分摊", "", "", "", "", ""])
    rows.append([])

    # 五、加计扣除测算
    rows.append(["五、加计扣除测算", "", "", "", "", ""])
    tax = state.get("tax") or {}
    rows.append(["可加计扣除的研发费用", "", "", "", round(float(tax.get("deduct_base") or 0), 2), ""])
    rows.append(["加计扣除比例", "100%", "", "", "", ""])
    rows.append(["加计扣除额", "", "", "", round(float(tax.get("deduct_amount") or 0), 2), ""])
    rows.append(["适用所得税税率", "25%", "", "", "", ""])
    rows.append(["测算少缴税额", "", "", "", round(float(tax.get("saved") or 0), 2),
                 "仅为测算，需结合当期应纳税所得额是否足够抵扣"])
    rows.append([])

    # 六、留存备查
    rows.append(["六、留存备查资料清单", "", "", "", "", ""])
    rows.append(["序号", "资料名称", "状态", "", "", "说明"])
    for c in (L.get("checklist") or []):
        rows.append([c.get("no", ""), c.get("name", ""), c.get("status", ""),
                     "", "", c.get("note", "")])
    rows.append([])

    # 七、未计入的
    skipped = L.get("skipped") or []
    if skipped:
        rows.append(["七、待人工复核（尚未计入本辅助账）", "", "", "", "", ""])
        rows.append(["费用说明", "", "", "", "金额（元）", ""])
        for s in skipped:
            rows.append([s.get("description", ""), "", "", "",
                         round(float(s.get("amount") or 0), 2), ""])
        rows.append([])

    rows.append(["编制说明", "本表由系统根据归集数据自动生成，"
                 "最终口径以企业财务账簿和主管税务机关认定为准", "", "", "", ""])
    return rows


def export_ledger(state: dict, path: str) -> str:
    """生成辅助账 Excel，返回文件路径。"""
    rows = build_ledger_rows(state)
    period = (state.get("ledger") or {}).get("period") or "辅助账"
    _write_xlsx(path, f"研发支出辅助账{period}", rows,
                col_widths=[22, 14, 16, 18, 14, 40])
    return path
