#!/usr/bin/env python3
"""
Generate organized PageRank performance data in Excel format.
Data source: draw_pagerank.py — AtSpMV vs cuSPARSE PageRank performance comparison.
"""

import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill, numbers
from openpyxl.utils import get_column_letter
import os

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pagerank_performance_data.xlsx")

wb = openpyxl.Workbook()

# ── Style definitions ──────────────────────────────────────────────
header_font = Font(name="Times New Roman", size=12, bold=True, color="FFFFFF")
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
sub_header_fill = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
data_font = Font(name="Times New Roman", size=11)
title_font = Font(name="Times New Roman", size=14, bold=True)
thin_border = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)
center_align = Alignment(horizontal="center", vertical="center")
left_align = Alignment(horizontal="left", vertical="center")
speedup_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")


def style_header_row(ws, row, ncols):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = thin_border


def style_data_cell(ws, row, col, align="center"):
    cell = ws.cell(row=row, column=col)
    cell.font = data_font
    cell.alignment = center_align if align == "center" else left_align
    cell.border = thin_border
    return cell


# ══════════════════════════════════════════════════════════════════════
# Sheet 1: PageRank 性能对比 (主表)
# ══════════════════════════════════════════════════════════════════════
ws1 = wb.active
ws1.title = "PageRank性能对比"

# Raw data from draw_pagerank.py
matrix_names = [
    "cit-HepPh", "cit-HepTh", "ca-HepPh", "ca-HepTh",
    "web-BerkStan", "web-Google", "wiki-Vote", "wiki-RfA",
    "email-EuAll", "email-Enron",
]
atPre_us = [8689, 4887, 4869, 5049, 6332, 8848, 5451, 5127, 12438, 9716]
atIter_us = [1090, 1578, 2000, 2060, 14408, 11901, 1134, 1785, 2526, 2381]
cuspPre_us = [8052, 13116, 12064, 7944, 12946, 11092, 15777, 9843, 14796, 9404]
cuspIter_us = [7135, 10354, 11666, 8076, 13503, 16959, 11804, 7953, 10352, 8713]

# Compute derived values
at_total_ms = [(p + i) / 1000 for p, i in zip(atPre_us, atIter_us)]
cusp_total_ms = [(p + i) / 1000 for p, i in zip(cuspPre_us, cuspIter_us)]
at_setup_ms = [p / 1000 for p in atPre_us]
at_iter_ms = [i / 1000 for i in atIter_us]
cusp_setup_ms = [p / 1000 for p in cuspPre_us]
cusp_iter_ms = [i / 1000 for i in cuspIter_us]
speedup = [c / a for c, a in zip(cusp_total_ms, at_total_ms)]

# Graph metadata
graph_info = {
    "cit-HepPh":  ("Arxiv HEP-PH", "34,546", "421,578", "Citation network"),
    "cit-HepTh":  ("Arxiv HEP-TH", "27,770", "352,807", "Citation network"),
    "ca-HepPh":   ("Arxiv HEP-PH", "12,008", "237,010", "Collaboration network"),
    "ca-HepTh":   ("Arxiv HEP-TH", "9,877", "51,971", "Collaboration network"),
    "web-BerkStan": ("Berkeley-Stanford", "685,230", "7,600,595", "Web graph"),
    "web-Google":   ("Google", "875,713", "5,105,039", "Web graph"),
    "wiki-Vote":    ("Wikipedia", "7,115", "103,689", "Vote network"),
    "wiki-RfA":     ("Wikipedia", "10,835", "160,515", "Request for Adminship"),
    "email-EuAll":  ("EU research", "265,214", "420,045", "Email network"),
    "email-Enron":  ("Enron", "36,692", "367,662", "Email network"),
}

# Title row
ws1.merge_cells("A1:N1")
ws1.cell(row=1, column=1, value="PageRank 性能对比：AtSpMV vs cuSPARSE（GPU）").font = title_font
ws1.cell(row=1, column=1).alignment = center_align

# Subtitle
ws1.merge_cells("A2:N2")
ws1.cell(row=2, column=1, value="测试平台：NVIDIA GPU | 阻尼因子 α=0.85 | 收敛阈值 ε=1e-6 | 时间单位：ms（毫秒）").font = Font(name="Times New Roman", size=10, italic=True)
ws1.cell(row=2, column=1).alignment = center_align

# Headers (row 4)
headers = [
    "图数据集",
    "类型",
    "节点数",
    "边数",
    "AtSpMV\nSetup (ms)",
    "AtSpMV\nIter (ms)",
    "AtSpMV\nTotal (ms)",
    "cuSPARSE\nSetup (ms)",
    "cuSPARSE\nIter (ms)",
    "cuSPARSE\nTotal (ms)",
    "加速比\n(cuSPARSE/AtSpMV)",
    "AtSpMV Setup\n原始 (μs)",
    "AtSpMV Iter\n原始 (μs)",
    "cuSPARSE Setup\n原始 (μs)",
    "cuSPARSE Iter\n原始 (μs)",
]
# Use two rows for readability - row 3 as group header, row 4 as sub-header

# Row 3: category headers
# Actually let me simplify - just use row 3 for headers with line breaks
for col_idx, h in enumerate(headers, 1):
    cell = ws1.cell(row=4, column=col_idx, value=h)
style_header_row(ws1, 4, len(headers))

# Data rows
for i in range(10):
    row = 5 + i
    info = graph_info.get(matrix_names[i], ("-", "-", "-", "-"))
    values = [
        matrix_names[i],
        info[3],
        info[1],
        info[2],
        round(at_setup_ms[i], 3),
        round(at_iter_ms[i], 3),
        round(at_total_ms[i], 3),
        round(cusp_setup_ms[i], 3),
        round(cusp_iter_ms[i], 3),
        round(cusp_total_ms[i], 3),
        round(speedup[i], 2),
        atPre_us[i],
        atIter_us[i],
        cuspPre_us[i],
        cuspIter_us[i],
    ]
    for col_idx, v in enumerate(values, 1):
        cell = style_data_cell(ws1, row, col_idx, align="left" if col_idx <= 4 else "center")
        cell.value = v

# Highlight speedup column
for i in range(10):
    row = 5 + i
    cell = ws1.cell(row=row, column=11)
    cell.fill = speedup_fill
    cell.font = Font(name="Times New Roman", size=11, bold=True)

# Summary row
summary_row = 15
ws1.merge_cells(f"A{summary_row}:D{summary_row}")
ws1.cell(row=summary_row, column=1, value="平均加速比").font = Font(name="Times New Roman", size=11, bold=True)
ws1.cell(row=summary_row, column=1).alignment = center_align
ws1.cell(row=summary_row, column=1).border = thin_border
for c in range(2, 5):
    style_data_cell(ws1, summary_row, c)

avg_speedup = sum(speedup) / len(speedup)
ws1.cell(row=summary_row, column=11, value=round(avg_speedup, 2)).font = Font(name="Times New Roman", size=11, bold=True, color="CC0000")
ws1.cell(row=summary_row, column=11).alignment = center_align
ws1.cell(row=summary_row, column=11).border = thin_border
ws1.cell(row=summary_row, column=11).fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

# Column widths
col_widths = [16, 20, 12, 12, 14, 14, 14, 14, 14, 14, 14, 16, 16, 18, 18]
for idx, w in enumerate(col_widths):
    ws1.column_dimensions[get_column_letter(idx + 1)].width = w

# ══════════════════════════════════════════════════════════════════════
# Sheet 2: 加速比汇总
# ══════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("加速比汇总")

ws2.merge_cells("A1:F1")
ws2.cell(row=1, column=1, value="PageRank 加速比汇总（AtSpMV vs cuSPARSE）").font = title_font
ws2.cell(row=1, column=1).alignment = center_align

ws2.merge_cells("A2:F2")
ws2.cell(row=2, column=1, value="加速比 = cuSPARSE总时间 / AtSpMV总时间，数值越大表示 AtSpMV 性能优势越明显").font = Font(name="Times New Roman", size=10, italic=True)
ws2.cell(row=2, column=1).alignment = center_align

headers2 = [
    "排名", "图数据集", "AtSpMV Total (ms)", "cuSPARSE Total (ms)", "加速比", "性能优势"
]
for col_idx, h in enumerate(headers2, 1):
    ws2.cell(row=4, column=col_idx, value=h)
style_header_row(ws2, 4, len(headers2))

# Sort by speedup descending
sorted_idx = sorted(range(10), key=lambda i: speedup[i], reverse=True)

for rank, i in enumerate(sorted_idx):
    row = 5 + rank
    sp = speedup[i]
    if sp >= 3.0:
        advantage = "显著提升 (≥3x)"
    elif sp >= 2.0:
        advantage = "大幅提升 (2-3x)"
    elif sp >= 1.5:
        advantage = "明显提升 (1.5-2x)"
    else:
        advantage = "有所提升 (1-1.5x)"

    values = [
        rank + 1,
        matrix_names[i],
        round(at_total_ms[i], 3),
        round(cusp_total_ms[i], 3),
        round(sp, 2),
        advantage,
    ]
    for col_idx, v in enumerate(values, 1):
        cell = style_data_cell(ws2, row, col_idx, align="left" if col_idx in (2, 6) else "center")
        cell.value = v
    # Highlight speedup cell
    ws2.cell(row=row, column=5).fill = speedup_fill
    ws2.cell(row=row, column=5).font = Font(name="Times New Roman", size=11, bold=True)

# Stats summary
sr = 5 + 10 + 1  # summary start row
ws2.merge_cells(f"A{sr}:B{sr}")
ws2.cell(row=sr, column=1, value="统计摘要").font = Font(name="Times New Roman", size=12, bold=True)
ws2.cell(row=sr, column=1).fill = sub_header_fill
for c in range(1, 7):
    ws2.cell(row=sr, column=c).border = thin_border

stats = [
    ("最大加速比", max(speedup), matrix_names[speedup.index(max(speedup))]),
    ("最小加速比", min(speedup), matrix_names[speedup.index(min(speedup))]),
    ("平均加速比", avg_speedup, "-"),
    ("中位数加速比", sorted(speedup)[len(speedup) // 2], "-"),
]
for j, (label, val, best_matrix) in enumerate(stats):
    row = sr + 1 + j
    style_data_cell(ws2, row, 1, "left").value = label
    ws2.cell(row=row, column=1).font = Font(name="Times New Roman", size=11, bold=True)
    style_data_cell(ws2, row, 2).value = round(val, 2)
    ws2.cell(row=row, column=2).font = Font(name="Times New Roman", size=11, bold=True, color="CC0000")
    style_data_cell(ws2, row, 3).value = best_matrix

col_widths2 = [8, 18, 20, 22, 12, 22]
for idx, w in enumerate(col_widths2):
    ws2.column_dimensions[get_column_letter(idx + 1)].width = w

# ══════════════════════════════════════════════════════════════════════
# Sheet 3: 图数据集详情
# ══════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("图数据集信息")

ws3.merge_cells("A1:G1")
ws3.cell(row=1, column=1, value="PageRank 测试用图数据集详细信息").font = title_font
ws3.cell(row=1, column=1).alignment = center_align

ws3.merge_cells("A2:G2")
ws3.cell(row=2, column=1, value="数据来源：Stanford Network Analysis Project (SNAP) / SuiteSparse Matrix Collection").font = Font(name="Times New Roman", size=10, italic=True)
ws3.cell(row=2, column=1).alignment = center_align

headers3 = [
    "图数据集", "全称/描述", "节点数", "边数", "类型", "来源", "平均度数"
]
for col_idx, h in enumerate(headers3, 1):
    ws3.cell(row=4, column=col_idx, value=h)
style_header_row(ws3, 4, len(headers3))

for i in range(10):
    row = 5 + i
    info = graph_info.get(matrix_names[i], ("-", "-", "-", "-"))
    nodes = int(info[1].replace(",", ""))
    edges = int(info[2].replace(",", ""))
    avg_deg = round(edges / nodes, 2) if nodes > 0 else 0

    values = [
        matrix_names[i],
        info[0],
        info[1],
        info[2],
        info[3],
        "SNAP" if any(k in matrix_names[i] for k in ["cit-", "ca-", "web-", "wiki-", "email-"]) else "SuiteSparse",
        avg_deg,
    ]
    for col_idx, v in enumerate(values, 1):
        cell = style_data_cell(ws3, row, col_idx, align="left" if col_idx <= 2 or col_idx == 5 else "center")
        cell.value = v

col_widths3 = [16, 28, 14, 14, 24, 16, 12]
for idx, w in enumerate(col_widths3):
    ws3.column_dimensions[get_column_letter(idx + 1)].width = w

# ══════════════════════════════════════════════════════════════════════
# Sheet 4: 测试平台信息
# ══════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("测试平台与参数")

ws4.merge_cells("A1:B1")
ws4.cell(row=1, column=1, value="测试平台配置与算法参数").font = title_font
ws4.cell(row=1, column=1).alignment = center_align

platform_data = [
    ("GPU", "NVIDIA GPU (CUDA)"),
    ("CUDA 版本", "—"),
    ("PageRank 阻尼因子 α", "0.85"),
    ("收敛阈值 ε", "1e-6"),
    ("最大迭代次数", "—"),
    ("对比方法", "cuSPARSE (cusparseDcsrmv)"),
    ("本文方法", "AtSpMV (Adaptive Tiled SpMV)"),
    ("AtSpMV 技术特点", "自适应分块 + 负载均衡 + 访存优化"),
    ("测试矩阵来源", "SuiteSparse Matrix Collection"),
    ("数据单位 (原始)", "微秒 (μs)"),
    ("数据单位 (表中)", "毫秒 (ms)，原始值÷1000"),
    ("加速比定义", "cuSPARSE总耗时 / AtSpMV总耗时"),
    ("图表文件", "pagerank.pdf / pagerank.svg"),
    ("绘图脚本", "draw_pagerank.py"),
]

for i, (label, value) in enumerate(platform_data):
    row = 4 + i
    cell_l = style_data_cell(ws4, row, 1, "left")
    cell_l.value = label
    cell_l.font = Font(name="Times New Roman", size=11, bold=True)
    cell_v = style_data_cell(ws4, row, 2, "left")
    cell_v.value = value

ws4.column_dimensions["A"].width = 28
ws4.column_dimensions["B"].width = 50

# ══════════════════════════════════════════════════════════════════════
# Sheet 5: SPMV底层性能数据（摘要）
# ══════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("SpMV底层性能(摘要)")

ws5.merge_cells("A1:H1")
ws5.cell(row=1, column=1, value="SpMV 底层算子性能数据摘要（与 PageRank 性能关联）").font = title_font
ws5.cell(row=1, column=1).alignment = center_align

ws5.merge_cells("A2:H2")
ws5.cell(row=2, column=1, value="PageRank 每次迭代的核心操作为 SpMV（稀疏矩阵-向量乘），此表列出 SpMV 预条件器性能关键指标").font = Font(name="Times New Roman", size=10, italic=True)
ws5.cell(row=2, column=1).alignment = center_align

# Read summary from gpu_pre_full_clean.csv
import csv
spmv_data = []
csv_path = OUTPUT_DIR / "spmv_performance/gpu_spmv_preconditioner_full.csv"
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        spmv_data.append(row)

headers5 = [
    "矩阵名称", "行数", "列数", "非零元数 (NNZ)",
    "cuSPARSE (ms)", "AtSpMV (ms)", "加速比 (cuSPARSE/AtSpMV)", "预条件器加速比",
]
for col_idx, h in enumerate(headers5, 1):
    ws5.cell(row=4, column=col_idx, value=h)
style_header_row(ws5, 4, len(headers5))

# Show top 30 rows as summary
show_count = min(30, len(spmv_data))
for i in range(show_count):
    row = 5 + i
    d = spmv_data[i]
    cusp_time = float(d.get("cuspTime", 0))
    at_time = float(d.get("atTime", 0))
    spmv_speedup = round(cusp_time / at_time, 2) if at_time > 0 else 0
    cusp_pre = float(d.get("cuspPreTime", 0))
    at_pre = float(d.get("atPreTime", 0))
    pre_speedup = round(cusp_pre / at_pre, 2) if at_pre > 0 else 0

    values = [
        d.get("matrix", "-"),
        int(float(d.get("csr.nrows", 0))),
        int(float(d.get("csr.ncols", 0))),
        int(float(d.get("csr.nnz", 0))),
        cusp_time,
        at_time,
        spmv_speedup,
        pre_speedup,
    ]
    for col_idx, v in enumerate(values, 1):
        cell = style_data_cell(ws5, row, col_idx, align="left" if col_idx == 1 else "center")
        cell.value = v

# Note about full data
note_row = 5 + show_count + 1
ws5.merge_cells(f"A{note_row}:H{note_row}")
ws5.cell(row=note_row, column=1,
         value=f"注：以上为前 {show_count} 行摘要，完整数据（{len(spmv_data)} 个矩阵）见 spmv_performance/gpu_spmv_preconditioner_full.csv")
ws5.cell(row=note_row, column=1).font = Font(name="Times New Roman", size=10, italic=True, color="666666")

col_widths5 = [22, 10, 10, 14, 16, 14, 18, 18]
for idx, w in enumerate(col_widths5):
    ws5.column_dimensions[get_column_letter(idx + 1)].width = w

# ── Save ───────────────────────────────────────────────────────────
wb.save(OUTPUT_FILE)
print(f"Excel file saved to: {OUTPUT_FILE}")
print(f"Sheets: {wb.sheetnames}")
