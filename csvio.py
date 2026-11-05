# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：CSV 读取的编码兜底（Excel 可能把文件存成 GBK）
#   依赖：csv
#   输入输出：read_rows(path) -> list[dict]；必要时把文件转写成 UTF-8-SIG
#   调试提示：用户用 Excel 编辑后，文件编码会变，统一走这个函数读
# ---------------------------------------------------------------------------
import csv

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def read_rows(path: str):
    """按多个编码尝试读取 CSV；若用了 GBK 系编码，顺手把文件转回 UTF-8-SIG。"""
    last_err = None
    for enc in ENCODINGS:
        try:
            with open(path, encoding=enc, newline="") as fh:
                rows = list(csv.DictReader(fh))
            if enc != "utf-8-sig":
                _rewrite(path, rows)
            return rows
        except UnicodeDecodeError as exc:
            last_err = exc
    raise last_err


def _rewrite(path: str, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
