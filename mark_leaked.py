# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：给 eval.csv 补 leaked 列，并把「已入库」的题从计分中剔除
#   依赖：csv/csvio
#   输入输出：leaked=0 参与计分；leaked=1 只保留在题库
#   调试提示：某条工单的答案写进 cases.csv 后，把它标成 1
# ---------------------------------------------------------------------------
import sys

from csvio import read_rows

PATH = "data/eval.csv"
rows = read_rows(PATH)
fields = list(rows[0].keys()) if rows else []
if "leaked" not in fields:
    fields.append("leaked")
for r in rows:
    r.setdefault("leaked", "0")

if "--mark" in sys.argv:
    ids = set(sys.argv[sys.argv.index("--mark") + 1].split(","))
    for r in rows:
        if r["id"] in ids:
            r["leaked"] = "1"
    print("marked:", ",".join(sorted(ids)))

import csv
with open(PATH, "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

total = len([r for r in rows if r.get("symptom", "").strip()])
leaked = len([r for r in rows if r.get("symptom", "").strip() and r.get("leaked") == "1"])
print("eval rows:", total, "| leaked:", leaked, "| scoring:", total - leaked)
