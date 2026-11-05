# -*- coding: utf-8 -*-
"""Make category consistent per fault code (it was computed per cause row)."""
import csv
from collections import Counter, defaultdict

from kb_category import guess_category

SRC = "data/rules.csv"
rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))

by_code = defaultdict(list)
for r in rows:
    by_code[r["code"]].append(r)

changed = 0
for code, rs in by_code.items():
    # 用代码名判定，名称比原因文本稳定；EE 系列强制通讯
    cat = "通讯与总线" if code.upper().startswith("EE") else guess_category(rs[0]["name"])
    if len({r["category"] for r in rs}) > 1:
        changed += 1
    for r in rs:
        r["category"] = cat

fields = ["code", "name", "category", "domain", "level", "resettable", "error_code",
          "cause", "confirm", "action", "verify", "closed_loop", "page"]
with open(SRC, "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print("codes with inconsistent category fixed:", changed)
print("category:", dict(Counter(r["category"] for r in rows)))
