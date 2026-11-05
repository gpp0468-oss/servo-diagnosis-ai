# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：检查评测集与知识库是否重叠（防数据泄漏）
#   依赖：csvio
#   输入输出：报告重合条目；--auto 自动把重合项标记 leaked=1
#   调试提示：现象或根因关键词任一重合即视为泄露，宁可误报
# ---------------------------------------------------------------------------
import csv
import re
import sys

from csvio import read_rows

PUNCT = re.compile(r"[\s，。；、：（）()【】\[\]「」“”\"'.,;:!?\-]+")


def norm(s):
    return PUNCT.sub("", (s or "").lower())


def main():
    ev = [r for r in read_rows("data/eval.csv") if r.get("symptom", "").strip()]
    cases = read_rows("data/cases.csv")

    hits = []
    for r in ev:
        sym, kw = norm(r["symptom"]), norm(r.get("expected_cause_keywords", ""))
        for c in cases:
            ph, ca = norm(c["phenomenon"]), norm(c["cause"])
            if sym and ph and (sym in ph or ph in sym):
                hits.append((r["id"], "现象重合", c["phenomenon"][:40], r))
                break
            if kw and len(kw) >= 6 and kw in ca:
                hits.append((r["id"], "根因关键词重合", c["cause"][:40], r))
                break

    print("eval:", len(ev), "| cases:", len(cases), "| overlap:", len(hits))
    for h in hits:
        print("  %s  %s  %s" % (h[0], h[1], h[2]))

    if "--auto" in sys.argv and hits:
        ids = {h[0] for h in hits}
        rows = read_rows("data/eval.csv")
        fields = list(rows[0].keys())
        if "leaked" not in fields:
            fields.append("leaked")
        for r in rows:
            r.setdefault("leaked", "0")
            if r["id"] in ids:
                r["leaked"] = "1"
        with open("data/eval.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("auto-marked leaked:", ",".join(sorted(ids)))


if __name__ == "__main__":
    main()
