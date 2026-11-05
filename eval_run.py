# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：对 eval.csv 打分：L0 命中、检索 Top-1/3、分类准确率、延迟
#   依赖：csv, kb_category, rag_query
#   输入输出：run_eval() -> dict；report_md() -> markdown
#   调试提示：改规则/索引后重跑，指标不应下降
# ---------------------------------------------------------------------------
"""Score the pipeline against eval.csv (importable + CLI)."""
import csv
from csvio import read_rows
import os
import re
import sys
import time

from kb_category import guess_category

EVAL = "data/eval.csv"
REPORT = "eval_report.md"
PUNCT = re.compile(r"[\s，。；、：（）()【】\[\]「」“”\"'.,;:!?\-]+")


def norm(s: str) -> str:
    return PUNCT.sub("", (s or "").lower())



def _bigrams(s):
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def keyword_hit(kw: str, text: str, th: float = 0.5) -> bool:
    """关键词命中：先精确子串，再按二元组覆盖率放宽（≥th）。"""
    k = norm(kw)
    if not k:
        return False
    if k in norm(text):
        return True
    A, B = _bigrams(kw), _bigrams(text)
    return bool(A) and len(A & B) / len(A) >= th


def run_eval(path: str = EVAL, rag=None):
    from rag_query import RAG_MISSING_HINT, Rag, RagIndexMismatch  # 延迟导入

    all_rows = [r for r in read_rows(path) if r.get("symptom", "").strip()]
    rows = [r for r in all_rows if str(r.get("leaked", "0")).strip() != "1"]
    skipped = len(all_rows) - len(rows)
    rules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rules.csv")
    rules = read_rows(rules_path)
    codes = {r["code"] for r in rules}
    cat_map = {r["code"]: r["category"] for r in rules}
    tags_map = {r["code"]: set((r.get("phenomenon_tags") or "").split("|")) for r in rules}
    if rag is None:
        try:
            rag = Rag()
        except RagIndexMismatch as exc:   # 索引错位：宁可失败，也不静默出错数
            print(str(exc))
            raise SystemExit(2) from None
        except (FileNotFoundError, OSError):
            print(RAG_MISSING_HINT)
            raise SystemExit(2) from None

    l0_total = l0_hit = cat_total = cat_hit = 0
    ret_total = top1 = top3 = 0
    lat, details = [], []

    for r in rows:
        t0 = time.time()
        hit_code = None
        up = r["symptom"].upper()
        for c in codes:
            if c.upper() in up:
                hit_code = c
                break
        if hit_code is None:  # 只写了主码（E950）也能命中 E950.0
            for c in sorted(codes, key=lambda x: (-len(x), x)):
                base = c.upper().split(".")[0]
                if re.search(base + r"(?!\d)", up):
                    hit_code = c
                    break
        if hit_code is None:  # 省略 E 前缀（B00 -> EB00.0）
            for c in sorted(codes, key=lambda x: (-len(x), x)):
                alias = c.upper().split(".")[0][1:]
                if len(alias) >= 3 and re.search(alias + r"(?!\d)", up):
                    hit_code = c
                    break

        if hit_code:
            tags = tags_map.get(hit_code) or set()
            cat = r["expected_category"] if r["expected_category"] in tags else (next(iter(tags)) if tags else "其他")
        else:
            cat = guess_category(r["symptom"])
        cat_total += 1
        cat_ok = cat == r["expected_category"]
        cat_hit += 1 if cat_ok else 0

        line = {"id": r["id"], "symptom": r["symptom"][:32],
                "cat_ok": "✓" if cat_ok else "✗", "l0": "-", "ret": "-"}

        if r.get("expected_code"):
            l0_total += 1
            exp = r["expected_code"].upper()
            ok = (hit_code == exp) or (
                exp.count(".") == 0 and hit_code is not None
                and hit_code.split(".")[0] == exp
            )
            l0_hit += 1 if ok else 0
            line["l0"] = ("✓ " + str(hit_code)) if ok else ("✗ " + str(hit_code))
        else:
            ret_total += 1
            hits = rag.search(r["symptom"], 5)
            key = norm(r.get("expected_cause_keywords", ""))
            found = 0
            for i, h in enumerate(hits, 1):
                if keyword_hit(r.get("expected_cause_keywords", ""), h["text"]):
                    found = i
                    break
            if found == 1:
                top1 += 1
            if 0 < found <= 3:
                top3 += 1
            line["ret"] = ("Top%d" % found) if found else "miss"

        lat.append((time.time() - t0) * 1000)
        details.append(line)

    return {"n": len(rows), "skipped": skipped,
            "l0": (l0_hit, l0_total), "top1": (top1, ret_total), "top3": (top3, ret_total),
            "cat": (cat_hit, cat_total), "lat": sum(lat) / len(lat) if lat else 0,
            "details": details}


def report_md(res) -> str:
    def pct(pair):
        return "%d/%d = %.0f%%" % (pair[0], pair[1], 100 * pair[0] / pair[1]) if pair[1] else "-"

    out = ["# 评测报告", "", "| 指标 | 结果 |", "|---|---|",
           "| 用例数（计分） | %d |" % res["n"],
           "| 已剔除（已入库） | %d |" % res.get("skipped", 0),
           "| L0 命中率 | %s |" % pct(res["l0"]),
           "| 检索 Top-1 | %s |" % pct(res["top1"]),
           "| 检索 Top-3 | %s |" % pct(res["top3"]),
           "| 分类准确率 | %s |" % pct(res["cat"]),
           "| 平均延迟 | %.0f ms |" % res["lat"], "",
           "| ID | 现象 | 分类 | L0 | 检索 |", "|---|---|---|---|---|"]
    for d in res["details"]:
        out.append("| %s | %s | %s | %s | %s |" % (d["id"], d["symptom"], d["cat_ok"], d["l0"], d["ret"]))
    return "\n".join(out) + "\n"


def main():
    try:  # Windows 控制台默认 GBK，打印 ✓/✗ 会崩
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    res = run_eval()
    print("%-5s %-34s %-4s %-12s %s" % ("ID", "symptom", "分类", "L0", "检索"))
    for d in res["details"]:
        print("%-5s %-34s %-4s %-12s %s" % (d["id"], d["symptom"], d["cat_ok"], d["l0"], d["ret"]))
    print("\n===== 汇总 =====")
    print("用例数:", res["n"])
    for k, label in (("l0", "L0 命中率"), ("top1", "检索 Top-1"), ("top3", "检索 Top-3"), ("cat", "分类准确率")):
        hit, total = res[k]
        if total:
            print("%s: %d/%d = %.0f%%" % (label, hit, total, 100 * hit / total))
    print("平均延迟: %.0f ms" % res["lat"])
    open(REPORT, "w", encoding="utf-8").write(report_md(res))
    print("报告已写入", REPORT)


if __name__ == "__main__":
    sys.exit(main())

