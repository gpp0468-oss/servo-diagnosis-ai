# -*- coding: utf-8 -*-
"""置信度审计：检索分数到底有没有"知道自己错了"的能力？

背景：本系统原本没有弃权通道，任何输入都会输出 Top-1，所以过度自信率
按定义就是 100%。本脚本回答真问题：

    如果加一个分数阈值来弃权，能拦住多少错、会误杀多少对？

结论（见 docs/09-置信度审计.md）：分数分布与对错几乎无关，阈值无效。

用法：
    python confidence_audit.py
需要先构建检索索引（见 README「完整评测」一节）。
"""
import sys

from csvio import read_rows
from eval_run import keyword_hit
from rag_query import RAG_MISSING_HINT, Rag, RagIndexMismatch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    try:
        rag = Rag()
    except RagIndexMismatch as exc:   # P0-8：索引错位时必须失败，不能输出错误结论
        print(str(exc))
        return 2
    except (FileNotFoundError, OSError):
        print(RAG_MISSING_HINT)
        return 2

    rows = [r for r in read_rows("data/eval.csv")
            if r.get("symptom", "").strip() and str(r.get("leaked", "0")).strip() != "1"]
    ret = [r for r in rows if not r.get("expected_code")]

    recs = []
    for r in ret:
        hits = rag.search(r["symptom"], 5)
        if not hits:
            recs.append({"id": r["id"], "ok": False, "score": 0.0, "rank": 0})
            continue
        found = 0
        for i, h in enumerate(hits, 1):
            if keyword_hit(r.get("expected_cause_keywords", ""), h["text"]):
                found = i
                break
        recs.append({"id": r["id"], "ok": found == 1, "score": hits[0]["score"],
                     "rank": found, "top": hits[0]["text"][:26]})

    print("检索层用例 %d 条（带标准答案）\n" % len(recs))
    print("%-5s %-4s %7s %6s  %s" % ("ID", "对错", "top1分", "命中位", "Top1 片段"))
    for x in recs:
        print("%-5s %-4s %7.4f %6s  %s"
              % (x["id"], "✓" if x["ok"] else "✗", x["score"],
                 x.get("rank") or "-", x.get("top", "")))

    ok = [x for x in recs if x["ok"]]
    bad = [x for x in recs if not x["ok"]]
    print("\n答对 %d 条，答错 %d 条" % (len(ok), len(bad)))
    if ok:
        print("  答对时 top1 分数：min %.4f  中位 %.4f  max %.4f"
              % (min(x["score"] for x in ok),
                 sorted(x["score"] for x in ok)[len(ok) // 2],
                 max(x["score"] for x in ok)))
    if bad:
        print("  答错时 top1 分数：min %.4f  中位 %.4f  max %.4f"
              % (min(x["score"] for x in bad),
                 sorted(x["score"] for x in bad)[len(bad) // 2],
                 max(x["score"] for x in bad)))

    print("\n=== 阈值扫描：低于阈值就弃权 ===")
    print("%8s %8s %10s %10s %10s" % ("阈值", "弃权率", "错案拦截", "对案误杀", "过度自信率"))
    for t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        abst = [x for x in recs if x["score"] < t]
        caught = [x for x in bad if x["score"] < t]
        killed = [x for x in ok if x["score"] < t]
        still_bad = len(bad) - len(caught)
        overconf = (100.0 * still_bad / len(bad)) if bad else 0.0
        print("%8.2f %7.0f%% %7d/%-2d %7d/%-2d %9.0f%%"
              % (t, 100.0 * len(abst) / len(recs),
                 len(caught), len(bad), len(killed), len(ok), overconf))

    print("\n注：现状无弃权通道 → 过度自信率恒为 100%（答错也照样出结论）")
    print("注：样本仅 %d 条，阈值结论不可当定论，只用于说明方法。" % len(recs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
