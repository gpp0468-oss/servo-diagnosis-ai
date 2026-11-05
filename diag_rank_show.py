# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 模块说明
#   职责：诊断排序的体检报告——先查数据问题，再逐码对比新旧策略
#         （单码排序表的打印已并入 diag_rank.show_ranking，这里不重复实现）
#   依赖：diag_rank, csvio
#   输入输出：只读 rules.csv，不写文件
#   用法：
#     python diag_rank_show.py            # 体检 + 逐码对比摘要
#     python diag_rank_show.py --all      # 再打印所有多假设码的排序表
#     python diag_rank_show.py E620.0     # 只看一个码的排序表
#   调试提示：如果「收敛率」很低，说明似然太平（判据写得太泛），
#             此时任何排序结论都不能信，先去补判据。
# ---------------------------------------------------------------------------
import argparse
import statistics
import sys

import diag_rank as dr
from csvio import read_rows

DEFAULT_CSV = "data/rules.csv"


def load(path=DEFAULT_CSV):
    by = {}
    for r in read_rows(path):
        by.setdefault(r["code"], []).append({
            "假设": r["cause"] or "",
            "判据": r["confirm"] or "",
            "verify": r["verify"] or "",
            "check_type": r["check_type"] or r["verify"] or "观察",
        })
    return by


def dedupe(hyps):
    seen, out = set(), []
    for h in hyps:
        if h["假设"] in seen:
            continue
        seen.add(h["假设"])
        out.append(h)
    return out


def audit(by):
    print("=== 数据体检 ===")
    dup = [(c, len(h), len(dedupe(h))) for c, h in by.items() if len(h) != len(dedupe(h))]
    print("同码重复假设：%s" % ("，".join("%s %d→%d" % t for t in dup) if dup else "无"))

    no_confirm = [c for c, h in by.items() if any(not x["判据"] for x in h)]
    print("含空判据的码：%d 个 %s"
          % (len(no_confirm), "—— " + " / ".join(no_confirm[:8]) + ("..." if len(no_confirm) > 8 else "")))

    fell_back = [c for c, h in by.items() if any(not x["check_type"] for x in h)]
    print("check_type 缺失回退 verify 的码：%s" % (len(fell_back) or "无"))

    maxh = max(len(dedupe(h)) for h in by.values())
    multi = [c for c, h in by.items() if len(dedupe(h)) >= 2]
    print("故障码 %d 个，多假设码 %d 个，单码假设数最多 %d 条" % (len(by), len(multi), maxh))

    # 判据的区分度到底值多少：加入相关性前后，首选建议变了的码有几个
    changed = 0
    for c, h in by.items():
        h = dedupe(h)
        if len(h) < 2:
            continue
        on = [x[3]["假设"] for x in dr.rank_checks(h, use_relevance=True)]
        off = [x[3]["假设"] for x in dr.rank_checks(h, use_relevance=False)]
        if on != off:
            changed += 1
    if multi:
        print("判据区分度改变首选建议的码：%d/%d（%.0f%%）"
              % (changed, len(multi), 100.0 * changed / len(multi)))
    print()


def main(argv):
    ap = argparse.ArgumentParser(description="诊断排序体检报告")
    ap.add_argument("code", nargs="?", help="只看这个故障码，例如 E620.0")
    ap.add_argument("--all", action="store_true", help="打印所有多假设码的排序表")
    ap.add_argument("--top", type=int, default=0, help="只打印排序表的前 N 个码，0=不打印")
    ap.add_argument("--trials", type=int, default=200, help="模拟每真值的重复次数")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="rules.csv 路径")
    args = ap.parse_args(argv)

    by = load(args.csv)

    if args.code:
        hit = [(c, h) for c, h in by.items() if c.upper() == args.code.upper()]
        if not hit:
            print("没有这个故障码：%s" % args.code)
            return 2
        for c, h in hit:
            dr.show_ranking(c, dedupe(h))
        return 0

    audit(by)

    multi = [(c, dedupe(h)) for c, h in by.items() if len(dedupe(h)) >= 2]
    multi.sort(key=lambda x: (-len(x[1]), x[0]))

    show_n = len(multi) if args.all else args.top
    for c, h in multi[:show_n]:
        dr.show_ranking(c, h)
    if show_n:
        print()

    print("=== 逐码对比（每真值 %d 次）===" % args.trials)
    rows = []
    for c, h in multi:
        D = dr.dep_matrix(h)
        old, old_c = dr._simulate(h, dr._policy_cost, D, args.trials)
        new, new_c = dr._simulate(h, dr._policy_ig, D, args.trials)
        rows.append((c, len(h), old, new, new - old, old_c, new_c))

    for c, n, old, new, delta, oc, nc in sorted(rows, key=lambda r: (r[4], r[0]))[:8]:
        print("  变快 %-8s n=%-2d %.2f→%.2f 步  收敛 %.0f%%→%.0f%%"
              % (c, n, old, new, 100 * oc, 100 * nc))
    print("  ...")
    for c, n, old, new, delta, oc, nc in sorted(rows, key=lambda r: (-r[4], r[0]))[:3]:
        if delta > 0:
            print("  变慢 %-8s n=%-2d %.2f→%.2f 步  收敛 %.0f%%→%.0f%%"
                  % (c, n, old, new, 100 * oc, 100 * nc))

    olds = [r[2] for r in rows]
    news = [r[3] for r in rows]
    ocs = [r[5] for r in rows]
    ncs = [r[6] for r in rows]
    print()
    print("  旧策略  平均 %.2f 步 | 中位 %.2f | 平均收敛率 %.0f%%"
          % (statistics.mean(olds), statistics.median(olds), 100 * statistics.mean(ocs)))
    print("  新策略  平均 %.2f 步 | 中位 %.2f | 平均收敛率 %.0f%%"
          % (statistics.mean(news), statistics.median(news), 100 * statistics.mean(ncs)))
    if statistics.mean(olds) > 0:
        print("  降幅 %.1f%%" % (100 * (1 - statistics.mean(news) / statistics.mean(olds))))
    better = sum(1 for r in rows if r[4] < -1e-9)
    worse = sum(1 for r in rows if r[4] > 1e-9)
    print("  逐码：变快 %d / 变慢 %d / 持平 %d" % (better, worse, len(rows) - better - worse))
    if min(statistics.mean(ocs), statistics.mean(ncs)) < 0.8:
        print("  ⚠ 收敛率偏低，步数可能被假设条数封顶")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
