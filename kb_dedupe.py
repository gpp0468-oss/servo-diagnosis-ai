# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# 模块说明
#   职责：字符二元组 Jaccard，检测重复/疑似重复/冲突
#   依赖：os, re
#   输入输出：check_entry() -> [(级别, 来源, 旧条目, 说明)]；
#            单独运行时：python kb_dedupe.py 体检 data/cases.csv
#   调试提示：阈值 0.55；调小更严格
# ---------------------------------------------------------------------------
"""入库前查重与冲突提示（字符二元组 Jaccard）。"""
import os
import re

PUNCT = re.compile(r"[\s，。；、：（）()【】\[\]「」“”\"'.,;:!?]")


def norm(s: str) -> str:
    return PUNCT.sub("", (s or "").lower())


def bigrams(s: str):
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def jaccard(a: str, b: str) -> float:
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def check_entry(entry, cases, feedback, th=0.55):
    """entry: dict(phenomenon, cause, action, confirm)
    返回 [(级别, 来源, 已有条目, 说明)]，级别 ∈ 重复 / 疑似重复 / 冲突。"""
    issues = []
    pool = [("知识库", c) for c in cases] + [("待审经验", f) for f in feedback
                                             if str(f.get("通过", "")).lower() in ("true", "1", "是")]
    for src, old in pool:
        same_p = norm(entry["phenomenon"]) == norm(old.get("phenomenon", ""))
        same_c = norm(entry["cause"]) == norm(old.get("cause", ""))
        if same_p and same_c:
            issues.append(("重复", src, old, "现象与根因完全相同"))
            continue
        c_sim = jaccard(entry["cause"], old.get("cause", ""))
        if c_sim >= th:
            issues.append(("疑似重复", src, old, f"根因相似度 {c_sim:.0%}"))
            continue
        if same_p:
            a_sim = jaccard(entry.get("action", ""), old.get("action", ""))
            if a_sim < 0.2:
                issues.append(("冲突", src, old, f"同一现象但处置不同（措施相似度 {a_sim:.0%}）"))
    return issues


def _main():
    """体检 data/cases.csv：两两比对，每对只计一次。

    与 check_entry() 的差异：这里把「现象相同、原因不同、措施不同」视为
    **正常结构**（同一现象本来就可能对应多条原因，各自处置不同），不计为冲突。
    只有「现象相同 + 原因相近 + 措施却不同」才是真冲突。
    """
    import sys

    try:  # Windows 控制台默认 GBK
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    from csvio import read_rows

    path = "data/cases.csv"
    if not os.path.exists(path):
        print("找不到 %s —— 请在仓库根目录执行本脚本。" % path)
        return 2
    cases = read_rows(path)

    counts = {"重复": 0, "疑似重复": 0, "冲突": 0}
    multi_cause = 0          # 现象相同但原因不同：正常结构
    samples = []

    def _hit(level, desc, a):
        counts[level] = counts.get(level, 0) + 1
        if len(samples) < 5:
            samples.append((level, a.get("phenomenon", "")[:26], desc))

    for i in range(len(cases)):
        a = cases[i]
        for j in range(i + 1, len(cases)):
            b = cases[j]
            same_p = norm(a.get("phenomenon", "")) == norm(b.get("phenomenon", ""))
            same_c = norm(a.get("cause", "")) == norm(b.get("cause", ""))
            if same_p and same_c:
                _hit("重复", "现象与根因完全相同", a)
                continue
            c_sim = jaccard(a.get("cause", ""), b.get("cause", ""))
            if c_sim >= 0.55:
                _hit("疑似重复", "根因相似度 %d%%" % (100 * c_sim), a)
                continue
            a_sim = jaccard(a.get("action", ""), b.get("action", ""))
            if same_p and c_sim >= 0.2 and a_sim < 0.2:
                _hit("冲突", "同一现象、原因相近但处置不同（措施相似度 %d%%）" % (100 * a_sim), a)
            elif same_p:
                multi_cause += 1

    print("kb_dedupe: %s 共 %d 条，两两比对 %d 对"
          % (path, len(cases), len(cases) * (len(cases) - 1) // 2))
    print("  重复 %d / 疑似重复 %d / 冲突 %d"
          % (counts["重复"], counts["疑似重复"], counts["冲突"]))
    print("  （另有 %d 对「现象相同、原因不同」，属多原因正常结构，不计为冲突）" % multi_cause)
    for level, ph, desc in samples:
        print("  [%s] %s —— %s" % (level, ph, desc))
    if not any(counts.values()):
        print("  未发现重复或冲突。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
